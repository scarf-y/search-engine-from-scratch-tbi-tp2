import os
import pickle
import contextlib
import heapq
import time
import math
import argparse

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, RicePostings
from fst import FSTDictionary
from tqdm import tqdm

ENCODINGS = {
    "standard": StandardPostings,
    "vbe": VBEPostings,
    "rice": RicePostings
}

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.term_fst = None
        self.term_fst_path = os.path.join(self.output_dir, "terms.fst")

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def _build_term_fst(self):
        self.term_fst = FSTDictionary.from_id_to_str(self.term_id_map.id_to_str)

    def _save_term_fst(self):
        if self.term_fst is None:
            self._build_term_fst()
        with open(self.term_fst_path, "wb") as f:
            pickle.dump(self.term_fst, f)

    def _load_term_fst(self):
        if os.path.exists(self.term_fst_path):
            with open(self.term_fst_path, "rb") as f:
                self.term_fst = pickle.load(f)
        else:
            # Backward-compatible jika index lama belum punya terms.fst
            self._build_term_fst()

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        self._save_term_fst()

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        self._load_term_fst()

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    @staticmethod
    def _ensure_postings_tf_alignment(term, postings, tf_list, encoding):
        """
        Validasi bahwa postings dan tf_list punya panjang yang sama.
        Jika tidak sama, kemungkinan besar index dibaca dengan encoding berbeda
        dari encoding saat indexing.
        """
        if len(postings) != len(tf_list):
            raise ValueError(
                f"Postings/TF length mismatch for term_id={term}: "
                f"len(postings)={len(postings)} vs len(tf_list)={len(tf_list)}. "
                "Likely caused by encoding mismatch. "
                f"Please rebuild index using the same encoding ({encoding.__name__}) "
                "or run retrieval with matching --encoding."
            )

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._query_term_ids(query)
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    self._ensure_postings_tf_alignment(term, postings, tf_list, self.postings_encoding)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Melakukan Ranked Retrieval dengan skema TaaT menggunakan BM25.

        Formula:
            score(D, Q) = sum_{t in Q n D} log(N/df_t) *
                          ((k1 + 1) * tf_tD) /
                          (k1 * ((1 - b) + b * (dl / avdl)) + tf_tD)
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._query_term_ids(query)
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            scores = {}
            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avdl = merged_index.avg_doc_length if merged_index.avg_doc_length > 0 else 1.0

            for term in terms:
                if term not in merged_index.postings_dict:
                    continue

                df = merged_index.postings_dict[term][1]
                if df == 0:
                    continue
                idf = math.log(N / df)
                postings, tf_list = merged_index.get_postings_list(term)
                self._ensure_postings_tf_alignment(term, postings, tf_list, self.postings_encoding)

                for i in range(len(postings)):
                    doc_id, tf = postings[i], tf_list[i]
                    if tf <= 0:
                        continue
                    dl = merged_index.doc_length.get(doc_id, 0)
                    denom = k1 * ((1 - b) + b * (dl / avdl)) + tf
                    if denom == 0:
                        continue
                    score = idf * (((k1 + 1) * tf) / denom)
                    if doc_id not in scores:
                        scores[doc_id] = 0.0
                    scores[doc_id] += score

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    @staticmethod
    def _bm25_term_score(tf, df, N, dl, avdl, k1, b):
        """Kontribusi satu term BM25 untuk satu dokumen."""
        if tf <= 0 or df <= 0 or N <= 0:
            return 0.0
        idf = math.log(N / df)
        denom = k1 * ((1 - b) + b * (dl / avdl)) + tf
        if denom == 0:
            return 0.0
        return idf * (((k1 + 1) * tf) / denom)

    @staticmethod
    def _bm25_term_upper_bound(max_tf, df, N, k1, b):
        """
        Upper bound longgar untuk kontribusi term BM25 (dipakai WAND).
        Asumsi kasus terbaik pada normalisasi panjang dokumen (dl -> 0).
        """
        if max_tf <= 0 or df <= 0 or N <= 0:
            return 0.0
        idf = math.log(N / df)
        denom = k1 * (1 - b) + max_tf
        if denom == 0:
            return 0.0
        return idf * (((k1 + 1) * max_tf) / denom)

    def retrieve_bm25_wand(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Melakukan Top-K retrieval BM25 dengan algoritma WAND.
        Menggunakan upper bound kontribusi per-term agar tidak semua kandidat
        dokumen dihitung skornya.
        """
        if k <= 0:
            return []
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = self._query_term_ids(query)
        if not terms:
            return []

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avdl = merged_index.avg_doc_length if merged_index.avg_doc_length > 0 else 1.0

            states = []
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue

                postings, tf_list = merged_index.get_postings_list(term)
                self._ensure_postings_tf_alignment(term, postings, tf_list, self.postings_encoding)
                if not postings:
                    continue

                meta = merged_index.postings_dict[term]
                df = meta[1]
                max_tf = meta[4] if len(meta) > 4 else max(tf_list)
                upper_bound = self._bm25_term_upper_bound(max_tf, df, N, k1, b)
                if upper_bound <= 0:
                    continue

                states.append({
                    "term": term,
                    "df": df,
                    "postings": postings,
                    "tf_list": tf_list,
                    "ptr": 0,
                    "ub": upper_bound
                })

            if not states:
                return []

            topk_heap = []  # min-heap berisi (score, doc_id)
            threshold = 0.0

            while True:
                active_states = [s for s in states if s["ptr"] < len(s["postings"])]
                if not active_states:
                    break

                active_states.sort(key=lambda s: s["postings"][s["ptr"]])

                score_limit = 0.0
                pivot_doc = None
                for state in active_states:
                    score_limit += state["ub"]
                    if score_limit > threshold:
                        pivot_doc = state["postings"][state["ptr"]]
                        break

                if pivot_doc is None:
                    break

                smallest_doc = active_states[0]["postings"][active_states[0]["ptr"]]
                if smallest_doc == pivot_doc:
                    candidate_doc = pivot_doc
                    score = 0.0

                    for state in active_states:
                        while state["ptr"] < len(state["postings"]) and \
                              state["postings"][state["ptr"]] < candidate_doc:
                            state["ptr"] += 1

                        if state["ptr"] < len(state["postings"]) and \
                           state["postings"][state["ptr"]] == candidate_doc:
                            tf = state["tf_list"][state["ptr"]]
                            dl = merged_index.doc_length.get(candidate_doc, 0)
                            score += self._bm25_term_score(
                                tf, state["df"], N, dl, avdl, k1, b
                            )

                    if score > threshold:
                        heapq.heappush(topk_heap, (score, candidate_doc))
                        if len(topk_heap) > k:
                            heapq.heappop(topk_heap)
                        if len(topk_heap) == k:
                            threshold = topk_heap[0][0]

                    for state in active_states:
                        if state["ptr"] < len(state["postings"]) and \
                           state["postings"][state["ptr"]] == candidate_doc:
                            state["ptr"] += 1
                else:
                    lead_state = active_states[0]
                    while lead_state["ptr"] < len(lead_state["postings"]) and \
                          lead_state["postings"][lead_state["ptr"]] < pivot_doc:
                        lead_state["ptr"] += 1

            ranked = sorted(topk_heap, key=lambda x: x[0], reverse=True)
            return [(score, self.doc_id_map[doc_id]) for (score, doc_id) in ranked]

    def _query_term_ids(self, query):
        """
        Konversi query string -> list term_id menggunakan FST dictionary.
        Terms yang tidak ditemukan di-skip.
        """
        if self.term_fst is None:
            self._load_term_fst()

        term_ids = []
        for word in query.split():
            term_id = self.term_fst.lookup(word)
            if term_id is not None:
                term_ids.append(term_id)
        return term_ids

    def suggest_terms(self, prefix, limit=10):
        """
        Ambil kandidat term berdasarkan prefix dari FST.
        """
        if len(self.term_id_map) == 0:
            self.load()
        if self.term_fst is None:
            self._load_term_fst()
        return [term for (term, _) in self.term_fst.prefix_search(prefix, limit=limit)]

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build inverted index with BSBI."
    )
    parser.add_argument("--data-dir", default="collection", help="Path to collection directory")
    parser.add_argument("--output-dir", default="index", help="Path to index output directory")
    parser.add_argument("--index-name", default="main_index", help="Main merged index file name")
    parser.add_argument(
        "--encoding",
        choices=sorted(ENCODINGS.keys()),
        default="vbe",
        help="Postings encoding algorithm"
    )
    args = parser.parse_args()

    BSBI_instance = BSBIIndex(data_dir=args.data_dir,
                              postings_encoding=ENCODINGS[args.encoding],
                              output_dir=args.output_dir,
                              index_name=args.index_name)
    BSBI_instance.index() # memulai indexing!
