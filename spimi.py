import argparse
import contextlib
import os

from tqdm import tqdm

from bsbi import BSBIIndex, ENCODINGS
from index import InvertedIndexReader, InvertedIndexWriter


class SPIMIIndex(BSBIIndex):
    """
    SPIMI indexer terpisah dari BSBI.
    Tetap menghasilkan format index metadata yang sama agar kompatibel
    dengan pipeline retrieval/evaluation yang sudah ada.
    """

    def _iter_documents(self):
        """
        Iterator dokumen dalam urutan deterministik:
        sort block directory, lalu sort filename di dalam block.
        """
        block_dirs = sorted(next(os.walk(self.data_dir))[1])
        for block_dir_relative in block_dirs:
            block_dir = "./" + self.data_dir + "/" + block_dir_relative
            for filename in sorted(next(os.walk(block_dir))[2]):
                yield block_dir + "/" + filename

    def _flush_chunk(self, term_tf_dict, index_id):
        """
        Flush satu dictionary SPIMI (term -> {doc -> tf}) ke intermediate index.
        """
        self.intermediate_indices.append(index_id)
        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
            for term_id in sorted(term_tf_dict.keys()):
                postings = sorted(term_tf_dict[term_id].keys())
                tf_list = [term_tf_dict[term_id][doc_id] for doc_id in postings]
                index.append(term_id, postings, tf_list)

    def index(self, docs_per_chunk=100):
        """
        Build index dengan SPIMI.
        docs_per_chunk mengontrol kapan in-memory dictionary di-flush ke disk.
        """
        if docs_per_chunk <= 0:
            raise ValueError("docs_per_chunk harus > 0")

        os.makedirs(self.output_dir, exist_ok=True)

        self.intermediate_indices = []
        term_tf_dict = {}
        docs_in_chunk = 0
        chunk_id = 0

        document_paths = list(self._iter_documents())
        for docname in tqdm(document_paths, desc="SPIMI indexing"):
            doc_id = self.doc_id_map[docname]
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in self._process_text(f.read(), remove_stopwords=True):
                    term_id = self.term_id_map[token]
                    if term_id not in term_tf_dict:
                        term_tf_dict[term_id] = {}
                    if doc_id not in term_tf_dict[term_id]:
                        term_tf_dict[term_id][doc_id] = 0
                    term_tf_dict[term_id][doc_id] += 1

            docs_in_chunk += 1
            if docs_in_chunk >= docs_per_chunk:
                self._flush_chunk(term_tf_dict, f"intermediate_spimi_{chunk_id}")
                term_tf_dict = {}
                docs_in_chunk = 0
                chunk_id += 1

        if term_tf_dict:
            self._flush_chunk(term_tf_dict, f"intermediate_spimi_{chunk_id}")

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            if self.intermediate_indices:
                with contextlib.ExitStack() as stack:
                    indices = [
                        stack.enter_context(
                            InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir)
                        )
                        for index_id in self.intermediate_indices
                    ]
                    self.merge(indices, merged_index)
        self.build_and_save_positional_index()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build inverted index with SPIMI.")
    parser.add_argument("--data-dir", default="collection", help="Path to collection directory")
    parser.add_argument("--output-dir", default="index", help="Path to index output directory")
    parser.add_argument("--index-name", default="main_index", help="Main merged index file name")
    parser.add_argument(
        "--encoding",
        choices=sorted(ENCODINGS.keys()),
        default="vbe",
        help="Postings encoding algorithm"
    )
    parser.add_argument(
        "--docs-per-chunk",
        type=int,
        default=100,
        help="How many documents to accumulate before flushing one SPIMI block"
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    spimi = SPIMIIndex(
        data_dir=args.data_dir,
        postings_encoding=ENCODINGS[args.encoding],
        output_dir=args.output_dir,
        index_name=args.index_name
    )
    spimi.index(docs_per_chunk=args.docs_per_chunk)
