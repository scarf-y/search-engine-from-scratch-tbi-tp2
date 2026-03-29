import argparse
import math
import re

from bsbi import BSBIIndex
from compression import StandardPostings, VBEPostings, RicePostings

ENCODINGS = {
  "standard": StandardPostings,
  "vbe": VBEPostings,
  "rice": RicePostings
}

SCORINGS = ("tfidf", "bm25", "bm25_wand")

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan

      Returns
      -------
      Float
        score RBP
  """
  score = 0.0
  for i in range(1, len(ranking) + 1):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
  """
  Menghitung Discounted Cumulative Gain (DCG) untuk ranking biner.
  Definisi:
      DCG = rel_1 + sum_{i=2..k} rel_i / log2(i)
  """
  if not ranking:
    return 0.0

  score = float(ranking[0])
  for i in range(2, len(ranking) + 1):
    score += ranking[i - 1] / math.log2(i)
  return score

def ndcg(ranking):
  """
  Menghitung Normalized DCG:
      NDCG = DCG / IDCG
  dengan IDCG adalah DCG dari ranking ideal.
  """
  actual_dcg = dcg(ranking)
  ideal_ranking = sorted(ranking, reverse = True)
  ideal_dcg = dcg(ideal_ranking)
  if ideal_dcg == 0:
    return 0.0
  return actual_dcg / ideal_dcg

def average_precision(ranking, n_relevant):
  """
  Menghitung Average Precision (AP) untuk ranking biner.
  AP = rata-rata precision@i pada semua posisi i dimana rel_i = 1.
  """
  if n_relevant == 0:
    return 0.0

  rel_so_far = 0
  precision_sum = 0.0
  for i in range(1, len(ranking) + 1):
    if ranking[i - 1] == 1:
      rel_so_far += 1
      precision_sum += rel_so_far / i
  return precision_sum / n_relevant

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels)
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.
  """
  qrels = {"Q" + str(i): {i: 0 for i in range(1, max_doc_id + 1)}
           for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      if qid in qrels and 1 <= did <= max_doc_id:
        qrels[qid][did] = 1
  return qrels

def _extract_doc_id(doc_path):
  """
  Ambil numeric doc id dari path dokumen:
  ./collection/11/1032.txt -> 1032
  """
  match = re.search(r'(?:/|\\)(\d+)\.txt$', doc_path)
  if not match:
    raise ValueError(f"Tidak bisa ekstrak doc id dari path: {doc_path}")
  return int(match.group(1))

def _retrieve(bsbi_instance, scoring, query, k, k1, b):
  if scoring == "tfidf":
    return bsbi_instance.retrieve_tfidf(query, k = k)
  if scoring == "bm25":
    return bsbi_instance.retrieve_bm25(query, k = k, k1 = k1, b = b)
  return bsbi_instance.retrieve_bm25_wand(query, k = k, k1 = k1, b = b)

######## >>>>> EVALUASI !

def eval(qrels,
         query_file = "queries.txt",
         k = 1000,
         scoring = "tfidf",
         postings_encoding = VBEPostings,
         data_dir = "collection",
         output_dir = "index",
         index_name = "main_index",
         k1 = 1.2,
         b = 0.75,
         rbp_p = 0.8):
  """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-k documents
  """
  bsbi_instance = BSBIIndex(data_dir = data_dir,
                            postings_encoding = postings_encoding,
                            output_dir = output_dir,
                            index_name = index_name)

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []

    for qline in file:
      parts = qline.strip().split()
      if not parts:
        continue
      qid = parts[0]
      query = " ".join(parts[1:])

      ranking = []
      for (score, doc) in _retrieve(bsbi_instance, scoring, query, k, k1, b):
        did = _extract_doc_id(doc)
        ranking.append(qrels[qid][did])

      # Jika retrieval mengembalikan kurang dari k dokumen, sisanya diasumsikan non-relevan
      if len(ranking) < k:
        ranking.extend([0] * (k - len(ranking)))
      else:
        ranking = ranking[:k]

      n_relevant = sum(qrels[qid].values())
      rbp_scores.append(rbp(ranking, p = rbp_p))
      dcg_scores.append(dcg(ranking))
      ndcg_scores.append(ndcg(ranking))
      ap_scores.append(average_precision(ranking, n_relevant))

  scoring_name = scoring.upper()
  print(f"Hasil evaluasi {scoring_name} terhadap 30 queries")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
  print("AP score =", sum(ap_scores) / len(ap_scores))

def build_arg_parser():
  parser = argparse.ArgumentParser(description="Evaluate retrieval effectiveness.")
  parser.add_argument("--qrels-file", default="qrels.txt", help="Path to qrels file")
  parser.add_argument("--query-file", default="queries.txt", help="Path to query file")
  parser.add_argument("--max-q-id", type=int, default=30, help="Maximum query ID in qrels")
  parser.add_argument("--max-doc-id", type=int, default=1033, help="Maximum doc ID in qrels")
  parser.add_argument("--k", type=int, default=1000, help="Top-k cutoff for evaluation")
  parser.add_argument(
    "--encoding",
    choices=sorted(ENCODINGS.keys()),
    default="vbe",
    help="Postings encoding used by index"
  )
  parser.add_argument(
    "--scoring",
    choices=SCORINGS,
    default="tfidf",
    help="Scoring function"
  )
  parser.add_argument("--data-dir", default="collection", help="Path to collection directory")
  parser.add_argument("--output-dir", default="index", help="Path to index directory")
  parser.add_argument("--index-name", default="main_index", help="Merged index name")
  parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter")
  parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
  parser.add_argument("--rbp-p", type=float, default=0.8, help="RBP persistence parameter")
  return parser

if __name__ == '__main__':
  parser = build_arg_parser()
  args = parser.parse_args()

  qrels = load_qrels(qrel_file = args.qrels_file,
                     max_q_id = args.max_q_id,
                     max_doc_id = args.max_doc_id)

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels,
       query_file = args.query_file,
       k = args.k,
       scoring = args.scoring,
       postings_encoding = ENCODINGS[args.encoding],
       data_dir = args.data_dir,
       output_dir = args.output_dir,
       index_name = args.index_name,
       k1 = args.k1,
       b = args.b,
       rbp_p = args.rbp_p)
