import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings

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
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-k documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection',
                            postings_encoding = VBEPostings,
                            output_dir = 'index')

  with open(query_file) as file:
    rbp_scores = []
    dcg_scores = []
    ndcg_scores = []
    ap_scores = []

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
        did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
        ranking.append(qrels[qid][did])

      # Jika retrieval mengembalikan kurang dari k dokumen, sisanya diasumsikan non-relevan
      if len(ranking) < k:
        ranking.extend([0] * (k - len(ranking)))
      else:
        ranking = ranking[:k]

      n_relevant = sum(qrels[qid].values())
      rbp_scores.append(rbp(ranking))
      dcg_scores.append(dcg(ranking))
      ndcg_scores.append(ndcg(ranking))
      ap_scores.append(average_precision(ranking, n_relevant))

  print("Hasil evaluasi TF-IDF terhadap 30 queries")
  print("RBP score =", sum(rbp_scores) / len(rbp_scores))
  print("DCG score =", sum(dcg_scores) / len(dcg_scores))
  print("NDCG score =", sum(ndcg_scores) / len(ndcg_scores))
  print("AP score =", sum(ap_scores) / len(ap_scores))

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels)
