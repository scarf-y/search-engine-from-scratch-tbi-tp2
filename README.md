# TP2 - Search Engine from Scratch (TBI)

## Overview
This project implements a simple search engine pipeline using BSBI indexing and inverted index retrieval.
The codebase was extended to complete the main TP2 requirements.

## What Has Been Added
### 1) New Bit-Level Compression: Rice Coding
- Added `RicePostings` in `compression.py`.
- Supports:
  - `encode` / `decode` for gap-based postings lists
  - `encode_tf` / `decode_tf` for term frequency lists

### 2) BM25 Scoring
- Added BM25 retrieval in `bsbi.py` via `retrieve_bm25(...)`.
- Uses document length normalization:
  - `dl` from `doc_length`
  - `avdl` (`avg_doc_length`) precomputed and stored in index metadata

### 3) Additional Evaluation Metrics
- Extended `evaluation.py` with:
  - `DCG`
  - `NDCG`
  - `AP` (Average Precision)
- Existing `RBP` metric is still available.

### 4) WAND Top-K Retrieval for BM25
- Added `retrieve_bm25_wand(...)` in `bsbi.py`.
- Inverted index metadata now stores extra per-term statistic (`max_tf`) to compute upper bounds for WAND pruning.

## Project Structure
- `bsbi.py`: indexing orchestration + retrieval methods (TF-IDF, BM25, BM25-WAND)
- `index.py`: inverted index reader/writer and metadata storage
- `compression.py`: postings compression classes (Standard, VBE, Rice)
- `evaluation.py`: retrieval effectiveness evaluation
- `search.py`: sample search script
- `collection/`: document collection
- `queries.txt`, `qrels.txt`: evaluation inputs

## Requirements
- Python 3.x
- `tqdm`

Install dependency:
```bash
pip install tqdm
```

## How To Run
### 1) Build Index
```bash
python bsbi.py
```

Build index with specific compression:
```bash
python bsbi.py --encoding rice
```

### 2) Run Search
Default search (sample queries, TF-IDF):
```bash
python search.py
```

Search with custom mode:
```bash
python search.py --encoding rice --scoring bm25_wand --k 10 --query "lipid metabolism in pregnancy"
```

### 3) Run Evaluation
Default evaluation (TF-IDF):
```bash
python evaluation.py
```

Evaluation with custom mode:
```bash
python evaluation.py --encoding rice --scoring bm25_wand --k 1000
```

## BM25 / WAND Usage Example
```python
from bsbi import BSBIIndex
from compression import VBEPostings

bsbi = BSBIIndex(data_dir="collection", output_dir="index", postings_encoding=VBEPostings)

print(bsbi.retrieve_bm25("lipid metabolism in pregnancy", k=10))
print(bsbi.retrieve_bm25_wand("lipid metabolism in pregnancy", k=10))
```

## Notes
- Re-run indexing (`python bsbi.py`) after metadata schema changes to ensure all statistics are persisted.
- The retrieval/evaluation `--encoding` must match the encoding used when building index files.
- For correctness checks, BM25 and BM25-WAND top-k results can be compared on the same queries.
