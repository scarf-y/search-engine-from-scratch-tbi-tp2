# TP2 - Search Engine from Scratch (TBI)

## Author
- Name: Daffa Naufal Rahadian
- NPM: 2306213003

## Overview
This project is an end-to-end search engine implementation for TP2, including indexing, compression, ranked retrieval, boolean/phrase/proximity retrieval, evaluation metrics, and several bonus features.

## Main Requirements Implemented
1. Bit-level compression
- Added Rice coding (`RicePostings`) in `compression.py` for postings and TF list encoding/decoding.

2. BM25 scoring
- Added BM25 retrieval with document-length normalization.
- Index metadata stores `doc_length` and `avg_doc_length` for BM25.

3. Evaluation metrics
- Added `DCG`, `NDCG`, and `AP` in `evaluation.py`.
- Existing `RBP` is still available.

4. WAND top-k retrieval
- Added BM25 + WAND (`retrieve_bm25_wand`) to avoid scoring every candidate document.

## Bonus Features Implemented
1. SPIMI indexing mode
- Added separate indexer in `spimi.py`.
- Output format is compatible with the same `search.py` and `evaluation.py` pipeline.

2. FST dictionary
- Added `fst.py` and persisted `terms.fst`.
- Used for term lookup and prefix suggestions.

3. Adaptive retrieval
- Added `retrieve_adaptive` to switch between BM25 and BM25+WAND based on query characteristics.

4. Positional index + phrase/proximity retrieval
- Added positional index file (`<index_name>.pos`).
- Added exact phrase and proximity search modes.

5. Query spell correction
- Added Levenshtein-based query correction using FST candidate terms.

6. Boolean query parser
- Supports `AND`, `OR`, `NOT`, parenthesis, and phrase clauses.

7. Snippets and pseudo relevance feedback (PRF)
- Optional snippet rendering with highlight.
- Optional PRF query expansion.

## Text Preprocessing (Now Applied)
Normalization is applied consistently in BSBI and SPIMI indexing and retrieval query processing:
- Lowercasing
- Regex tokenization
- Stopword removal (English stopword list)
- Simple suffix-based stemming

This normalization is used in:
- Index building (`bsbi.py`, `spimi.py`)
- Query term lookup for retrieval
- Boolean term extraction/scoring terms
- Spell correction input normalization
- Snippet query-term matching
- PRF term collection and expansion

## Project Structure
- `bsbi.py`: BSBI indexing and retrieval methods
- `spimi.py`: SPIMI indexing mode
- `compression.py`: Standard, VBE, Rice postings compression
- `index.py`: inverted index reader/writer + metadata
- `fst.py`: FST dictionary implementation
- `search.py`: CLI for retrieval and search features
- `evaluation.py`: evaluation pipeline and metrics
- `collection/`, `queries.txt`, `qrels.txt`: dataset and evaluation inputs

## Installation
```bash
pip install tqdm
```

## Usage
### 1) Build index (BSBI)
```bash
python bsbi.py --encoding vbe --output-dir index --index-name main_index
```

### 2) Build index (SPIMI)
```bash
python spimi.py --encoding vbe --output-dir index --index-name main_index --docs-per-chunk 100
```

### 3) Search examples
BM25:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --query "lipid metabolism in toxemia and normal pregnancy" --k 10
```

BM25 + WAND:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring bm25_wand --query "lipid metabolism in toxemia and normal pregnancy" --k 10
```

Adaptive:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring adaptive --query "lipid metabolism in toxemia and normal pregnancy" --k 10
```

Phrase:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring phrase --query "lipid metabolism" --k 10
```

Proximity:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring proximity --proximity-distance 3 --query "lipid pregnancy" --k 10
```

Boolean (PowerShell-safe):
```bash
python --% search.py --encoding vbe --output-dir index --index-name main_index --scoring boolean --boolean-base bm25 --query "\"lipid metabolism\" AND pregnancy" --k 10
```

With spell correction:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --spell-correct --query "lipiid metabolisim pregnncy" --k 10
```

With snippets:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --with-snippet --query "lipid metabolism pregnancy" --k 5
```

With PRF:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --prf --prf-docs 5 --expand-terms 3 --query "lipid metabolism pregnancy" --k 10
```

FST prefix suggestion:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --suggest-prefix met --suggest-limit 10 --query "lipid" --k 1
```

### 4) Evaluation
```bash
python evaluation.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --k 1000
```

Other available scoring values:
- `tfidf`
- `bm25`
- `bm25_wand`
- `adaptive`

## Notes
- Re-run indexing after major indexing/preprocessing changes.
- Retrieval/evaluation `--encoding` must match index build encoding.
- `bsbi.py` and `spimi.py` both write to the same output directory by default; running one after another overwrites index files.
- BM25, BM25_WAND, and Adaptive can have very similar effectiveness metrics because they share BM25 scoring; differences are mainly runtime behavior.
