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

### 5) Adaptive Retrieval
- Added `retrieve_adaptive(...)` in `bsbi.py`.
- Adaptive strategy selects between:
  - `BM25`
  - `BM25 + WAND`
- Heuristic is based on query length and document frequency profile of query terms.

## Bonus Feature
### SPIMI Indexing Mode (Separate Module)
- Added `spimi.py` as a separate SPIMI indexer.
- SPIMI accumulates in-memory dictionary:
  - `term_id -> {doc_id: tf}`
- The dictionary is flushed to intermediate index files every configurable number of documents, then merged into the final index.
- Output index format is compatible with existing `search.py` and `evaluation.py`.

### FST-Based Term Dictionary
- Added `fst.py` as a finite-state transducer dictionary for term storage and lookup.
- During indexing, term dictionary is also persisted as `terms.fst`.
- During retrieval, query term lookup uses FST.
- Added prefix suggestion feature in `search.py` (`--suggest-prefix`).

Why this is useful:
- Adds a structured automaton-based dictionary representation (bonus requirement).
- Enables efficient prefix traversal for term suggestions.

Why impact may look small in this project:
- The collection is relatively small, so speed/memory differences vs simple dictionary are limited.
- FST mostly improves dictionary organization and prefix-query capability, not ranking metric values (RBP/DCG/NDCG/AP).

### Positional Index + Phrase/Proximity Retrieval
- Added positional index persistence as `<index_name>.pos`.
- Positional index is built after BSBI/SPIMI indexing and stores:
  - `term_id -> {doc_id -> [positions]}`
- Added exact phrase retrieval (`--scoring phrase`).
- Added proximity retrieval (`--scoring proximity` + `--proximity-distance`).

### Query Spell Correction
- Added spelling suggestion and query auto-correction based on FST candidates + Levenshtein edit distance.
- In search CLI, enable via `--spell-correct`.
- Useful when query terms are misspelled (e.g., `pregnacy -> pregnancy`).

### Boolean Query Parser
- Added boolean retrieval mode with support for:
  - `AND`, `OR`, `NOT`
  - parenthesis: `( ... )`
  - phrase clause: `"..."` (e.g., `"lipid metabolism" AND pregnancy`)
- Added scoring backend choice for ranking matched boolean documents:
  - `--boolean-base bm25` (default)
  - `--boolean-base tfidf`

## Project Structure
- `bsbi.py`: indexing orchestration + retrieval methods (TF-IDF, BM25, BM25-WAND, Adaptive, Phrase, Proximity, Boolean)
- `index.py`: inverted index reader/writer and metadata storage
- `compression.py`: postings compression classes (Standard, VBE, Rice)
- `evaluation.py`: retrieval effectiveness evaluation
- `fst.py`: finite-state transducer dictionary for terms (bonus)
- `spimi.py`: separate SPIMI indexing module (bonus)
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

Build index using SPIMI (bonus):
```bash
python spimi.py --encoding rice --docs-per-chunk 100
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

Adaptive retrieval mode:
```bash
python search.py --encoding rice --scoring adaptive --k 10 --query "lipid metabolism in pregnancy"
```

Prefix suggestions from FST:
```bash
python search.py --encoding rice --suggest-prefix lip --suggest-limit 10 --query "lipid metabolism in pregnancy"
```

Exact phrase retrieval:
```bash
python search.py --encoding rice --scoring phrase --k 10 --query "lipid metabolism"
```

Proximity retrieval:
```bash
python search.py --encoding rice --scoring proximity --proximity-distance 3 --k 10 --query "lipid pregnancy"
```

Spell-corrected search:
```bash
python search.py --encoding rice --scoring bm25 --spell-correct --max-edit-distance 2 --k 10 --query "lipd metabolism in pregnacy"
```

Boolean retrieval:
```bash
python search.py --encoding rice --scoring boolean --boolean-base bm25 --k 10 --query "(lipid OR pregnancy) AND NOT toxemia"
```

Boolean retrieval with phrase clause:
```bash
python --% search.py --encoding rice --scoring boolean --boolean-base bm25 --k 10 --query "\"lipid metabolism\" AND pregnancy"
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

Adaptive evaluation:
```bash
python evaluation.py --encoding rice --scoring adaptive --k 1000
```

## BM25 / WAND Usage Example
```python
from bsbi import BSBIIndex
from compression import VBEPostings

bsbi = BSBIIndex(data_dir="collection", output_dir="index", postings_encoding=VBEPostings)

print(bsbi.retrieve_bm25("lipid metabolism in pregnancy", k=10))
print(bsbi.retrieve_bm25_wand("lipid metabolism in pregnancy", k=10))
print(bsbi.retrieve_adaptive("lipid metabolism in pregnancy", k=10))
```

## Runtime Benchmark Snippet
```python
import time
from bsbi import BSBIIndex
from compression import VBEPostings

bsbi = BSBIIndex(data_dir="collection", output_dir="index", postings_encoding=VBEPostings)

queries = []
with open("queries.txt", encoding="utf8") as f:
    for line in f:
        parts = line.strip().split()
        if parts:
            queries.append(" ".join(parts[1:]))

def bench(name, fn):
    t0 = time.perf_counter()
    for q in queries:
        fn(q)
    dt = time.perf_counter() - t0
    print(f"{name}: total={dt:.6f}s avg={dt/len(queries):.6f}s")

k = 200
bench("BM25", lambda q: bsbi.retrieve_bm25(q, k=k))
bench("BM25_WAND", lambda q: bsbi.retrieve_bm25_wand(q, k=k))
bench("ADAPTIVE", lambda q: bsbi.retrieve_adaptive(q, k=k))
```

Example result on this collection (one run):
- `BM25: total=0.556743s avg=0.018558s`
- `BM25_WAND: total=0.687092s avg=0.022903s`
- `ADAPTIVE: total=1.033690s avg=0.034456s`

## Notes
- Re-run indexing (`python bsbi.py`) after metadata schema changes to ensure all statistics are persisted.
- `spimi.py` writes to `index` by default (same as `bsbi.py`), so running one indexer after another will overwrite index files.
- The retrieval/evaluation `--encoding` must match the encoding used when building index files.
- FST term suggestions are meant as dictionary capability demo and do not directly optimize ranking quality metrics.
- Phrase and proximity retrieval depend on positional index file (`<index_name>.pos`), so indexing must be run first.
- BM25, BM25-WAND, and Adaptive can produce identical effectiveness scores because they use the same BM25 scoring function; differences are mostly in runtime behavior.
- On PowerShell, boolean phrase queries with inner quotes are safest with `python --% ...`.
