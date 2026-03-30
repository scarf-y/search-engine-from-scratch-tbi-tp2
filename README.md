# TP2 - Search Engine from Scratch (TBI)

## Author
- Name: Daffa Naufal Rahadian
- NPM: 2306213003

## Project Summary
This repository contains a complete mini search engine pipeline for TP2:
- Inverted indexing with `BSBI` and `SPIMI`
- Multiple postings compressions (`standard`, `vbe`, `rice`)
- Ranked retrieval (`TF-IDF`, `BM25`, `BM25 + WAND`, `Adaptive`)
- Structured retrieval (`phrase`, `proximity`, `boolean`)
- Query support features (spell correction, snippets, PRF expansion, FST prefix suggestion)
- Evaluation metrics (`RBP`, `DCG`, `NDCG`, `AP`)

## What You Can Do Right Now
### Main assignment features
- Bit-level compression with Rice coding (`RicePostings` in `compression.py`)
- BM25 scoring with document length normalization
- WAND top-k retrieval for BM25
- Evaluation with DCG, NDCG, AP (plus existing RBP)

### Bonus features implemented
- SPIMI indexing mode (`spimi.py`)
- FST term dictionary (`fst.py`) + prefix suggestions
- Adaptive retrieval strategy
- Positional index for phrase/proximity retrieval
- Boolean query parser (`AND`, `OR`, `NOT`, parenthesis, phrase clause)
- Query spell correction (FST candidate + edit distance)
- Result snippets with term highlighting
- Pseudo relevance feedback (PRF) query expansion

### Text preprocessing and normalization
Applied consistently in indexing and query handling:
- lowercasing
- regex tokenization
- stopword removal (English list)
- simple suffix-based stemming

## Repository Layout
- `bsbi.py`: BSBI indexing and all retrieval methods
- `spimi.py`: SPIMI indexer (compatible output format)
- `index.py`: inverted index reader/writer and metadata
- `compression.py`: `StandardPostings`, `VBEPostings`, `RicePostings`
- `fst.py`: finite-state term dictionary and prefix lookup
- `search.py`: CLI retrieval runner
- `evaluation.py`: evaluation script and metrics
- `util.py`: helper structures/utilities
- `collection/`: document collection
- `queries.txt`, `qrels.txt`: evaluation inputs
- `index/`, `tmp/`: output/work directories

## Index Artifacts Generated
After indexing, the output directory (`--output-dir`, default `index`) contains:
- `<index_name>.index`: compressed postings bytes
- `<index_name>.dict`: dictionary metadata (`postings_dict`, term order, `doc_length`, `avg_doc_length`)
- `<index_name>.pos`: positional index for phrase/proximity
- `terms.dict`: serialized `term_id_map`
- `docs.dict`: serialized `doc_id_map`
- `terms.fst`: serialized FST dictionary
- intermediate index files (during build/merge)

## Requirements
- Python 3.x
- `tqdm`

Install dependency:
```bash
pip install tqdm
```

## End-to-End Workflow
### 1) Build index (choose one indexer)
BSBI:
```bash
python bsbi.py --data-dir collection --output-dir index --index-name main_index --encoding vbe
```

SPIMI:
```bash
python spimi.py --data-dir collection --output-dir index --index-name main_index --encoding vbe --docs-per-chunk 100
```

Notes:
- Running BSBI and SPIMI to the same `--output-dir`/`--index-name` will overwrite previous index outputs.
- Retrieval and evaluation must use the same encoding used at indexing.

### 2) Run search
Basic BM25 query:
```bash
python search.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --query "lipid metabolism in toxemia and normal pregnancy" --k 10
```

### 3) Evaluate effectiveness
```bash
python evaluation.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --k 1000
```

## Search Modes and Examples
### Ranked retrieval
TF-IDF:
```bash
python search.py --encoding vbe --scoring tfidf --query "lipid metabolism pregnancy" --k 10
```

BM25:
```bash
python search.py --encoding vbe --scoring bm25 --k1 1.2 --b 0.75 --query "lipid metabolism pregnancy" --k 10
```

BM25 + WAND:
```bash
python search.py --encoding vbe --scoring bm25_wand --query "lipid metabolism pregnancy" --k 10
```

Adaptive:
```bash
python search.py --encoding vbe --scoring adaptive --query "lipid metabolism pregnancy" --k 10
```

### Structured retrieval
Phrase:
```bash
python search.py --encoding vbe --scoring phrase --query "lipid metabolism" --k 10
```

Proximity (distance <= N):
```bash
python search.py --encoding vbe --scoring proximity --proximity-distance 3 --query "lipid pregnancy" --k 10
```

Boolean with ranking backend:
```bash
python search.py --encoding vbe --scoring boolean --boolean-base bm25 --query "(lipid OR pregnancy) AND NOT toxemia" --k 10
```

Boolean with phrase clause on PowerShell:
```bash
python --% search.py --encoding vbe --scoring boolean --boolean-base bm25 --query "\"lipid metabolism\" AND pregnancy" --k 10
```

## Query Assistance Features
Spell correction:
```bash
python search.py --encoding vbe --scoring bm25 --spell-correct --max-edit-distance 2 --query "lipiid metabolisim pregnncy" --k 10
```

Snippets and highlights:
```bash
python search.py --encoding vbe --scoring bm25 --with-snippet --snippet-window 8 --snippet-max-chars 220 --query "lipid metabolism pregnancy" --k 5
```

Pseudo relevance feedback (PRF):
```bash
python search.py --encoding vbe --scoring bm25 --prf --prf-docs 5 --expand-terms 3 --query "lipid metabolism pregnancy" --k 10
```

FST prefix suggestions:
```bash
python search.py --encoding vbe --suggest-prefix met --suggest-limit 10 --query "lipid" --k 1
```

## Complete CLI Reference
### `bsbi.py`
```text
python bsbi.py [--data-dir] [--output-dir] [--index-name] [--encoding {rice,standard,vbe}]
```

### `spimi.py`
```text
python spimi.py [--data-dir] [--output-dir] [--index-name] [--encoding {rice,standard,vbe}] [--docs-per-chunk]
```

### `search.py`
```text
python search.py
  [--encoding {rice,standard,vbe}]
  [--scoring {tfidf,bm25,bm25_wand,adaptive,phrase,proximity,boolean}]
  [--k K] [--k1 K1] [--b B]
  [--boolean-base {bm25,tfidf}]
  [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--index-name INDEX_NAME]
  [--proximity-distance N]
  [--spell-correct] [--max-edit-distance N]
  [--with-snippet] [--snippet-window N] [--snippet-max-chars N]
  [--prf] [--prf-docs N] [--expand-terms N]
  [--suggest-prefix PREFIX] [--suggest-limit N]
  [--query QUERY]  # can be repeated
```

### `evaluation.py`
```text
python evaluation.py
  [--qrels-file FILE] [--query-file FILE]
  [--max-q-id N] [--max-doc-id N]
  [--k K]
  [--encoding {rice,standard,vbe}]
  [--scoring {tfidf,bm25,bm25_wand,adaptive}]
  [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--index-name INDEX_NAME]
  [--k1 K1] [--b B] [--rbp-p P]
```

## Evaluation Metrics
`evaluation.py` reports mean scores across all queries in `queries.txt`:
- `RBP`
- `DCG`
- `NDCG`
- `AP` (Average Precision)

## Compression Options
Use one of:
- `standard`: no advanced compression
- `vbe`: Variable-Byte Encoding
- `rice`: Rice coding (bit-level)

Example rebuild with Rice:
```bash
python bsbi.py --encoding rice
```

Then retrieve with the same encoding:
```bash
python search.py --encoding rice --scoring bm25 --query "lipid metabolism pregnancy"
```

## Practical Notes
- Rebuild index after changes in preprocessing/indexing logic.
- Keep encoding consistent across indexing, search, and evaluation.
- WAND/Adaptive may produce similar effectiveness scores to BM25 because all use BM25 scoring, but runtime behavior can differ.
- If boolean phrase queries fail on PowerShell due quoting, use `python --% ...` form.

## Quick Health Check Commands
```bash
python -m py_compile bsbi.py spimi.py search.py evaluation.py index.py compression.py fst.py
python bsbi.py --encoding vbe --output-dir index --index-name main_index
python search.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --query "lipid metabolism in toxemia and normal pregnancy" --k 5
python evaluation.py --encoding vbe --output-dir index --index-name main_index --scoring bm25 --k 100
```
