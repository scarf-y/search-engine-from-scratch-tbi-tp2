import argparse

from bsbi import BSBIIndex
from compression import StandardPostings, VBEPostings, RicePostings

ENCODINGS = {
    "standard": StandardPostings,
    "vbe": VBEPostings,
    "rice": RicePostings
}

DEFAULT_QUERIES = [
    "alkylated with radioactive iodoacetate",
    "psychodrama for disturbed children",
    "lipid metabolism in toxemia and normal pregnancy"
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run retrieval for sample/query inputs.")
    parser.add_argument(
        "--encoding",
        choices=sorted(ENCODINGS.keys()),
        default="vbe",
        help="Postings encoding used by the built index"
    )
    parser.add_argument(
        "--scoring",
        choices=["tfidf", "bm25", "bm25_wand", "adaptive", "phrase", "proximity", "boolean"],
        default="tfidf",
        help="Scoring method"
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k results")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--boolean-base", choices=["bm25", "tfidf"], default="bm25",
                        help="Base scorer for boolean retrieval ranking")
    parser.add_argument("--data-dir", default="collection", help="Path to collection directory")
    parser.add_argument("--output-dir", default="index", help="Path to index directory")
    parser.add_argument("--index-name", default="main_index", help="Merged index name")
    parser.add_argument("--proximity-distance", type=int, default=3, help="Max distance for proximity retrieval")
    parser.add_argument("--spell-correct", action="store_true", help="Enable query spell correction")
    parser.add_argument("--max-edit-distance", type=int, default=2, help="Max edit distance for spell correction")
    parser.add_argument("--with-snippet", action="store_true", help="Show result snippets with query-term highlights")
    parser.add_argument("--snippet-window", type=int, default=8, help="Context window size for snippets")
    parser.add_argument("--snippet-max-chars", type=int, default=220, help="Maximum snippet length in characters")
    parser.add_argument("--prf", action="store_true", help="Enable pseudo relevance feedback query expansion")
    parser.add_argument("--prf-docs", type=int, default=5, help="Top docs used for PRF expansion")
    parser.add_argument("--expand-terms", type=int, default=3, help="Number of expansion terms for PRF")
    parser.add_argument("--suggest-prefix", help="Prefix for FST term suggestions")
    parser.add_argument("--suggest-limit", type=int, default=10, help="Maximum suggestion count")
    parser.add_argument(
        "--query",
        action="append",
        help="Query string (can be used multiple times). If omitted, built-in sample queries are used."
    )
    return parser


def run_search(args):
    bsbi = BSBIIndex(data_dir=args.data_dir,
                     postings_encoding=ENCODINGS[args.encoding],
                     output_dir=args.output_dir,
                     index_name=args.index_name)

    if args.suggest_prefix:
        suggestions = bsbi.suggest_terms(args.suggest_prefix, limit=args.suggest_limit)
        print(f"Suggestions for prefix '{args.suggest_prefix}':")
        for term in suggestions:
            print(term)
        print()

    queries = args.query if args.query else DEFAULT_QUERIES

    for query in queries:
        query_to_use = query
        if args.spell_correct:
            corrected_query, corrections = bsbi.correct_query(
                query,
                max_edit_distance=args.max_edit_distance,
                top_n=1
            )
            if corrections:
                print(f"Spell-corrected query: {corrected_query}")
                for original, suggested in corrections:
                    print(f"  {original} -> {suggested}")
            query_to_use = corrected_query

        if args.prf:
            prf_scoring = args.scoring if args.scoring in ("tfidf", "bm25", "bm25_wand", "adaptive") else "bm25"
            expanded_query, added_terms = bsbi.expand_query_prf(
                query_to_use,
                top_docs=args.prf_docs,
                expand_terms=args.expand_terms,
                scoring=prf_scoring,
                k1=args.k1,
                b=args.b
            )
            if added_terms:
                print(f"PRF expanded query: {expanded_query}")
                print(f"  added terms: {', '.join(added_terms)}")
            query_to_use = expanded_query

        print("Query  : ", query)
        print("Results:")
        if args.scoring == "tfidf":
            results = bsbi.retrieve_tfidf(query_to_use, k=args.k)
        elif args.scoring == "bm25":
            results = bsbi.retrieve_bm25(query_to_use, k=args.k, k1=args.k1, b=args.b)
        elif args.scoring == "bm25_wand":
            results = bsbi.retrieve_bm25_wand(query_to_use, k=args.k, k1=args.k1, b=args.b)
        elif args.scoring == "adaptive":
            results = bsbi.retrieve_adaptive(query_to_use, k=args.k, k1=args.k1, b=args.b)
        elif args.scoring == "phrase":
            results = bsbi.retrieve_phrase(query_to_use, k=args.k)
        elif args.scoring == "proximity":
            results = bsbi.retrieve_proximity(
                query_to_use,
                max_distance=args.proximity_distance,
                k=args.k
            )
        else:  # boolean
            try:
                results = bsbi.retrieve_boolean(
                    query_to_use,
                    k=args.k,
                    base_scoring=args.boolean_base,
                    k1=args.k1,
                    b=args.b
                )
            except ValueError as exc:
                print(f"Boolean query error: {exc}")
                print()
                continue

        for (score, doc) in results:
            print(f"{doc:30} {score:>.3f}")
            if args.with_snippet:
                snippet = bsbi.build_snippet(
                    doc,
                    query_to_use,
                    window=args.snippet_window,
                    max_chars=args.snippet_max_chars
                )
                if snippet:
                    print(f"  {snippet}")
        print()


if __name__ == "__main__":
    parser = build_arg_parser()
    run_search(parser.parse_args())
