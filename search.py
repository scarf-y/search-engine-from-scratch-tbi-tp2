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
        choices=["tfidf", "bm25", "bm25_wand"],
        default="tfidf",
        help="Scoring method"
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k results")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument("--data-dir", default="collection", help="Path to collection directory")
    parser.add_argument("--output-dir", default="index", help="Path to index directory")
    parser.add_argument("--index-name", default="main_index", help="Merged index name")
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

    queries = args.query if args.query else DEFAULT_QUERIES

    for query in queries:
        print("Query  : ", query)
        print("Results:")
        if args.scoring == "tfidf":
            results = bsbi.retrieve_tfidf(query, k=args.k)
        elif args.scoring == "bm25":
            results = bsbi.retrieve_bm25(query, k=args.k, k1=args.k1, b=args.b)
        else:
            results = bsbi.retrieve_bm25_wand(query, k=args.k, k1=args.k1, b=args.b)

        for (score, doc) in results:
            print(f"{doc:30} {score:>.3f}")
        print()


if __name__ == "__main__":
    parser = build_arg_parser()
    run_search(parser.parse_args())
