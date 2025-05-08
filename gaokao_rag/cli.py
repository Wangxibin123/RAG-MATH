import argparse
import json
import sys
from pathlib import Path # Added for path operations

# Use a try-except block for retriever import for robustness during early setup
try:
    from .retriever import build_index, query, STORE, CFG # expose STORE and CFG for CLI access
    if CFG.store_name == 'faiss':
        from .store.faiss import FaissStore
except ImportError as e:
    print(f"Warning: Could not import from .retriever or .store: {e}")
    print("CLI functionality might be limited until project is fully set up and installed.")
    # Define dummy functions if import fails, so script can still be parsed by argparser
    def build_index():
        print("Error: build_index not available. Check project setup.")
    def query(stem, k):
        print("Error: query not available. Check project setup.")
        return []
    STORE = None
    CFG = None
    FaissStore = None # Placeholder

def main():
    parser = argparse.ArgumentParser(description="Gaokao-RAG CLI for importing, querying, and managing indexes.")
    subparsers = parser.add_subparsers(dest="cmd", help="Available commands", required=True)

    subparsers.add_parser("import-text",   help="只构建文本索引")
    qt = subparsers.add_parser("query-text", help="纯文本相似度检索")
    qt.add_argument("stem"); qt.add_argument("-k", "--k", type=int, default=10)   # ★ 同时支持 -k/--k

    # Import command
    import_parser = subparsers.add_parser("import", help="Import data and build the index.")
    # No arguments needed for import for now, but can be added later (e.g., --file)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query for math problems.")
    query_parser.add_argument("stem", type=str, help="The math problem stem to query for.")
    query_parser.add_argument("-k", "--k", type=int, default=10, 
                              help="Number of results to return (defaults to config).")
    
    if FaissStore: # Only add dump/load if FaissStore is available
        dump_parser = subparsers.add_parser("dump", help="Dump the FAISS index and ID map to a file.")
        dump_parser.add_argument("--output-path", type=str, default="models/faiss_dump/gaokao_index.bin",
                                 help="Base path to save the FAISS index and map. (e.g., my_index.bin will save my_index.bin and my_index.bin.map)")

        load_parser = subparsers.add_parser("load", help="Load a FAISS index and ID map from a file.")
        load_parser.add_argument("--input-path", type=str, default="models/faiss_dump/gaokao_index.bin",
                                 help="Base path to load the FAISS index and map from.")

    args = parser.parse_args()

    if args.cmd == "import":
        print("Starting data import and index building...")
        build_index()
        print("Import and index building process finished.")
    elif args.cmd == "query":
        # If k is not provided via CLI, it will use the default from CFG in query function
        results = query(args.stem, args.k)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    # else: # Not needed because subparsers(required=True) handles no command
    #     parser.print_help()
    elif args.cmd == "dump":
        if CFG and CFG.store_name == 'faiss' and STORE and isinstance(STORE, FaissStore):
            print(f"Dumping FAISS index to: {args.output_path} (and .map)")
            # Ensure output directory exists
            output_p = Path(args.output_path)
            output_p.parent.mkdir(parents=True, exist_ok=True)
            STORE.dump(str(output_p))
        else:
            print("Error: 'dump' command is only available for FAISS store. Check your conf/base.yaml (store: faiss)")
    elif args.cmd == "load":
        if CFG and CFG.store_name == 'faiss' and STORE and isinstance(STORE, FaissStore):
            print(f"Loading FAISS index from: {args.input_path} (and .map)")
            if STORE.load(args.input_path):
                print("FAISS index loaded successfully.")
            else:
                print("Failed to load FAISS index.")
        else:
            print("Error: 'load' command is only available for FAISS store. Check your conf/base.yaml (store: faiss)")

    elif args.cmd == "import-text":
        from gaokao_rag.text_only import build_text_index
        build_text_index()
    elif args.cmd == "query-text":
        from gaokao_rag.text_only import query_text_only
        print(json.dumps(query_text_only(args.stem, args.k), ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main() 