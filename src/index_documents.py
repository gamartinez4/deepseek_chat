"""CLI utility to index product documents."""
from __future__ import annotations

import argparse
from pathlib import Path

from .retrieval.vector_store import build_index


def _gather_texts(folder: Path) -> list[str]:
    texts: list[str] = []
    for file in folder.rglob("*.txt"):
        texts.append(file.read_text(encoding="utf-8"))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Index product description documents.")
    parser.add_argument("--docs-path", type=str, default="data/", help="Folder with .txt files.")
    parser.add_argument("--index-path", type=str, default=None, help="Output index directory.")
    args = parser.parse_args()

    docs = _gather_texts(Path(args.docs_path))
    if not docs:
        print(f"No .txt docs found in {args.docs_path}")
        return

    build_index(docs, args.index_path)
    print(f"Indexed {len(docs)} documents.")


if __name__ == "__main__":
    main() 