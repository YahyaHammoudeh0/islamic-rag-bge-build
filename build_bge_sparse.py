#!/usr/bin/env python3
"""
Islamic RAG — BGE-M3 Sparse Index Builder
==========================================
Reads all JSONL data files, encodes each document using BGE-M3 sparse weights,
and saves the sparse index + doc store for use in the RAG pipeline.

Requirements:
    pip install FlagEmbedding scipy numpy tqdm

Output files (drop these into your db/ directory):
    bge_sparse.npz          — scipy sparse matrix (n_docs × vocab_size)
    bge_sparse_meta.pkl     — doc ID mapping and collection info
    doc_store.pkl           — all texts + metadata (used at query time)

Usage:
    python build_bge_sparse.py
    python build_bge_sparse.py --data-dir ./data --output-dir ./output
    python build_bge_sparse.py --batch-size 64  # larger batches for fast GPUs
"""

import os
import sys
import json
import pickle
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR   = Path(__file__).parent / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
BGE_MODEL_NAME     = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 32   # reduce to 8-16 if OOM on CPU


# ── Load BGE-M3 ───────────────────────────────────────────────────────────────

def load_model(device: str = "auto", model_dir: str = None):
    """Load BGE-M3 model. Auto-detects GPU if available.
    
    Args:
        device:    'auto', 'cuda', or 'cpu'
        model_dir: path to local model dir (skip HuggingFace download)
    """
    from FlagEmbedding import BGEM3FlagModel

    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"  [BGE-M3] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("  [BGE-M3] No CUDA GPU found, using CPU")
        except ImportError:
            device = "cpu"
            print("  [BGE-M3] PyTorch not found, using CPU")

    model_path = model_dir if model_dir else BGE_MODEL_NAME
    use_fp16 = (device != "cpu")
    print(f"  [BGE-M3] Loading model from: {model_path}  (fp16={use_fp16})...")
    model = BGEM3FlagModel(
        model_path,
        use_fp16=use_fp16,
        device=device,
    )
    print("  [BGE-M3] Model loaded ✓")
    return model


# ── Load data ─────────────────────────────────────────────────────────────────

def load_all_docs(data_dir: Path) -> list[dict]:
    """Load all JSONL files from data directory."""
    docs = []
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"ERROR: No .jsonl files found in {data_dir}")
        sys.exit(1)

    print(f"\nLoading {len(jsonl_files)} collections from {data_dir}")
    for path in jsonl_files:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
                    count += 1
        print(f"  {path.stem}: {count:,} docs")

    print(f"\nTotal: {len(docs):,} documents loaded")
    return docs


# ── Build sparse index ────────────────────────────────────────────────────────

def build_sparse_index(model, docs: list[dict], batch_size: int):
    """
    Encode all docs with BGE-M3 sparse weights.
    Returns (sparse_matrix, vocab_size, collection_map)
    """
    texts = [d["text"] for d in docs]
    n_docs = len(texts)

    print(f"\nEncoding {n_docs:,} documents (batch_size={batch_size})...")
    print("This will take a while. Progress bar below:\n")

    all_sparse = []   # list of {token_id: weight} dicts
    t0 = time.time()

    current_batch = batch_size
    i = 0
    pbar = tqdm(total=n_docs, desc="BGE-M3 sparse encode")

    while i < n_docs:
        batch = texts[i : i + current_batch]
        try:
            output = model.encode(
                batch,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
                batch_size=len(batch),
            )
            all_sparse.extend(output["lexical_weights"])
            pbar.update(len(batch))
            i += current_batch
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch > 1:
                current_batch = max(1, current_batch // 2)
                print(f"\n  OOM — reducing batch size to {current_batch} and retrying...", flush=True)
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                print(f"\n  ERROR at batch {i}: {e}")
                raise

    pbar.close()

    elapsed = time.time() - t0
    docs_per_sec = n_docs / elapsed
    print(f"\nEncoding complete: {elapsed/60:.1f} min ({docs_per_sec:.1f} docs/sec)")

    # ── Build sparse matrix ───────────────────────────────────────────────────
    print("\nBuilding sparse matrix...")

    # Find vocab size (max token id + 1)
    vocab_size = 0
    for sparse_vec in all_sparse:
        if sparse_vec:
            max_id = max(int(k) for k in sparse_vec.keys())
            vocab_size = max(vocab_size, max_id + 1)
    vocab_size = max(vocab_size, 250002)  # BGE-M3 uses XLM-R tokenizer (~250K vocab)
    print(f"  Vocab size: {vocab_size:,}")

    # Build COO data for sparse matrix
    rows, cols, data = [], [], []
    for row_idx, sparse_vec in enumerate(all_sparse):
        for token_id, weight in sparse_vec.items():
            rows.append(row_idx)
            cols.append(int(token_id))
            data.append(float(weight))

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(n_docs, vocab_size),
        dtype=np.float32
    )
    print(f"  Matrix shape: {matrix.shape}, non-zeros: {matrix.nnz:,}")

    # ── Collection map ────────────────────────────────────────────────────────
    collection_map = [
        (d.get("collection", "unknown"), d.get("metadata", {}).get("id", str(i)))
        for i, d in enumerate(docs)
    ]

    return matrix, vocab_size, collection_map


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(matrix, vocab_size, collection_map, docs, output_dir: Path):
    """Save all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Sparse matrix
    sparse_path = output_dir / "bge_sparse.npz"
    print(f"\nSaving sparse matrix → {sparse_path}")
    save_npz(str(sparse_path), matrix)
    print(f"  Size: {sparse_path.stat().st_size / 1e6:.1f} MB")

    # 2. Sparse metadata
    meta_path = output_dir / "bge_sparse_meta.pkl"
    print(f"Saving sparse metadata → {meta_path}")
    meta = {
        "vocab_size":      vocab_size,
        "doc_ids":         [f"{d.get('collection','?')}::{i}" for i, d in enumerate(docs)],
        "collection_map":  collection_map,
        "n_docs":          len(docs),
        "model":           BGE_MODEL_NAME,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=4)
    print(f"  Size: {meta_path.stat().st_size / 1e6:.1f} MB")

    # 3. Doc store (text + metadata for lookup at query time)
    doc_store_path = output_dir / "doc_store.pkl"
    print(f"Saving doc store → {doc_store_path}")
    store = {
        "doc_ids":    [f"{d.get('collection','?')}::{i}" for i, d in enumerate(docs)],
        "docs":       [d["text"] for d in docs],
        "meta":       [{"metadata": d.get("metadata", {})} for d in docs],
        "ordered_ids": [f"{d.get('collection','?')}::{i}" for i, d in enumerate(docs)],
        "m_pq":       "N/A",
    }
    with open(doc_store_path, "wb") as f:
        pickle.dump(store, f, protocol=4)
    print(f"  Size: {doc_store_path.stat().st_size / 1e6:.1f} MB")

    print(f"\n✅ All output files saved to: {output_dir.resolve()}")
    print("\nFiles to copy back to db/:")
    for p in [sparse_path, meta_path, doc_store_path]:
        print(f"  {p.name} ({p.stat().st_size/1e6:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build BGE-M3 sparse index for Islamic RAG")
    parser.add_argument("--data-dir",    type=Path, default=DEFAULT_DATA_DIR,   help="Directory with .jsonl files")
    parser.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for index files")
    parser.add_argument("--batch-size",  type=int,  default=DEFAULT_BATCH_SIZE, help="Encoding batch size (lower if OOM)")
    parser.add_argument("--device",      type=str,  default="auto",             help="Device: auto, cuda, cpu")
    parser.add_argument("--model-dir",   type=str,  default=None,               help="Path to local BGE-M3 model dir (skips HuggingFace download)")
    args = parser.parse_args()

    print("=" * 60)
    print("Islamic RAG — BGE-M3 Sparse Index Builder")
    print("=" * 60)
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device:     {args.device}")
    print(f"Model:      {args.model_dir or 'BAAI/bge-m3 (HuggingFace)'}")
    print()

    # Load model
    model = load_model(device=args.device, model_dir=args.model_dir)

    # Load docs
    docs = load_all_docs(args.data_dir)

    # Build index
    matrix, vocab_size, collection_map = build_sparse_index(model, docs, args.batch_size)

    # Save
    save_outputs(matrix, vocab_size, collection_map, docs, args.output_dir)

    print("\nDone! Copy the 3 files from output/ into your db/ directory.")
    print("The query.py will auto-detect and use BGE-M3 sparse search.")


if __name__ == "__main__":
    main()
