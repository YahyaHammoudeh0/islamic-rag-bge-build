# Islamic RAG — BGE-M3 Sparse Index Builder

Builds a BGE-M3 sparse index over Islamic texts (Quran, Hadith, Tafsir, Fiqh) for use in a hybrid RAG pipeline.

## What this does

Encodes 80,235 documents using [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) sparse lexical weights. Output files plug directly into the RAG query engine.

## Setup

```bash
git clone <this-repo>
cd islamic-rag-bge-build
pip install -r requirements.txt
```

## Run

```bash
# Auto-detects GPU if available
python build_bge_sparse.py

# Custom options
python build_bge_sparse.py --batch-size 64 --device cuda --output-dir ./output
```

**Estimated time:**
- GPU (T4/A100): ~5-15 min
- CPU (16-core): ~8-11 hours

## Output

Three files will be created in `output/`:

| File | Description |
|---|---|
| `bge_sparse.npz` | Sparse matrix (n_docs × vocab_size) |
| `bge_sparse_meta.pkl` | Doc ID mapping and collection metadata |
| `doc_store.pkl` | All texts + metadata for query-time lookup |

Copy these to your `db/` directory. The query engine auto-detects and uses them.

## Collections

| Collection | Docs |
|---|---|
| Quran | 6,236 |
| Hadith (Bukhari/Muslim) | 14,939 |
| Sunan (Abu Dawud, Nasa'i, Ibn Majah, Tirmidhi, Malik) | 21,037 |
| Nawawi 40 + Hadith Qudsi | 82 |
| Tafsir (Ibn Kathir, Jalalayn, Ibn Abbas, Maarif, Asbab al-Nuzul) | 27,667 |
| Fiqh (Four Schools, Bidayat al-Mujtahid) | 6,420 |
| Aqeedah (Tahawiyya, Wasitiyya) | 1,204 |
| Usul al-Fiqh | 1,122 |
| Ihya Ulum al-Din | 1,530 |
| **Total** | **80,235** |
