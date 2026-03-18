# FaceVault 🔐

> **Dynamic, unsupervised face identity management** — no model retraining required.  
> Powered by **ArcFace** embeddings + **FAISS-HNSW** approximate nearest-neighbour search.

---

## Overview

FaceVault is a Streamlit web application that lets you build and search a face identity database **on the fly**. It uses state-of-the-art metric learning instead of traditional softmax classification, which means:

- **New identities can be registered instantly** — just upload a photo and give them a name.
- **No GPU or retraining needed** — the deep features are extracted once per image; all similarity logic runs on CPU in milliseconds.
- **Portable** — the entire database fits in two files (`app_data.pkl` + `vector_index.bin`).

---

## Architecture

```
Upload Image
     │
     ▼
┌────────────────────┐
│  DeepFace / ArcFace │  ← 512-dim face embedding (L2-normalised)
└────────────┬───────┘
             │ query vector
             ▼
┌────────────────────────────────────────┐
│  FAISS HNSW Index  (O(log n) search)   │
│  • Multi-layer navigable small world   │
│  • faiss.normalize_L2 before insert    │
└────────────┬───────────────────────────┘
             │ nearest neighbour + distance
             ▼
┌──────────────────────────────────────┐
│  Cluster Dictionary (in-memory + pkl)│
│  cluster_id → [image_paths …]        │
└──────────────────────────────────────┘
```

| Module | Technology | Purpose |
|--------|-----------|---------|
| Face embedding | ArcFace (via `deepface`) | 512-dim angular-margin features |
| Similarity search | FAISS HNSW (`faiss-cpu`) | Sub-linear nearest-neighbour lookup |
| State persistence | `pickle` + `faiss.write_index` | Survive Streamlit re-runs |
| UI | Streamlit | Browser-based interface |

---

## Features

| Tab | Description |
|-----|-------------|
| 🔍 **Database Explorer** | Browse all registered identities; filter by name or cluster ID. |
| ➕ **Add / Search Face** | Upload a photo → ArcFace extract → HNSW search → add to existing person **or** register as new. |

`app_nonHNSW.py` provides a lightweight read-only explorer using a flat (brute-force) FAISS index — useful for debugging or comparing search quality.

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/FaceVault.git
cd FaceVault
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the image dataset

FaceVault expects face images organised as:

```
images/
├── Person_Name/
│   ├── Person_Name_0001.jpg
│   └── …
└── Another_Person/
    └── …
```

> **Recommended dataset:** [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)  
> Extract it into the `images/` directory — the path-fixer in `load_data()` normalises paths automatically across machines.

### 3. Build the initial FAISS index

If you already have `app_data.pkl` and `vector_index.bin` (e.g. shared by a collaborator), place them in the project root and skip this step.

Otherwise, run the provided notebook (or your own indexing script) to populate the files from scratch.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Configuration

All tuneable parameters live in **`config.py`**:

| Constant | Default | Description |
|----------|---------|-------------|
| `MATCH_THRESHOLD` | `0.50` | L2-distance threshold for confident vs. low-confidence match |
| `HNSW_M` | `32` | HNSW graph connectivity (higher → better recall, more RAM) |
| `HNSW_EF_SEARCH` | `128` | Query-time beam width (higher → better recall, slower) |
| `THUMBNAIL_COLUMNS` | `5` | Gallery columns in the explorer view |

---

## Project Structure

```
FaceVault/
├── app.py              # Main Streamlit app (HNSW index, read + write)
├── app_nonHNSW.py      # Read-only explorer (flat FAISS index)
├── config.py           # Centralised tunable constants
├── requirements.txt    # Python dependencies
├── .gitignore          # Excludes large binaries and dataset
└── README.md
```

---

## Notes & Limitations

- `vector_index.bin` and `app_data.pkl` are excluded from version control (see `.gitignore`) because they can be several hundred MB.  
  Share them separately (e.g. Google Drive, Git LFS, DVC) or regenerate from the dataset.
- The LFW image dataset is similarly excluded. Download it directly from the [official source](http://vis-www.cs.umass.edu/lfw/).
- ArcFace embeddings are extracted with CPU inference; expect ~1–2 s per image on a modern laptop.

---

## License

MIT — see `LICENSE` for details.
