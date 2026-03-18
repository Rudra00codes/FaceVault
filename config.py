"""
config.py — Centralised configuration for FaceVault.

Edit these constants to tune the application's behaviour
without touching the main application code.
"""

# ── Matching ──────────────────────────────────────────────────────────────────
# Cosine-similarity threshold (L2-normalised vectors → L2 distance ≈ cosine distance).
# A query whose nearest-neighbour distance is *above* this value is considered
# a confident match; below it the match is flagged as low-confidence.
#
# Acceptable range: 0.0 – 2.0  (typical safe values: 0.40 – 0.65)
MATCH_THRESHOLD: float = 0.50

# ── FAISS HNSW index parameters ───────────────────────────────────────────────
# M      : number of bi-directional links per node (higher → better recall, more RAM)
# efSearch: beam width at query time (higher → better recall, slower)
HNSW_M: int = 32
HNSW_EF_SEARCH: int = 128

# ArcFace embedding dimension (fixed by the model — do not change)
EMBEDDING_DIM: int = 512

# ── File paths ────────────────────────────────────────────────────────────────
DATA_FILE: str = "app_data.pkl"
INDEX_FILE: str = "vector_index.bin"
IMAGES_DIR: str = "images"

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE: str = "FaceVault"
PAGE_ICON: str = "🔐"
THUMBNAIL_COLUMNS: int = 5          # number of image columns in the gallery view
