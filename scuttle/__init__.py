"""Scuttle: Frontier-calibrated single-pass decision model for parallel swarm graph search."""
from scuttle.indexer import index_directory
from scuttle.model import ABEL_W, DEFAULT_MODEL_ID, THRESHOLDS, ScuttleWorker
from scuttle.searcher import CorpusBM25, ScuttleSearcher
from scuttle.world import World, load_world

__version__ = "0.1.0"
__all__ = [
    "ABEL_W",
    "CorpusBM25",
    "DEFAULT_MODEL_ID",
    "ScuttleSearcher",
    "ScuttleWorker",
    "THRESHOLDS",
    "World",
    "index_directory",
    "load_world",
]
