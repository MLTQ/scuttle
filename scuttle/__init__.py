"""Scuttle: Frontier-calibrated single-pass decision model for parallel swarm graph search."""
from scuttle.indexer import index_directory
from scuttle.model import ABEL_W, DEFAULT_MODEL_ID, THRESHOLDS, ScuttleWorker, frontier_value
from scuttle.searcher import CorpusBM25, ScuttleSearcher
from scuttle.sources import index_dataswamp, index_sqlite, index_structured_file, load_or_build_source
from scuttle.world import World, load_world

__version__ = "0.3.0"
__all__ = [
    "ABEL_W",
    "CorpusBM25",
    "DEFAULT_MODEL_ID",
    "ScuttleSearcher",
    "ScuttleWorker",
    "THRESHOLDS",
    "World",
    "frontier_value",
    "index_dataswamp",
    "index_directory",
    "index_sqlite",
    "index_structured_file",
    "load_or_build_source",
    "load_world",
]
