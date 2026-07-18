from pathlib import Path

from src.core.vector.bm25_sparse import Bm25SparseEncoder
from src.core.vector.qdrant_ingestion import CheckpointStore


def test_bm25_sparse_encoder_generates_sparse_vectors():
    encoder = Bm25SparseEncoder()
    encoder.fit(
        documents=[
            "материализм и диалектика",
            "диалектика и революция",
            "классовая борьба и революция",
        ]
    )
    vector = encoder.encode_document("диалектика и революция")
    assert vector.indices
    assert vector.values
    assert len(vector.indices) == len(vector.values)


def test_bm25_sparse_encoder_state_roundtrip(tmp_path: Path):
    encoder = Bm25SparseEncoder()
    encoder.fit(documents=["мир труд май", "труд и капитал"])
    state_path = tmp_path / "bm25_state.json"
    encoder.save(path=state_path)
    loaded = Bm25SparseEncoder.load(path=state_path)
    vector = loaded.encode_query("труд капитал")
    assert vector.indices


def test_checkpoint_store_roundtrip(tmp_path: Path):
    checkpoint = CheckpointStore(path=tmp_path / "checkpoint.offset")
    assert checkpoint.load() == 0
    checkpoint.save(128)
    assert checkpoint.load() == 128
