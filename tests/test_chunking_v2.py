from src.core.preprocessing.chunker_v2 import chunk_document
from src.core.preprocessing.chunking_config import ChunkingConfig
from src.core.preprocessing.chunking_quality import (
    bad_boundary_ratio,
    token_window_compliance_ratio,
)


def _config() -> ChunkingConfig:
    return ChunkingConfig(
        min_tokens=8,
        max_tokens=16,
        overlap_ratio=0.1,
        thesis_markers=["следовательно", "однако"],
        chapter_markers=[r"^ГЛАВА\s+\d+"],
        section_markers=[r"^§\s*\d+"],
        min_chunk_chars=30,
        max_bad_boundary_ratio=0.2,
    )


def test_chunk_document_produces_metadata_and_ids():
    text = (
        "ГЛАВА 1\n\n"
        "Материализм рассматривает объективную реальность как первичную основу бытия. "
        "Следовательно, общественные отношения должны анализироваться через материальные условия.\n\n"
        "§ 2\n\n"
        "Однако идеализм утверждает первичность идеи, что порождает спор в философии. "
        "Поэтому требуется диалектический метод для разрешения противоречия."
    )
    chunks = chunk_document(
        source_id="src_demo",
        source_path="Ленин/текст.txt",
        author="Ленин",
        work="текст",
        stance_type="core_self",
        text=text,
        config=_config(),
    )
    assert chunks
    assert all(chunk.chunk_id.startswith("chunk_") for chunk in chunks)
    assert all(chunk.source_id == "src_demo" for chunk in chunks)
    assert all(chunk.chapter in {"ГЛАВА 1", "unknown"} for chunk in chunks)
    assert all(chunk.token_count >= 1 for chunk in chunks)


def test_chunk_quality_metrics():
    text = (
        "ГЛАВА 1\n\n"
        "Диалектика объясняет движение через противоречия и их снятие. "
        "Следовательно требуется анализ конкретной исторической ситуации.\n\n"
        "Это второй абзац, который продолжает аргументацию и завершает тезис."
    )
    chunks = chunk_document(
        source_id="src_metrics",
        source_path="Автор/файл.txt",
        author="Автор",
        work="файл",
        stance_type="contextual",
        text=text,
        config=_config(),
    )
    ratio = bad_boundary_ratio(chunks=chunks)
    compliance = token_window_compliance_ratio(
        chunks=chunks, min_tokens=8, max_tokens=16
    )
    assert 0.0 <= ratio <= 1.0
    assert 0.0 <= compliance <= 1.0
