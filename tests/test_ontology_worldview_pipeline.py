from src.core.ontology.tagger import tag_document
from src.core.ontology.taxonomy import OntologyTaxonomy
from src.core.ontology.worldview_graph import TaggedSource, build_worldview_graph


def _build_taxonomy() -> OntologyTaxonomy:
    return OntologyTaxonomy(
        concepts={"материализм": ["материя первична"], "идеализм": ["идея первична"]},
        entities=["Ленин", "Маркс"],
        contradiction_pairs=[("материализм", "идеализм")],
        argument_markers={"inference": ["следовательно"], "contrast": ["однако"]},
        zero_shot_labels={
            "class_analysis": "класс пролетариат буржуазия",
            "ideology_philosophy": "материализм идеализм философия",
        },
    )


def test_tag_document_extracts_expected_fields():
    taxonomy = _build_taxonomy()
    text = (
        "Ленин утверждал, что материя первична, однако идея первична отвергается, "
        "следовательно материализм и философия являются основанием анализа."
    )
    tags = tag_document(text=text, taxonomy=taxonomy)

    assert "материализм" in tags.concepts
    assert "идеализм" in tags.concepts
    assert "Ленин" in tags.entities
    assert "материализм<->идеализм" in tags.contradiction_hits
    assert tags.argument_pattern == "inference"
    assert tags.zero_shot_label == "ideology_philosophy"


def test_build_worldview_graph_produces_nodes_and_edges():
    tagged_sources = [
        TaggedSource(
            source_id="s1",
            source_path="Ленин/doc1.txt",
            stance_type="core_self",
            concepts=["материализм", "идеализм"],
            entities=["Ленин"],
            contradiction_hits=["материализм<->идеализм"],
            argument_pattern="inference",
            zero_shot_label="ideology_philosophy",
        ),
        TaggedSource(
            source_id="s2",
            source_path="Маркс/doc2.txt",
            stance_type="influence_agree",
            concepts=["материализм"],
            entities=["Маркс"],
            contradiction_hits=[],
            argument_pattern="unknown",
            zero_shot_label="class_analysis",
        ),
    ]
    graph_payload = build_worldview_graph(tagged_sources=tagged_sources)
    node_ids = {node["id"] for node in graph_payload["nodes"]}

    assert "doc:s1" in node_ids
    assert "concept:материализм" in node_ids
    assert any(
        edge["source"] == "doc:s1" and edge["target"] == "concept:материализм"
        for edge in graph_payload["edges"]
    )
