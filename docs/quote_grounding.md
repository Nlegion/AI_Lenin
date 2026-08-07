# Quote grounding and citation metadata

## Chunk metadata fields used for citations

When rendering or validating attributions, only these payload/meta keys are trusted
(from RAG / Qdrant retrieval chunks — never from the news body):

| Field | Meaning |
|-------|---------|
| `author` | Author if present |
| `work` / `title` | Work title if present |
| `volume` / `том` | PSS volume number |
| `page` / `стр` / `page_start` | Page if present |
| `source_id` | Stable source document id |
| `chunk_id` | Chunk id (required for allowlist binding) |
| `quote` / `thesis` | Optional pre-annotated cite span (explicit cite-marked) |

If a field is missing, the citation renderer must omit it — never invent `том`/`стр`/work.

## Quote candidates

1. Regex spans in «…», `"…"`, „…“, “…” inside a single chunk.
2. Explicit meta `quote` / `thesis` if present on the chunk.
3. Each candidate is bound to exactly one `chunk_id`.
4. Trivial lead-ins are rejected via `quality_postcheck.trivial_quote_stoplist`.
5. Min length: `min_quote_chars` / `min_quote_content_tokens`.

## Grounding

Normalize (NFKC, casefold, ё→е, unify quotes/dashes/whitespace). An answer quote is
grounded iff the normalized span is a substring of its candidate’s chunk text.
No paraphrase fuzzy match. No cross-chunk splicing.

## Feature flags

See `config/quality_postcheck.yaml`. Update marker lists via code review
(`docs/news_guard_patterns.md`).
