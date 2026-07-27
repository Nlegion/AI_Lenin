"""Ensure payload index on stance_type for dialectical filtered retrieve.

One-shot ops script — not part of hot-path boot. Requires qdrant-client>=1.7.0.

Note: Local (embedded) Qdrant ignores payload indexes; filters still work via scan.
Prefer server Qdrant for index performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.settings.dialectical_constants import MIN_QDRANT_CLIENT_VERSION  # noqa: E402


def _client_version() -> str:
    import importlib.metadata as metadata

    return metadata.version("qdrant-client")


def ensure_stance_index(
    *,
    qdrant_path: Path | None = None,
    server_url: str | None = None,
    collection_name: str,
    wait_timeout_sec: float = 600.0,
    poll_interval_sec: float = 2.0,
) -> int:
    version = _client_version()
    if Version(version) < Version(MIN_QDRANT_CLIENT_VERSION):
        print(
            f"ERROR: qdrant-client {version} < {MIN_QDRANT_CLIENT_VERSION}. "
            f"Upgrade client or use an alternate payload-index API.",
            file=sys.stderr,
        )
        return 2

    # Explicit exception types used elsewhere; import verifies package layout.
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.exceptions import (  # noqa: F401
        ResponseHandlingException,
        UnexpectedResponse,
    )

    is_local = server_url is None
    if is_local:
        if qdrant_path is None:
            print("ERROR: qdrant_path required for local mode", file=sys.stderr)
            return 1
        client = QdrantClient(path=str(qdrant_path))
    else:
        client = QdrantClient(url=server_url)

    existing = {item.name for item in client.get_collections().collections}
    if collection_name not in existing:
        print(f"ERROR: collection not found: {collection_name}", file=sys.stderr)
        return 1

    info = client.get_collection(collection_name=collection_name)
    schema = getattr(info, "payload_schema", None) or {}
    if isinstance(schema, dict) and "stance_type" in schema:
        print(f"OK: stance_type index already present on {collection_name}")
        return 0

    print(f"Creating keyword payload index on stance_type for {collection_name} ...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="stance_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=not is_local,
        )
        local_noop = any("no effect in the local Qdrant" in str(item.message) for item in caught)

    if is_local or local_noop:
        print(
            "OK: create_payload_index invoked. "
            "Local Qdrant ignores payload indexes (filters still scan); "
            "use server Qdrant for indexed filter performance."
        )
        return 0

    deadline = time.perf_counter() + wait_timeout_sec
    while time.perf_counter() < deadline:
        info = client.get_collection(collection_name=collection_name)
        schema = getattr(info, "payload_schema", None) or {}
        if isinstance(schema, dict) and "stance_type" in schema:
            print("OK: stance_type index ready")
            return 0
        time.sleep(poll_interval_sec)
    print("ERROR: timed out waiting for stance_type index readiness", file=sys.stderr)
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure Qdrant stance_type payload index.")
    parser.add_argument("--qdrant-path", default="database/qdrant_local")
    parser.add_argument("--server-url", default="")
    parser.add_argument("--collection", default="philosophy_ontology_giga_v1")
    parser.add_argument("--wait-timeout-sec", type=float, default=600.0)
    args = parser.parse_args()
    if args.server_url:
        return ensure_stance_index(
            server_url=args.server_url,
            collection_name=args.collection,
            wait_timeout_sec=args.wait_timeout_sec,
        )
    return ensure_stance_index(
        qdrant_path=(REPO_ROOT / args.qdrant_path).resolve(),
        collection_name=args.collection,
        wait_timeout_sec=args.wait_timeout_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
