# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Does the sample store obey ``StorageService``?

Only contract assertions live here — nothing about how ``LocalStore`` happens to
lay bytes out on disk (pruned directories, verbatim ``file.json``,
``.cache/staging``). Those are properties of the local adapter, not of the
interface, and a parity suite that asserts them is one that gets skipped.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.errors import InvalidArgument, StorageConfigInvalid
from dgml_core.storage_service import StorageConfig
from dgml_storage_s3 import S3MongoStore

# --------------------------------------------------------------------- config


def test_unknown_option_is_rejected(s3_config: StorageConfig) -> None:
    bad = StorageConfig(
        provider=s3_config.provider,
        root=s3_config.root,
        options={**s3_config.options, "buckett": "typo"},
    )
    with pytest.raises(StorageConfigInvalid, match="unknown fields"):
        S3MongoStore.parse_config(bad)


@pytest.mark.parametrize("missing", ["bucket", "mongo_database"])
def test_required_options(s3_config: StorageConfig, missing: str) -> None:
    options = {k: v for k, v in s3_config.options.items() if k != missing}
    with pytest.raises(StorageConfigInvalid, match=missing):
        S3MongoStore.parse_config(
            StorageConfig(provider=s3_config.provider, root=s3_config.root, options=options)
        )


def test_no_config_field_would_carry_a_credential() -> None:
    """Credentials must come from the environment.

    ``storage_resolve`` decides what is secret by substring match on the option
    *name*; anything unmatched is persisted to the plaintext registry. So an
    option named e.g. ``mongo_uri`` (which would hold a password) must never
    appear here — this pins that intent rather than trusting a code review."""
    hints = ("key", "secret", "token", "password", "credential", "uri", "url", "dsn", "conn")
    offenders = [
        f
        for f in S3MongoStore.config_fields
        if any(h in f.lower() for h in hints) and f != "endpoint_url"
    ]
    assert offenders == [], f"credential-shaped config fields: {offenders}"


def test_prefix_option_isolates_workspaces_sharing_a_bucket(s3_config: StorageConfig) -> None:
    """``prefix`` is what lets several workspaces share one bucket.

    Without it a second workspace inherits the first's blobs while getting a
    fresh document store, and ``dgml check`` reports them as ``missing_metadata``
    orphans. The keys must carry the prefix on the wire and have it stripped on
    the way back, so callers never see it."""
    import boto3

    opts = dict(s3_config.options)
    client_kwargs: dict[str, object] = {"region_name": "us-east-1"}
    if opts.get("endpoint_url"):
        client_kwargs["endpoint_url"] = opts["endpoint_url"]

    stores = {}
    for tenant in ("alpha", "beta"):
        cfg = StorageConfig(s3_config.provider, s3_config.root, {**opts, "prefix": tenant})
        stores[tenant] = S3MongoStore(S3MongoStore.parse_config(cfg))
        stores[tenant].put_blob("files/f1/report.pdf", tenant.encode())

    # Same key, different tenants, no collision.
    for tenant, store_ in stores.items():
        assert store_.get_blob("files/f1/report.pdf") == tenant.encode()
        assert store_.list_blobs("files/") == ["files/f1/report.pdf"]  # prefix stripped

    # …and the prefix really is on the wire.
    raw = boto3.client("s3", **client_kwargs).list_objects_v2(Bucket=opts["bucket"])
    assert sorted(o["Key"] for o in raw["Contents"]) == [
        "alpha/files/f1/report.pdf",
        "beta/files/f1/report.pdf",
    ]

    # Deleting one tenant's prefix leaves the other intact.
    stores["alpha"].delete_blobs("files/")
    assert stores["alpha"].list_blobs("") == []
    assert stores["beta"].list_blobs("") == ["files/f1/report.pdf"]


# ---------------------------------------------------------------------- blobs


def test_blob_round_trip(store: S3MongoStore) -> None:
    store.put_blob("files/f1/report.pdf", b"%PDF-1.4\n")
    assert store.get_blob("files/f1/report.pdf") == b"%PDF-1.4\n"
    assert store.blob_exists("files/f1/report.pdf") is True


def test_missing_blob_raises_file_not_found(store: S3MongoStore) -> None:
    assert store.blob_exists("files/nope/x.pdf") is False
    with pytest.raises(FileNotFoundError):
        store.get_blob("files/nope/x.pdf")


def test_put_blob_overwrites(store: S3MongoStore) -> None:
    store.put_blob("k/a", b"one")
    store.put_blob("k/a", b"two")
    assert store.get_blob("k/a") == b"two"


def test_delete_blob_is_idempotent(store: S3MongoStore) -> None:
    store.put_blob("k/a", b"x")
    store.delete_blob("k/a")
    store.delete_blob("k/a")  # missing key is a no-op, not an error
    assert store.blob_exists("k/a") is False


def test_upload_and_download_round_trip(store: S3MongoStore, tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    store.upload_blob("files/f1/page_images/page_1.png", src)

    dest = tmp_path / "out" / "page_1.png"
    store.download_blob("files/f1/page_images/page_1.png", dest)
    assert dest.read_bytes() == b"payload"

    with pytest.raises(FileNotFoundError):
        store.download_blob("files/f1/page_images/page_9.png", tmp_path / "missing.png")


def test_list_blobs_is_sorted_and_prefix_scoped(store: S3MongoStore) -> None:
    for key in ("files/f1/b.pdf", "files/f1/a.pdf", "files/f2/c.pdf"):
        store.put_blob(key, b"x")
    assert store.list_blobs("files/f1/") == ["files/f1/a.pdf", "files/f1/b.pdf"]
    assert len(store.list_blobs("files/")) == 3
    assert store.list_blobs("") == ["files/f1/a.pdf", "files/f1/b.pdf", "files/f2/c.pdf"]
    assert store.list_blobs("files/nope/") == []


def test_list_blobs_matches_by_raw_string_prefix(store: S3MongoStore) -> None:
    """Mid-segment prefixes select siblings — the property `LocalStore` had to
    be fixed for (W6.1), and the one S3 gives natively."""
    store.put_blob("files/f1/a.pdf", b"a")
    store.put_blob("files/f1x/b.pdf", b"b")
    assert store.list_blobs("files/f1") == ["files/f1/a.pdf", "files/f1x/b.pdf"]
    assert store.list_blobs("files/f1/") == ["files/f1/a.pdf"]


def test_list_blobs_pages_past_the_thousand_key_limit(store: S3MongoStore) -> None:
    """The guaranteed bug if the paginator is skipped.

    ``list_objects_v2`` returns at most 1000 keys per response, while the
    contract is *every* key under the prefix. ``_entity_ids`` lists a whole
    workspace, which passes 1000 blobs at roughly 60 files — so a store that
    ignores ``NextContinuationToken`` silently returns a short list."""
    total = 1050
    for n in range(total):
        store.put_blob(f"files/f1/page_images/page_{n:05d}.png", b"x")
    keys = store.list_blobs("files/f1/page_images/")
    assert len(keys) == total
    assert keys == sorted(keys)


def test_delete_blobs_removes_a_whole_prefix(store: S3MongoStore) -> None:
    store.put_blob("files/f1/a.pdf", b"a")
    store.put_blob("files/f1/page_images/page_1.png", b"b")
    store.put_blob("files/f2/keep.pdf", b"c")

    store.delete_blobs("files/f1/")

    assert store.list_blobs("files/f1/") == []
    assert store.list_blobs("files/f2/") == ["files/f2/keep.pdf"]
    store.delete_blobs("files/nothing/")  # no-op, not an error


def test_delete_blobs_batches_past_the_thousand_key_limit(store: S3MongoStore) -> None:
    """``delete_objects`` also caps at 1000 keys per call."""
    for n in range(1200):
        store.put_blob(f"files/f1/page_images/page_{n:05d}.png", b"x")
    store.delete_blobs("files/f1/")
    assert store.list_blobs("files/f1/") == []


def test_sha256_blob_is_the_plain_digest_of_the_bytes(store: S3MongoStore) -> None:
    """Not the ETag. S3's ETag is a checksum-of-checksums for multipart uploads;
    this value becomes an attestation leaf, so it is part of the on-chain
    contract and must be the plain SHA-256 of the full byte sequence."""
    payload = b"proof-of-origin" * 5000
    store.put_blob("files/f1/report.pdf", payload)
    assert store.sha256_blob("files/f1/report.pdf") == hashlib.sha256(payload).hexdigest()


# ------------------------------------------------------------------ documents


def test_document_round_trip_without_id_leakage(store: S3MongoStore) -> None:
    """Mongo needs an ``_id``; the DGML body must not gain one. ``FileRecord``
    and friends parse these dicts and do not expect an extra field."""
    body = {"id": "f1", "original_filename": "a.pdf", "sha256": "ab" * 32}
    store.put_doc(layout.Collection.FILES, "f1", body)
    assert store.get_doc(layout.Collection.FILES, "f1") == body
    assert store.find_docs(layout.Collection.FILES, {}) == [body]


def test_put_doc_replaces_rather_than_merges(store: S3MongoStore) -> None:
    store.put_doc(layout.Collection.FILES, "f1", {"id": "f1", "page_count": 3})
    store.put_doc(layout.Collection.FILES, "f1", {"id": "f1"})
    assert store.get_doc(layout.Collection.FILES, "f1") == {"id": "f1"}


def test_missing_document_is_none(store: S3MongoStore) -> None:
    assert store.get_doc(layout.Collection.FILES, "nope") is None


def test_find_docs_queries_and_empty_query_returns_all(store: S3MongoStore) -> None:
    for did, fid in (("d1", "f1"), ("d1", "f2"), ("d2", "f1")):
        store.put_doc(
            layout.Collection.ASSIGNMENTS,
            layout.pair_id(did, fid),
            {"docset_id": did, "file_id": fid},
        )
    assert len(store.find_docs(layout.Collection.ASSIGNMENTS, {})) == 3
    assert sorted(
        d["file_id"] for d in store.find_docs(layout.Collection.ASSIGNMENTS, {"docset_id": "d1"})
    ) == ["f1", "f2"]
    assert sorted(
        d["docset_id"] for d in store.find_docs(layout.Collection.ASSIGNMENTS, {"file_id": "f1"})
    ) == ["d1", "d2"]


def test_composite_ids_round_trip(store: S3MongoStore) -> None:
    """Assignments and extraction stats are keyed by ``"<docset>/<file>"``."""
    pair = layout.pair_id("d1", "f1")
    store.put_doc(layout.Collection.EXTRACTION_STATS, pair, {"matched": 3})
    assert store.get_doc(layout.Collection.EXTRACTION_STATS, pair) == {"matched": 3}
    store.delete_doc(layout.Collection.EXTRACTION_STATS, pair)
    assert store.get_doc(layout.Collection.EXTRACTION_STATS, pair) is None


def test_delete_doc_is_idempotent_and_delete_docs_counts(store: S3MongoStore) -> None:
    store.delete_doc(layout.Collection.FILES, "never-existed")  # no error
    for did, fid in (("d1", "f1"), ("d1", "f2"), ("d2", "f1")):
        store.put_doc(
            layout.Collection.ASSIGNMENTS,
            layout.pair_id(did, fid),
            {"docset_id": did, "file_id": fid},
        )
    assert store.delete_docs(layout.Collection.ASSIGNMENTS, {"docset_id": "d1"}) == 2
    assert len(store.find_docs(layout.Collection.ASSIGNMENTS, {})) == 1


def test_append_doc_rejects_an_addressed_collection(store: S3MongoStore) -> None:
    """Must match ``LocalStore``. Mongo would happily insert an id-less document
    into any collection, but appending to an addressed one is a caller bug — and
    a caller bug that raises on one backend and passes on another is precisely
    the divergence this package exists to catch."""
    with pytest.raises(InvalidArgument, match="append-only"):
        store.append_doc(layout.Collection.FILES, {"id": "f1"})


def test_mongo_username_is_not_a_config_field(s3_config: StorageConfig) -> None:
    """There is no half-credential in config.

    A username with no way to supply a password builds a URI pymongo rejects
    (``ConfigurationError: A password is required``), and adding the password
    key would put it in the plaintext registry. Authentication is all-or-nothing
    via ``DGML_MONGO_URI``."""
    assert "mongo_username" not in S3MongoStore.config_fields
    assert not any("password" in f or "secret" in f for f in S3MongoStore.config_fields)
    with pytest.raises(StorageConfigInvalid, match="unknown fields"):
        S3MongoStore.parse_config(
            StorageConfig(
                provider=s3_config.provider,
                root=s3_config.root,
                options={**s3_config.options, "mongo_username": "admin"},
            )
        )


def test_append_doc_is_append_only_and_ordered(store: S3MongoStore) -> None:
    store.append_doc(layout.Collection.USAGE, {"op": "transcribe", "cost_usd": 0.01})
    store.append_doc(layout.Collection.USAGE, {"op": "label", "cost_usd": 0.02})
    events = store.find_docs(layout.Collection.USAGE, {})
    assert [e["op"] for e in events] == ["transcribe", "label"]
    assert all("_id" not in e for e in events)


def test_blobs_and_documents_are_separate_namespaces(store: S3MongoStore) -> None:
    """The property local disk cannot have, and the reason this package exists.

    ``docsets/<id>/schema.json`` was once written with ``put_blob`` and read with
    ``get_doc``; on one filesystem that accidentally worked. Here a document and
    a blob at the same-looking address are simply different objects."""
    store.put_blob("docsets/d1/schema.json", b'{"from": "blob"}')
    store.put_doc(layout.Collection.DOCSETS, "d1", {"from": "document"})

    assert store.get_blob("docsets/d1/schema.json") == b'{"from": "blob"}'
    assert store.get_doc(layout.Collection.DOCSETS, "d1") == {"from": "document"}
    # A document never shows up in the blob namespace.
    assert store.list_blobs("docsets/") == ["docsets/d1/schema.json"]
