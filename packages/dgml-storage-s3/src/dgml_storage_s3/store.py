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

"""A sample :class:`~dgml_core.storage_service.StorageService`: S3 + MongoDB.

Blobs live in an S3-compatible bucket, documents in MongoDB collections — the
two APIs ``StorageService`` was modelled on, so nearly every method here is a
one-line delegation. That is the point: this is a reference to copy, not a
tuned production store.

**Sample, not supported.** It is resolved by dotted path like any third party's
own store::

    [storage.default]
    provider = "dgml_storage_s3:S3MongoStore"
    bucket = "dgml-dev"
    endpoint_url = "http://localhost:9000"   # MinIO; omit for real AWS S3
    mongo_host = "localhost"
    mongo_database = "dgml_dev"

MinIO is not a separate backend — it speaks the S3 API, so the same class runs
against it locally and against AWS in production by changing ``endpoint_url``.

Credentials
-----------

**Credentials are read from the environment, never from DGML config.** S3 uses
boto3's default chain (``AWS_ACCESS_KEY_ID`` / ``~/.aws/credentials`` / an IAM
role); Mongo reads ``DGML_MONGO_URI`` if set. Config carries *identity only* —
bucket, endpoint, database — which is also exactly what should define the store
fingerprint.

This is not merely stylistic. ``dgml_core.storage_resolve`` splits config into a
non-secret snapshot (persisted to the plaintext registry) and secret options
(kept out of it), and the split is a **substring match on the option name**
against ``("key", "secret", "token", "password", "credential")``. An option
named ``mongo_uri`` holding ``mongodb://admin:hunter2@host`` matches none of
those, so the password would be written to the registry in plaintext. If you
ever add an inline-credential option, its name must contain one of those
substrings.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dgml_core.errors import DgmlError, InvalidArgument, StorageConfigInvalid
from dgml_core.layout import Collection
from dgml_core.storage_service import StorageConfig, StorageService

#: Environment variable holding the full MongoDB connection string, including
#: any credentials. Deliberately not a config key — see the module docstring.
MONGO_URI_ENV = "DGML_MONGO_URI"

#: S3 caps a single ``delete_objects`` call at 1000 keys.
_DELETE_BATCH = 1000


class S3MongoStore(StorageService):
    """Blobs in an S3-compatible bucket, documents in MongoDB."""

    name = "s3-mongo"
    config_fields = frozenset(
        {
            "bucket",
            "region",
            "endpoint_url",
            "prefix",
            "mongo_host",
            "mongo_port",
            "mongo_database",
        }
    )

    # ---- configuration ----

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        bucket = config.options.get("bucket")
        if not isinstance(bucket, str) or not bucket.strip():
            raise StorageConfigInvalid(f"[storage] provider {cls.name!r} requires a 'bucket'")
        database = config.options.get("mongo_database")
        if not isinstance(database, str) or not database.strip():
            raise StorageConfigInvalid(
                f"[storage] provider {cls.name!r} requires a 'mongo_database'"
            )
        port = config.options.get("mongo_port")
        if port is not None and not isinstance(port, int):
            raise StorageConfigInvalid("'mongo_port' must be an integer")
        prefix = config.options.get("prefix")
        if prefix is not None and not isinstance(prefix, str):
            raise StorageConfigInvalid("'prefix' must be a string")
        return config

    def __init__(self, config: StorageConfig) -> None:
        # Lazy SDK import with an actionable message, per the ABC's contract:
        # a workspace that never opens this store must not need boto3/pymongo.
        try:
            import boto3
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise DgmlError(
                "the s3-mongo sample store needs boto3 and pymongo: pip install dgml-storage-s3"
            ) from exc

        opts = config.options
        self._bucket = str(opts["bucket"])
        # An optional key prefix lets several workspaces share one bucket. Kept
        # normalized to "" or "…/" so key joins never double or drop a slash.
        raw_prefix = str(opts.get("prefix") or "").strip("/")
        self._prefix = f"{raw_prefix}/" if raw_prefix else ""

        session_kwargs: dict[str, Any] = {}
        if opts.get("region"):
            session_kwargs["region_name"] = str(opts["region"])
        client_kwargs: dict[str, Any] = dict(session_kwargs)
        if opts.get("endpoint_url"):
            client_kwargs["endpoint_url"] = str(opts["endpoint_url"])
        # Credentials come from boto3's default chain — env, shared config, or
        # an instance role. Never from DGML config.
        # Untyped: boto3 ships no stubs, and adding boto3-stubs for a sample
        # would put a large dev dependency in the tree for little gain.
        self._s3: Any = boto3.client("s3", **client_kwargs)

        # Authentication is all-or-nothing via the environment. There is
        # deliberately no username/password config key: a half-credential in
        # config (a username with no way to supply the password) builds a URI
        # pymongo rejects outright, and adding the password key is exactly the
        # plaintext-registry leak this design avoids.
        uri = os.environ.get(MONGO_URI_ENV)
        if not uri:
            host = str(opts.get("mongo_host") or "localhost")
            port = int(opts.get("mongo_port") or 27017)
            uri = f"mongodb://{host}:{port}"
        self._db: Any = MongoClient(uri)[str(opts["mongo_database"])]

    # ---- key mapping ----

    def _obj(self, key: str) -> str:
        """The S3 object key for a store key (adds the configured prefix)."""
        return f"{self._prefix}{key}"

    def _key(self, obj: str) -> str:
        """The store key for an S3 object key (strips the configured prefix)."""
        return obj[len(self._prefix) :] if self._prefix else obj

    # ---- Blobs (S3) ----

    def put_blob(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=self._obj(key), Body=data)

    def get_blob(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=self._obj(key))
        except ClientError as exc:
            if _is_missing(exc):
                raise FileNotFoundError(f"no blob at key {key!r}") from exc
            raise
        body: bytes = response["Body"].read()
        return body

    def delete_blob(self, key: str) -> None:
        # S3 delete is already idempotent — a missing key is not an error.
        self._s3.delete_object(Bucket=self._bucket, Key=self._obj(key))

    def blob_exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._obj(key))
        except ClientError as exc:
            if _is_missing(exc):
                return False
            raise
        return True

    def list_blobs(self, prefix: str) -> list[str]:
        # MUST paginate. list_objects_v2 returns at most 1000 keys per response,
        # and the contract here is *every* key under the prefix — `_entity_ids`
        # lists a whole workspace, which passes 1000 blobs at ~60 files. Ignoring
        # NextContinuationToken would silently return a short list rather than
        # fail, which is the worst failure mode there is.
        paginator = self._s3.get_paginator("list_objects_v2")
        keys = [
            self._key(obj["Key"])
            for page in paginator.paginate(Bucket=self._bucket, Prefix=self._obj(prefix))
            for obj in page.get("Contents", [])
        ]
        return sorted(keys)

    def upload_blob(self, key: str, src: Path) -> None:
        self._s3.upload_file(str(src), self._bucket, self._obj(key))

    def download_blob(self, key: str, dest: Path) -> None:
        from botocore.exceptions import ClientError

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3.download_file(self._bucket, self._obj(key), str(dest))
        except ClientError as exc:
            if _is_missing(exc):
                raise FileNotFoundError(f"no blob at key {key!r}") from exc
            raise

    def delete_blobs(self, prefix: str) -> None:
        # Cascades call this last (see WorkspaceOps): the authoritative record
        # dies first, so an interrupted cascade leaves recoverable orphaned bytes.
        keys = self.list_blobs(prefix)
        for start in range(0, len(keys), _DELETE_BATCH):
            batch = keys[start : start + _DELETE_BATCH]
            self._s3.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": self._obj(k)} for k in batch], "Quiet": True},
            )

    # ---- Documents (MongoDB) ----
    #
    # Mongo needs an ``_id``; the DGML document body must not carry one. Every
    # write sets it and every read strips it, so ``get_doc`` returns exactly the
    # body ``put_doc`` was given — ``FileRecord.from_json`` and friends do not
    # expect an extra field.

    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        self._db[collection].replace_one({"_id": doc_id}, {**doc, "_id": doc_id}, upsert=True)

    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        found = self._db[collection].find_one({"_id": doc_id})
        return _strip_id(found) if found is not None else None

    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        # An empty query means the whole collection, not "no results".
        return [_strip_id(doc) for doc in self._db[collection].find(dict(query))]

    def delete_doc(self, collection: str, doc_id: str) -> None:
        self._db[collection].delete_one({"_id": doc_id})

    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        return int(self._db[collection].delete_many(dict(query)).deleted_count)

    def append_doc(self, collection: str, doc: dict[str, Any]) -> None:
        # Append-only (the usage log): no id, never fetched or replaced
        # individually. Mongo mints its own ``_id``, which reads strip.
        #
        # Rejected for every other collection, matching LocalStore. Mongo would
        # happily insert an id-less document anywhere, but appending to an
        # addressed collection is a caller bug, and a caller bug that raises on
        # one backend and passes on another is the whole class of defect this
        # package exists to surface.
        if collection != Collection.USAGE:
            raise InvalidArgument(
                f"{collection!r} is not an append-only collection; use put_doc "
                f"(append-only: {Collection.USAGE.value!r})"
            )
        self._db[collection].insert_one(dict(doc))


def _strip_id(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The document body without Mongo's ``_id`` routing key."""
    return {k: v for k, v in doc.items() if k != "_id"}


def _is_missing(exc: Any) -> bool:
    """Whether a botocore ``ClientError`` means "no such key/bucket".

    ``head_object`` reports a missing key as ``404``/``NoSuchKey`` depending on
    the operation and the server, and MinIO and AWS do not agree on every code,
    so match on both.
    """
    error = getattr(exc, "response", {}).get("Error", {})
    return str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}
