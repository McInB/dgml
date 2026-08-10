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

"""Fixtures for the sample store.

Two modes, same tests:

- **Real services** when ``DGML_TEST_S3_ENDPOINT`` / ``DGML_TEST_MONGO_URI`` are
  set (the ``docker compose`` stack, or anything else S3- and Mongo-compatible).
  This is what exercises wire behaviour — genuine pagination, real error codes.
- **In-process fakes** otherwise (``moto`` for S3, ``mongomock`` for Mongo).

The fallback matters. A suite that skips when Docker is not running looks like
coverage without being any, and that is how parity suites rot. Here the default
`uv run pytest` always runs the whole thing; CI additionally runs it against the
containers, where the fakes' approximations cannot hide anything.

Every test gets its own bucket and database, so runs never share state.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from dgml_core.storage import Workspace
from dgml_core.storage_service import StorageConfig
from dgml_storage_s3 import S3MongoStore

S3_ENDPOINT_ENV = "DGML_TEST_S3_ENDPOINT"
MONGO_URI_ENV = "DGML_TEST_MONGO_URI"

#: Whether the suite is pointed at real backing services rather than fakes.
USING_REAL_SERVICES = bool(os.environ.get(S3_ENDPOINT_ENV) or os.environ.get(MONGO_URI_ENV))

needs_real_services = pytest.mark.skipif(
    not USING_REAL_SERVICES,
    reason=f"needs real S3/Mongo; set {S3_ENDPOINT_ENV} and {MONGO_URI_ENV}",
)


@pytest.fixture(autouse=True)
def _fake_backends(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stand up in-process S3 and Mongo unless real services are configured."""
    if USING_REAL_SERVICES:
        monkeypatch.setenv("DGML_MONGO_URI", os.environ[MONGO_URI_ENV])
        yield
        return

    import mongomock
    import pymongo
    from moto import mock_aws

    # boto3 refuses to sign requests without credentials; moto ignores the values.
    for var, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(var, value)
    monkeypatch.delenv("DGML_MONGO_URI", raising=False)
    # store.py imports MongoClient inside __init__, so patching the module
    # attribute before construction is enough.
    monkeypatch.setattr(pymongo, "MongoClient", mongomock.MongoClient)

    with mock_aws():
        yield


@pytest.fixture
def s3_config(tmp_path: Path) -> StorageConfig:
    """A per-test bucket and database, with the bucket created."""
    import boto3

    suffix = uuid.uuid4().hex[:12]
    bucket = f"dgml-test-{suffix}"
    options: dict[str, object] = {
        "bucket": bucket,
        "mongo_database": f"dgml_test_{suffix}",
        "region": "us-east-1",
    }
    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    if endpoint:
        options["endpoint_url"] = endpoint

    client_kwargs = {"region_name": "us-east-1"}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    boto3.client("s3", **client_kwargs).create_bucket(Bucket=bucket)

    return StorageConfig(
        provider="dgml_storage_s3:S3MongoStore", root=tmp_path / "ws", options=options
    )


@pytest.fixture
def store(s3_config: StorageConfig) -> S3MongoStore:
    return S3MongoStore(S3MongoStore.parse_config(s3_config))


@pytest.fixture
def s3_workspace(s3_config: StorageConfig, tmp_path: Path) -> Workspace:
    """A workspace whose config selects the sample store, so the whole pipeline
    (``FileStore``, ``DocSetStore``, ``check_workspace``, attestation) runs
    against S3 + Mongo without any of them knowing."""
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    ws = Workspace(root=root)
    _write_store_config(root / "config.toml", s3_config)
    return ws


@pytest.fixture
def local_workspace(tmp_path: Path) -> Workspace:
    """A plain local-disk workspace, for the cross-backend parity assertions."""
    root = tmp_path / "local-ws"
    (root / "files").mkdir(parents=True)
    (root / "docsets").mkdir(parents=True)
    return Workspace(root=root)


def _write_store_config(path: Path, config: StorageConfig) -> None:
    lines = ["[storage.default]", f'provider = "{config.provider}"']
    for key, value in sorted(config.options.items()):
        rendered = str(value).lower() if isinstance(value, bool) else repr(str(value))
        lines.append(f"{key} = {rendered.replace(chr(39), chr(34))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_text_pdf(path: Path, pages: int = 2) -> Path:
    """A multi-page PDF with real embedded text.

    Text-bearing rather than blank on purpose: digital extraction on a blank PDF
    correctly records a permanent "no words found" error, which surfaces as a
    ``dgml check`` issue and would mask genuine problems.

    Built here rather than imported from the CLI suite's conftest — reaching
    across packages by ``sys.path`` is invisible to the type checker and breaks
    the moment either package moves."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    page_ids = list(range(4, 4 + pages))
    content_ids = list(range(4 + pages, 4 + 2 * pages))
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    add(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page_id, content_id in zip(page_ids, content_ids, strict=True):
        assert (
            add(
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
            )
            == page_id
        )
    for n, content_id in enumerate(content_ids, start=1):
        stream = f"BT /F1 24 Tf 100 700 Td (Invoice page {n} total 100.00) Tj ET\n".encode()
        assert (
            add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream") == content_id
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(out))
    return path


__all__ = [
    "USING_REAL_SERVICES",
    "needs_real_services",
]

# Re-exported so tests can build stores directly when they need two at once.
StoreFactory = Callable[[], S3MongoStore]


@pytest.fixture
def make_pdf() -> Callable[..., Path]:
    """Factory for a text-bearing multi-page PDF.

    A fixture rather than an importable helper: each package's ``tests/`` would
    otherwise need an ``__init__.py`` to make ``from .conftest import …``
    resolvable, and those collide across packages under mypy's module
    resolution. pytest injects conftest fixtures without any import."""
    return _write_text_pdf
