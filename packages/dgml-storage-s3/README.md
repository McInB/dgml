# dgml-storage-s3 — sample storage backend

**A sample, not a supported product.** It exists to prove the DGML storage
abstraction actually holds on a backend that is not local disk, and to serve as
a reference for anyone writing their own.

Blobs live in an S3-compatible bucket; documents live in MongoDB collections —
the two APIs `StorageService` was modelled on, so nearly every method is a
one-line delegation.

## Use it

```toml
# <workspace>/config.toml  — local dev against MinIO
[storage.default]
provider = "dgml_storage_s3:S3MongoStore"
bucket = "dgml-dev"
endpoint_url = "http://localhost:9000"
mongo_host = "localhost"
mongo_database = "dgml_dev"
```

Drop `endpoint_url` and set `region` to run against real AWS S3. MinIO is not a
separate backend — it speaks the S3 API, so it is only a different address.

## Credentials

**Never put credentials in DGML config.** S3 uses boto3's default chain
(`AWS_ACCESS_KEY_ID`, `~/.aws/credentials`, IAM role); MongoDB reads
`DGML_MONGO_URI` if set.

This is not stylistic. `dgml_core.storage_resolve` decides what is a secret by
**substring match on the option name** (`key`, `secret`, `token`, `password`,
`credential`); anything else is persisted to the plaintext workspace registry.
An option named `mongo_uri` holding `mongodb://admin:hunter2@host` matches none
of them, so the password would be written out in the clear. If you add an
inline-credential option, its name must contain one of those substrings.

## Run the tests

```bash
docker compose -f packages/dgml-storage-s3/docker-compose.yml up -d
DGML_TEST_S3_ENDPOINT=http://localhost:9000 \
DGML_TEST_MONGO_URI=mongodb://localhost:27017 \
  uv run pytest packages/dgml-storage-s3
```

With those variables unset the suite still runs, against in-process fakes
(`moto` for S3, `mongomock` for Mongo) — so it never silently skips. The fakes
verify *our* logic; the containers verify wire behaviour (pagination, real
error codes, multipart). CI runs both.

## If you adapt this for production

It inherits the default path bridges (`materialize`, `staged_write`,
`working_dir`) rather than overriding them. That is deliberate — those defaults
are what every third-party store gets for free, so inheriting them means they
are exercised against a real backend. But they stage through `tempfile`, and
`TMPDIR` is a RAM-backed tmpfs on many container images: a large
`staged_write` (a few hundred page images) would allocate into memory rather
than disk. A production store should override them to stage under
`StorageConfig.root`, as the `StorageService` docstring describes.

## Licences

| component | licence | how used |
|---|---|---|
| `boto3`, `pymongo` | Apache-2.0 | dependencies of this package ✅ |
| `moto`, `mongomock` | Apache-2.0 / ISC | dev/test only ✅ |
| MinIO server | AGPL-3.0 | dev/CI container, never redistributed |
| MongoDB Community | SSPL | dev/CI container, never redistributed |

The root `CLAUDE.md` bans SSPL and AGPL for **Python packages shipped inside the
wheel**. The two servers are network services we invoke, not code we ship — the
same reasoning that already permits ghostscript. `pip-licenses` will not flag
them, which is why it is written down here. [FerretDB](https://ferretdb.com)
(Apache-2.0) speaks the MongoDB wire protocol and is a drop-in for the compose
file if the SSPL question ever becomes live; `pymongo` talks to both unchanged.
