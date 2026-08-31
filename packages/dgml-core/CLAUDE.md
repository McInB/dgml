# `dgml-core` package

The **library** behind the `dgml` CLI. Distribution name `dgml-core`, import
name `dgml_core`. Everything that turns a PDF into DGML lives here: the
PDF→DGML generation pipeline, OCR, page-image rendering, digital/OCR/hybrid
text extraction, grounding, classification, attestation, the LLM client, and
workspace/storage CRUD.

The `dgml` package (the CLI) depends on this one and is the only first-party
caller; `translators-pdf` also depends on it (for the `DocConverter` ABC in
[src/dgml_core/conversion.py](src/dgml_core/conversion.py)). Nothing here may
import `dgml` (the CLI) — the dependency is strictly one-way.

## Public API

The supported library surface is what
[src/dgml_core/__init__.py](src/dgml_core/__init__.py) exports (e.g.
`Workspace`, `FileStore`, `DocSetStore`, the error hierarchy, attestation
helpers). Anything not exported is internal and may change without notice
pre-1.0. Consumers `import dgml_core`; `from dgml import …` is intentionally
unsupported (the CLI package re-exports nothing).

## Optional extras

`aws`, `azure`, `macos`, `pdfium`, `clustering`, and `chain` are declared
here. The `dgml` CLI mirrors them as pass-throughs, so `pip install dgml[aws]`
resolves to `dgml-core[aws]`. Keep the two extra lists in sync when you add or
rename one.

## OCR providers

`--text-mode ocr` dispatches through an `OcrProvider` ABC defined in
[src/dgml_core/ocr.py](src/dgml_core/ocr.py). Concrete providers live in sibling
modules — `src/dgml_core/ocr_aws.py`, `src/dgml_core/ocr_azure.py`,
`src/dgml_core/ocr_macos.py` — and register themselves via the `_PROVIDERS`
dict at the bottom of `ocr.py`.

Each provider owns three things: its SDK lazy-import (in `__init__`),
its config-section validation (`parse_config` classmethod), and its
per-page API call (`analyze_image`). The shared loop in
`extract_text_ocr` handles filesystem I/O and result aggregation —
providers never touch the disk.

To add a new provider: see the "Adding a new provider" section in the
[src/dgml_core/ocr.py](src/dgml_core/ocr.py) module docstring.

## Page renderers

Page-image rendering follows the same shape: a `PageRenderer` ABC, config
loader (`load_rendering_config`, the `[rendering]` section), and registry
live in [src/dgml_core/pages.py](src/dgml_core/pages.py); concrete renderers
live in sibling modules — `src/dgml_core/pages_ghostscript.py` (the default,
a subprocess over the system `gs` binary) and
`src/dgml_core/pages_pypdfium2.py` (PDFium in-process, the `pdfium` extra).
Renderers only write `page_N.png` files; the shared `render_pages` wrapper
owns the `$DGML_PAGE_CACHE` cache, stale-image cleanup, and page counting.
PDF *slicing* (`extract_pdf_pages`) is not provider-based — it always uses
ghostscript's `pdfwrite`.

To add a new renderer: see the "Adding a new renderer" section in the
[src/dgml_core/pages.py](src/dgml_core/pages.py) module docstring.
