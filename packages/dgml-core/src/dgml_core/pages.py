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

"""PDF page rendering — abstract renderer interface, config loader, dispatcher.

Loads the ``[rendering]`` section from ``<workspace>/config.toml`` (via
:func:`load_rendering_config`), dispatches :func:`render_pages` to the
configured renderer, and owns everything renderer-independent: the page
filename contract (``page_N.png``), the optional content-addressed render
cache, and PDF page counting / slicing helpers.

Renderer implementations live in sibling modules so this file stays
focused on the abstraction (mirroring :mod:`dgml_core.ocr`):

- :class:`dgml_core.pages_ghostscript.GhostscriptRenderer` — the system
  ``ghostscript`` binary, invoked as a subprocess (the zero-config default;
  see CLAUDE.md for the licensing rationale)
- :class:`dgml_core.pages_pypdfium2.Pypdfium2Renderer` — PDFium via the
  ``pypdfium2`` Python package (``pip install dgml[pdfium]``)

Adding a new renderer
---------------------

1. Add a value to :class:`RendererName`.
2. Create a new module ``pages_<name>.py`` with a subclass of
   :class:`PageRenderer`. Implement ``__init__`` (lazy-import the package
   or probe the binary; raise :class:`RendererNotAvailable` if missing)
   and ``render``.
3. Wire the subclass into ``_build_registry`` below.
4. If the renderer takes provider-specific config fields, declare them in
   ``config_fields`` and validate them in ``parse_config``.

PDF *slicing* (:func:`extract_pdf_pages`) is not part of the renderer
abstraction — it always goes through ghostscript's ``pdfwrite`` device.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from .errors import GhostscriptNotFound, RenderingConfigInvalid
from .models_config import ConfigSection

if TYPE_CHECKING:
    from .storage import Workspace

DEFAULT_DPI = 300
PAGE_FILENAME_TEMPLATE = "page_%d.png"
PAGE_GLOB = "page_*.png"

# Optional content-addressed render cache. When ``$DGML_PAGE_CACHE`` names a
# directory, :func:`render_pages` copies its output there keyed by the PDF's
# content hash (plus renderer + dpi) and, on a later call for identical bytes,
# copies back instead of re-rendering. Off by default — rendering is
# unchanged unless the env var is set. Intended for workflows that re-ingest the
# same PDFs into many workspaces (e.g. the clustering sweep's per-cell
# workspaces), where the render is otherwise repeated once per workspace.
PAGE_CACHE_ENV = "DGML_PAGE_CACHE"
_CACHE_COMPLETE_MARKER = ".complete"

GS_TIMEOUT_SECONDS = 600

# On Windows the console executable is named ``gswin64c`` / ``gswin32c``, not
# ``gs`` (the Artifex installer ships no ``gs.exe``). Probe those first, then
# fall back to ``gs`` for MSYS/Cygwin shells that expose the Unix name.
GS_BINARIES: tuple[str, ...] = (
    ("gswin64c", "gswin32c", "gs") if sys.platform == "win32" else ("gs",)
)


class RendererName(StrEnum):
    """Identifier of a page-render backend, as written in workspace config."""

    GHOSTSCRIPT = "ghostscript"
    PYPDFIUM2 = "pypdfium2"


# The renderer used when a workspace declares no [rendering] config: the
# system ghostscript binary, DGML's original renderer. Unlike OCR there is
# no platform split and no warning — ghostscript is the documented default.
DEFAULT_RENDERER = RendererName.GHOSTSCRIPT


@dataclass(frozen=True)
class RenderingConfig:
    """Parsed ``rendering`` section of the workspace config.

    By construction (via :func:`load_rendering_config`) this object is
    well-formed for the renderer it names. No renderer takes extra config
    fields today; the dataclass exists so adding one later (e.g. a pixel
    format) is not a signature change for every caller.
    """

    provider: RendererName = DEFAULT_RENDERER


def load_rendering_config(workspace: Workspace) -> RenderingConfig:
    """Read and validate the ``rendering`` section of ``<workspace>/config.toml``.

    When the merged config has no ``rendering`` section — or an empty one —
    defaults to ghostscript (:data:`DEFAULT_RENDERER`), silently: unlike OCR
    there is a built-in default on every platform. Raises
    :class:`RenderingConfigInvalid` when a section exists but is malformed.
    """
    # Imported lazily: config.py imports storage.py which must not need us first.
    from .config import load_merged_config

    section = load_merged_config(workspace).get(ConfigSection.RENDERING)
    if not section:
        # Absent or empty — `provider` is what selects a backend, so a bare
        # `[rendering]` is the same as none at all.
        return RenderingConfig()
    if not isinstance(section, dict):
        raise RenderingConfigInvalid("'rendering' must be a table")

    provider_str = section.get("provider")
    valid_providers = [r.value for r in RendererName]
    if provider_str not in valid_providers:
        raise RenderingConfigInvalid(
            f"'rendering.provider' must be one of {valid_providers} (got {provider_str!r})"
        )
    return _RENDERERS[RendererName(provider_str)].parse_config(section)


# ---------------------------------------------------------------------------
# Renderer interface
# ---------------------------------------------------------------------------


class PageRenderer(ABC):
    """Common interface for PDF page-image backends.

    Implementations are constructed from a :class:`RenderingConfig` (which
    is where lazy package imports / binary probes live) and implement
    :meth:`render` for a whole PDF. The shared wrapper :func:`render_pages`
    handles the render cache, clearing stale page images, and counting the
    output — renderers only need to write ``page_N.png`` files.

    Subclasses must declare ``config_fields`` listing the TOML keys they
    accept under ``rendering.*`` (besides the universal ``provider`` key);
    anything else is rejected by :meth:`_check_no_extra_fields` to catch
    typos and stale-after-switching-provider fields.
    """

    name: ClassVar[RendererName]
    config_fields: ClassVar[frozenset[str]]

    @classmethod
    def _check_no_extra_fields(cls, section: dict[str, Any]) -> None:
        """Raise :class:`RenderingConfigInvalid` for any keys in ``section``
        not in ``cls.config_fields`` (or the universal ``provider``)."""
        allowed = cls.config_fields | {"provider"}
        unknown = set(section.keys()) - allowed
        if unknown:
            raise RenderingConfigInvalid(
                f"unknown fields in 'rendering' for provider {cls.name.value!r}: "
                f"{sorted(unknown)}. Allowed: {sorted(allowed)}"
            )

    @classmethod
    def parse_config(cls, section: dict[str, Any]) -> RenderingConfig:
        """Build a :class:`RenderingConfig` from the ``rendering`` section of
        the workspace config (a plain TOML table as a dict).

        The default implementation rejects foreign or misspelled keys and
        returns a config naming this renderer — sufficient while renderers
        take no extra fields. A renderer that grows its own fields overrides
        this to validate them (raising :class:`RenderingConfigInvalid`)."""
        cls._check_no_extra_fields(section)
        return RenderingConfig(provider=cls.name)

    @abstractmethod
    def __init__(self, config: RenderingConfig) -> None:
        """Prepare the backend: lazy-import its package or probe its binary.
        Raise :class:`RendererNotAvailable` (or its ghostscript-specific
        subclass :class:`GhostscriptNotFound`) with an actionable install
        hint when the backend is missing."""

    @abstractmethod
    def render(self, pdf_path: Path, output_dir: Path, *, dpi: int) -> None:
        """Rasterize every page of ``pdf_path`` into ``output_dir``.

        Write one PNG per page named per :data:`PAGE_FILENAME_TEMPLATE`
        (``page_1.png`` …, 1-based, ghostscript's own numbering).
        ``output_dir`` exists and holds no stale page images when called.

        Raise :class:`PageRenderFailed` for backend errors. Renderers may
        be called for many PDFs from one process but are not called
        concurrently for the same output directory.
        """


def make_renderer(config: RenderingConfig) -> PageRenderer:
    """Instantiate the renderer class for ``config.provider``."""
    cls = _RENDERERS.get(config.provider)
    if cls is None:  # defensive — load_rendering_config validates already
        raise RenderingConfigInvalid(f"no renderer implementation for {config.provider!r}")
    return cls(config)


# ---------------------------------------------------------------------------
# Renderer-independent helpers
# ---------------------------------------------------------------------------


def ghostscript_path() -> str:
    """Return the absolute path to the ghostscript binary or raise :class:`GhostscriptNotFound`."""
    for name in GS_BINARIES:
        found = shutil.which(name)
        if found is not None:
            return found
    raise GhostscriptNotFound(
        f"ghostscript ({'/'.join(GS_BINARIES)}) is not installed or not on PATH"
    )


def pdf_page_count(path: Path) -> int:
    """Return the page count of ``path`` by walking pdfminer's page tree.

    Uses ``PDFPage.create_pages`` (a page-tree traversal, no layout analysis),
    so it's cheap and avoids trusting the possibly-wrong ``/Count`` field.
    """
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    with path.open("rb") as fh:
        document = PDFDocument(PDFParser(fh))
        return sum(1 for _ in PDFPage.create_pages(document))


def extract_pdf_pages(pdf_path: Path, output_path: Path, page_numbers: Sequence[int]) -> None:
    """Write a new PDF at ``output_path`` containing only ``page_numbers``.

    ``page_numbers`` are 1-based and may be non-contiguous; ghostscript's
    ``-sPageList`` emits the selected pages in ascending document order. Uses
    the ``pdfwrite`` device, so no Python PDF library is involved. This is
    PDF slicing, not rasterization — it always goes through ghostscript,
    independent of the configured page renderer.
    """
    import subprocess

    from .errors import PdfSliceFailed

    if not page_numbers:
        raise ValueError("page_numbers must be non-empty")

    gs = ghostscript_path()
    page_list = ",".join(str(n) for n in page_numbers)
    cmd = [
        gs,
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        f"-sPageList={page_list}",
        f"-sOutputFile={output_path}",
        str(pdf_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=GS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PdfSliceFailed(f"ghostscript timed out after {GS_TIMEOUT_SECONDS}s") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise PdfSliceFailed(f"ghostscript exited {result.returncode}: {stderr}")
    if not output_path.exists():
        raise PdfSliceFailed(f"ghostscript wrote no output for pages {page_list}")


def _page_cache_root() -> Path | None:
    """Cache directory from ``$DGML_PAGE_CACHE``, or ``None`` when unset/empty."""
    root = os.environ.get(PAGE_CACHE_ENV)
    return Path(root) if root else None


def _pdf_cache_key(pdf_path: Path, dpi: int, renderer: RendererName = DEFAULT_RENDERER) -> str:
    """Content hash keying the render cache: renderer + dpi + the PDF bytes.

    Renderer and dpi are folded in so a change to either invalidates entries
    rather than serving mismatched renders for the same bytes — a 150-dpi
    render and a 300-dpi render of the same PDF get distinct entries, and so
    do a ghostscript render and a pypdfium2 render (their pixels differ).
    """
    digest = hashlib.sha256(f"{renderer.value}:{dpi}\n".encode())
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_pages_from(src_dir: Path, output_dir: Path) -> int:
    """Clear ``output_dir``'s page PNGs and copy ``src_dir``'s in; return count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_GLOB):
        existing.unlink()
    pages = sorted(src_dir.glob(PAGE_GLOB))
    for png in pages:
        shutil.copy2(png, output_dir / png.name)
    return len(pages)


def render_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = DEFAULT_DPI,
    config: RenderingConfig | None = None,
) -> int:
    """Render each PDF page to a PNG at ``dpi``. Returns the number of pages written.

    ``config`` selects the backend; ``None`` means the ghostscript default
    (callers with a workspace at hand should pass
    :func:`load_rendering_config`'s result instead). Stale page images in
    ``output_dir`` are removed first so retries do not leave orphans behind.

    ``dpi`` trades resolution for speed and disk: 300 (the default) is archival
    quality; ~150 roughly halves rasterization time and file size and is usually
    ample for OCR and the downscaled clustering vision encoder. Whatever value
    is used here also has to reach digital text extraction, since ``page_text/``
    word boxes are expressed in *this* render's pixel space.

    When ``$DGML_PAGE_CACHE`` is set, an identical PDF (same bytes, renderer,
    and dpi) rendered before is served from that cache without invoking
    the backend (which then need not even be installed); otherwise the render
    is populated into the cache on success.

    PNG (not JPEG) is the canonical format: pixel-perfect for text-on-white
    document scans, ~5-10x smaller on disk than JPEG q92 for these
    workloads, and the same format the generation pipeline already
    consumes for LLM input — so workspace renders can be reused directly
    rather than re-rasterized through a second renderer.
    """
    if config is None:
        config = RenderingConfig()

    cache_entry: Path | None = None
    cache_root = _page_cache_root()
    if cache_root is not None:
        cache_entry = cache_root / _pdf_cache_key(pdf_path, dpi, config.provider)
        if (cache_entry / _CACHE_COMPLETE_MARKER).exists():
            return _replace_pages_from(cache_entry, output_dir)

    renderer = make_renderer(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_GLOB):
        existing.unlink()

    renderer.render(pdf_path, output_dir, dpi=dpi)

    count = len(list(output_dir.glob(PAGE_GLOB)))
    if cache_entry is not None:
        _populate_cache(cache_entry, output_dir)
    return count


def _populate_cache(cache_entry: Path, output_dir: Path) -> None:
    """Best-effort copy of the fresh render into the cache, marked complete last.

    Writing the ``.complete`` marker only after every PNG is copied means a
    reader either sees a fully populated entry or treats it as a miss — never a
    partial one. Cache I/O failures are swallowed: the render already succeeded.
    """
    try:
        cache_entry.mkdir(parents=True, exist_ok=True)
        for png in output_dir.glob(PAGE_GLOB):
            shutil.copy2(png, cache_entry / png.name)
        (cache_entry / _CACHE_COMPLETE_MARKER).write_text("", encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Renderer registry
#
# Built at module load by a function call so the renderer modules' imports
# of this module see a fully-defined PageRenderer ABC and RenderingConfig
# dataclass. Doing the import here (rather than at the top of the file)
# avoids a circular dependency: pages_ghostscript / pages_pypdfium2 import
# PageRenderer from us. (Same pattern as dgml_core.ocr's _PROVIDERS.)
# ---------------------------------------------------------------------------


def _register_renderers(
    classes: list[type[PageRenderer]],
) -> dict[RendererName, type[PageRenderer]]:
    """Build a name-keyed registry from a list of renderer classes.

    Iterating a list (rather than constructing a dict literal) lets us
    detect collisions: two renderers claiming the same
    :class:`RendererName` is a copy-paste bug that would otherwise
    silently overwrite. Raising here keeps the failure at import time,
    before any render call.
    """
    registry: dict[RendererName, type[PageRenderer]] = {}
    for cls in classes:
        if cls.name in registry:
            existing = registry[cls.name].__name__
            raise RuntimeError(
                f"duplicate PageRenderer registration for {cls.name.value!r}: "
                f"{existing} and {cls.__name__}"
            )
        registry[cls.name] = cls
    return registry


def _build_registry() -> dict[RendererName, type[PageRenderer]]:
    from .pages_ghostscript import GhostscriptRenderer
    from .pages_pypdfium2 import Pypdfium2Renderer

    return _register_renderers([GhostscriptRenderer, Pypdfium2Renderer])


_RENDERERS: dict[RendererName, type[PageRenderer]] = _build_registry()
