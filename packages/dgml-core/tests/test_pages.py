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

"""Tests for the page-render abstraction: the ghostscript default, the
pypdfium2 provider, the ``[rendering]`` config loader, and the render cache.

Ghostscript is faked by monkeypatching ``subprocess.run`` (and the binary
probe in ``pages_ghostscript``) so these tests run without the system binary
and can assert exactly how many times the renderer is invoked — the whole
point of the cache. The pypdfium2 tests run against the real PDFium (a dev
dependency, no system binary needed).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from dgml_core import pages, pages_ghostscript
from dgml_core.errors import RenderingConfigInvalid
from dgml_core.pages import (
    PAGE_CACHE_ENV,
    RendererName,
    RenderingConfig,
    load_rendering_config,
    render_pages,
)
from dgml_core.storage import Workspace

from .conftest import _write_text_pdf, write_config


def _fake_gs_factory(n_pages: int, counter: list[int]) -> object:
    """Return a fake ``subprocess.run`` that "renders" ``n_pages`` PNGs.

    Parses the ``-sOutputFile=.../page_%d.png`` template out of the command
    and writes one file per page, mirroring ghostscript's own numbering, then
    records the invocation in ``counter``.
    """

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        counter.append(1)
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        for i in range(1, n_pages + 1):
            Path(template.replace("%d", str(i))).write_bytes(b"\x89PNG\r\n\x1a\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake bytes for hashing")
    return p


def test_no_cache_env_always_renders(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    assert render_pages(pdf, tmp_path / "out1") == 2
    assert render_pages(pdf, tmp_path / "out2") == 2
    # No cache configured: ghostscript runs for every call.
    assert len(calls) == 2


def test_cache_miss_populates_then_hit_skips_ghostscript(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(3, calls))

    out1 = tmp_path / "out1"
    assert render_pages(pdf, out1) == 3
    assert len(calls) == 1  # miss -> one render
    assert sorted(p.name for p in out1.glob("page_*.png")) == [
        "page_1.png",
        "page_2.png",
        "page_3.png",
    ]

    # A later render of identical bytes into a fresh dir is served from cache;
    # ghostscript must not be invoked again — assert by making it fail loudly.
    monkeypatch.setattr(
        pages_ghostscript,
        "ghostscript_path",
        lambda: (_ for _ in ()).throw(AssertionError("gs called")),
    )
    out2 = tmp_path / "out2"
    assert render_pages(pdf, out2) == 3
    assert len(calls) == 1  # still one — the hit copied from cache
    assert sorted(p.name for p in out2.glob("page_*.png")) == [
        "page_1.png",
        "page_2.png",
        "page_3.png",
    ]


def test_cache_key_differs_by_content(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(1, calls))

    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 completely different bytes")

    render_pages(pdf, tmp_path / "a")
    render_pages(other, tmp_path / "b")
    # Distinct content -> distinct cache keys -> two renders, no false hit.
    assert len(calls) == 2


def test_partial_cache_entry_is_treated_as_miss(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    # A cache entry with PNGs but no `.complete` marker (e.g. an interrupted
    # writer) must not be trusted — render_pages should re-render.
    entry = cache / pages._pdf_cache_key(pdf, pages.DEFAULT_DPI)
    entry.mkdir(parents=True)
    (entry / "page_1.png").write_bytes(b"stale")

    assert render_pages(pdf, tmp_path / "out") == 2
    assert len(calls) == 1


def test_dpi_reaches_ghostscript(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    seen: list[list[str]] = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        Path(template.replace("%d", "1")).write_bytes(b"\x89PNG\r\n\x1a\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", capture)

    render_pages(pdf, tmp_path / "out", dpi=150)
    assert "-r150" in seen[0]


def test_cache_entries_do_not_collide_across_dpi(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same bytes at two resolutions are different renders. Serving the 300-dpi
    # entry for a 150-dpi request would hand back page images that disagree
    # with the page_text/ boxes written alongside them.
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    assert render_pages(pdf, tmp_path / "a", dpi=300) == 2
    assert render_pages(pdf, tmp_path / "b", dpi=150) == 2
    assert len(calls) == 2, "the second dpi must miss, not reuse the first render"
    assert pages._pdf_cache_key(pdf, 300) != pages._pdf_cache_key(pdf, 150)

    # ...and each is still cached in its own right.
    assert render_pages(pdf, tmp_path / "c", dpi=150) == 2
    assert len(calls) == 2


def test_cache_entries_do_not_collide_across_renderers(pdf: Path) -> None:
    # Same bytes at the same dpi through different backends are different
    # renders — the pixels differ subtly (anti-aliasing, ±1px rounding).
    key_gs = pages._pdf_cache_key(pdf, 300, RendererName.GHOSTSCRIPT)
    key_pdfium = pages._pdf_cache_key(pdf, 300, RendererName.PYPDFIUM2)
    assert key_gs != key_pdfium
    # The default parameter is the ghostscript default, so pre-existing cache
    # entries keyed before renderers were configurable stay valid.
    assert pages._pdf_cache_key(pdf, 300) == key_gs


# ---------------------------------------------------------------------------
# The pypdfium2 renderer (real PDFium — a dev dependency, no system binary)
# ---------------------------------------------------------------------------


def test_pypdfium2_renders_all_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "doc.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello World", "Second Page"])
    # Ghostscript must play no part — fail loudly if its probe runs.
    monkeypatch.setattr(
        pages_ghostscript,
        "ghostscript_path",
        lambda: (_ for _ in ()).throw(AssertionError("gs called")),
    )

    out = tmp_path / "out"
    config = RenderingConfig(provider=RendererName.PYPDFIUM2)
    assert render_pages(pdf_path, out, dpi=72, config=config) == 2
    pngs = sorted(p.name for p in out.glob("page_*.png"))
    assert pngs == ["page_1.png", "page_2.png"]
    for name in pngs:
        assert (out / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_pypdfium2_dpi_scales_pixel_dimensions(tmp_path: Path) -> None:
    # US Letter is 612x792 pt; at 72 dpi that's 612x792 px, at 144 dpi doubled.
    from dgml_core.ocr import _image_dimensions

    pdf_path = tmp_path / "doc.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"])
    config = RenderingConfig(provider=RendererName.PYPDFIUM2)

    out72 = tmp_path / "out72"
    render_pages(pdf_path, out72, dpi=72, config=config)
    assert _image_dimensions((out72 / "page_1.png").read_bytes()) == (612, 792)

    out144 = tmp_path / "out144"
    render_pages(pdf_path, out144, dpi=144, config=config)
    assert _image_dimensions((out144 / "page_1.png").read_bytes()) == (1224, 1584)


def test_pypdfium2_unreadable_pdf_raises_page_render_failed(tmp_path: Path) -> None:
    from dgml_core.errors import PageRenderFailed

    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"not a pdf at all")
    config = RenderingConfig(provider=RendererName.PYPDFIUM2)
    with pytest.raises(PageRenderFailed, match="pypdfium2"):
        render_pages(bogus, tmp_path / "out", config=config)


def test_pypdfium2_render_is_served_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    pdf_path = tmp_path / "doc.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"])
    config = RenderingConfig(provider=RendererName.PYPDFIUM2)

    assert render_pages(pdf_path, tmp_path / "a", config=config) == 1
    # A hit constructs no renderer at all — prove it by making construction fail.
    monkeypatch.setattr(
        pages,
        "make_renderer",
        lambda cfg: (_ for _ in ()).throw(AssertionError("renderer constructed on a cache hit")),
    )
    assert render_pages(pdf_path, tmp_path / "b", config=config) == 1


# ---------------------------------------------------------------------------
# load_rendering_config — reading the [rendering] section of config.toml
# ---------------------------------------------------------------------------


def test_rendering_config_defaults_to_ghostscript(workspace: Workspace) -> None:
    assert load_rendering_config(workspace) == RenderingConfig(provider=RendererName.GHOSTSCRIPT)


def test_rendering_config_empty_section_is_default(workspace: Workspace) -> None:
    write_config(workspace, {"rendering": {}})
    assert load_rendering_config(workspace).provider is RendererName.GHOSTSCRIPT


def test_rendering_config_selects_pypdfium2(workspace: Workspace) -> None:
    write_config(workspace, {"rendering": {"provider": "pypdfium2"}})
    assert load_rendering_config(workspace).provider is RendererName.PYPDFIUM2


def test_rendering_config_rejects_unknown_provider(workspace: Workspace) -> None:
    write_config(workspace, {"rendering": {"provider": "poppler"}})
    with pytest.raises(RenderingConfigInvalid, match="poppler"):
        load_rendering_config(workspace)


def test_rendering_config_rejects_unknown_fields(workspace: Workspace) -> None:
    write_config(workspace, {"rendering": {"provider": "ghostscript", "dpi": 300}})
    with pytest.raises(RenderingConfigInvalid, match="dpi"):
        load_rendering_config(workspace)
