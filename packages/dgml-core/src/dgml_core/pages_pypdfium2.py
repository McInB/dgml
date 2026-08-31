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

"""Page rendering via PDFium (the ``pypdfium2`` Python package).

An in-process alternative to the ghostscript subprocess: no system binary
to install, just ``pip install dgml[pdfium]``. pypdfium2 is
(Apache-2.0 OR BSD-3-Clause) and PDFium itself is BSD-3-Clause, both on
the direct-dependency allow-list; Pillow (MIT-CMU) writes the PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from .errors import PageRenderFailed, RendererNotAvailable
from .pages import (
    PAGE_FILENAME_TEMPLATE,
    PageRenderer,
    RendererName,
    RenderingConfig,
)

# PDF user space is 72 points per inch; PDFium takes a scale factor, not a dpi.
_PDF_POINTS_PER_INCH = 72.0


class Pypdfium2Renderer(PageRenderer):
    """Rasterize pages in-process with PDFium."""

    name: ClassVar[RendererName] = RendererName.PYPDFIUM2
    config_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: RenderingConfig) -> None:
        try:
            import pypdfium2
        except ImportError as exc:
            raise RendererNotAvailable(
                "rendering.provider is 'pypdfium2' but the pypdfium2 package is "
                "not installed — install it with: pip install dgml[pdfium]"
            ) from exc
        # Pillow writes the PNGs (pypdfium2's to_pil()); it ships with the
        # same extra, so its absence means a partial/hand-rolled install.
        try:
            import PIL.Image  # noqa: F401
        except ImportError as exc:
            raise RendererNotAvailable(
                "rendering.provider is 'pypdfium2' but Pillow is not installed — "
                "install it with: pip install dgml[pdfium]"
            ) from exc
        self._pdfium = pypdfium2

    def render(self, pdf_path: Path, output_dir: Path, *, dpi: int) -> None:
        scale = dpi / _PDF_POINTS_PER_INCH
        try:
            pdf = self._pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:  # PdfiumError, or OSError on unreadable input
            raise PageRenderFailed(f"pypdfium2 could not open {pdf_path.name}: {exc}") from exc
        try:
            for index in range(len(pdf)):
                page_num = index + 1
                try:
                    page = pdf[index]
                    bitmap = page.render(scale=scale)
                    image = bitmap.to_pil()
                    image.save(output_dir / (PAGE_FILENAME_TEMPLATE % page_num), format="PNG")
                except Exception as exc:
                    raise PageRenderFailed(
                        f"pypdfium2 failed rendering page {page_num} of {pdf_path.name}: {exc}"
                    ) from exc
        finally:
            pdf.close()
