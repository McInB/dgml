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

"""Page rendering via the system ``ghostscript`` binary (the default).

Ghostscript is invoked as a subprocess. It is a system-level dependency,
not a Python package — see CLAUDE.md for the licensing rationale.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import ClassVar

from .errors import PageRenderFailed
from .pages import (
    GS_TIMEOUT_SECONDS,
    PAGE_FILENAME_TEMPLATE,
    PageRenderer,
    RendererName,
    RenderingConfig,
    ghostscript_path,
)


class GhostscriptRenderer(PageRenderer):
    """Rasterize pages with ghostscript's ``png16m`` device."""

    name: ClassVar[RendererName] = RendererName.GHOSTSCRIPT
    config_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: RenderingConfig) -> None:
        # Probing here (not in render) keeps the "renderer missing" failure at
        # construction time, symmetric with the SDK imports in OCR providers —
        # and a cache hit never constructs a renderer, so gs need not be
        # installed to serve one.
        self._gs = ghostscript_path()

    def render(self, pdf_path: Path, output_dir: Path, *, dpi: int) -> None:
        output_template = str(output_dir / PAGE_FILENAME_TEMPLATE)
        cmd = [
            self._gs,
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            "-dSAFER",
            "-sDEVICE=png16m",
            f"-r{dpi}",
            f"-sOutputFile={output_template}",
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
            raise PageRenderFailed(f"ghostscript timed out after {GS_TIMEOUT_SECONDS}s") from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise PageRenderFailed(f"ghostscript exited {result.returncode}: {stderr}")
