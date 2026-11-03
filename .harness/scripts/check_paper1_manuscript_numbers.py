"""Check that every decimal number in the Paper 1 manuscript exists in a source document.

The manuscript (papers/preserve_vs_reify/preserve_vs_reify.tex) introduces no new measurement; every
figure is carried over from a results document that is itself checked against outputs/. This script
extracts each decimal number from the manuscript body (comments stripped) and fails if one does not
appear verbatim in any source document, so a typo or a stale number cannot slip into the paper.

    uv run --frozen python .harness/scripts/check_paper1_manuscript_numbers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "papers" / "preserve_vs_reify" / "preserve_vs_reify.tex"
SOURCES = [
    ROOT / "docs" / "hnn-ontology-encoder-v2-results.md",
    ROOT / "docs" / "hnn-ontology-paper1-draft-en.md",
    ROOT / "docs" / "stare-input-representation-v1-results.md",
    ROOT / "docs" / "stare-native-encoder-v1-results.md",
    ROOT / "docs" / "related-work-hkg-comparison.md",
    ROOT / "docs" / "charter-stare-input-representation-v1.md",
    ROOT / "docs" / "charter-stare-native-encoder-v1.md",
    ROOT / "docs" / "hahe-same-harness-v1-results.md",
]
# Not measurements: software versions (2.10, 2.6.1), the minimum sign-flip p 2/32 (0.0625), label smoothing (0.1).
# Derived in the text from source numbers (checked by hand on 2026-10-28):
#   0.0095 = 0.6648 - 0.6553 (Study 3 preserve minus Study 2 native StarE)
#   0.002  bound on |0.6553 - 0.654| = 0.0013 (port reproduction)
#   0.13   = 0.478 - 0.348 (GLoRE pair vs node aggregation, related-work comparison)
#   0.09   reified lead at epochs 70-80 in Study 2 (seed means 0.465 vs 0.372, stare results doc)
#   1.5    = 1537 / 1022 MB (Study 2 peak VRAM);  1.6 = (5,068,016 - 4,989,016) / 4,989,016 in %
#   4.9    = (5,318,216 - 5,068,016) / 5,068,016 in %
ALLOWED = {"2.10", "2.6.1", "0.0625", "0.1", "0.09", "0.13", "0.002", "0.0095", "1.5", "1.6", "4.9"}
NUMBER = re.compile(r"(?<![\w.])[+-]?\d+\.\d+(?![\w.])")
GRAPHICS_OPTIONS = re.compile(r"\\includegraphics\[[^\]]*\]")


def body(tex: str) -> str:
    lines = [line.split("%")[0] if not line.lstrip().startswith("%") else "" for line in tex.splitlines()]
    text = "\n".join(lines)
    return GRAPHICS_OPTIONS.sub("", text[text.index("\\begin{document}"):])


def main() -> int:
    tex = body(TEX.read_text(encoding="utf-8"))
    corpus = "\n".join(path.read_text(encoding="utf-8") for path in SOURCES)
    corpus_numbers = {n.lstrip("+-") for n in NUMBER.findall(corpus)}
    missing = []
    found = 0
    for raw in sorted({n.lstrip("+-") for n in NUMBER.findall(tex)}):
        if raw in ALLOWED:
            continue
        if raw in corpus_numbers:
            found += 1
        else:
            missing.append(raw)
    print(f"checked {found + len(missing)} distinct decimal numbers in {TEX.name}; {found} found in the sources")
    if missing:
        print("NOT FOUND in any source document:", ", ".join(missing))
        return 1
    print("every decimal number in the manuscript appears in a source document")
    return 0


if __name__ == "__main__":
    sys.exit(main())
