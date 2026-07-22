#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def notebook_source(path: Path) -> str:
    nb = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} NOTEBOOK.ipynb", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    source = notebook_source(path)

    require("GENCODE_GTF" in source, "GENCODE_GTF config is missing")
    require("gencode.v49.annotation.sorted.gtf" in source, "GENCODE v49 sorted GTF is not configured")
    require("TARGET_MODE = 'three_prime_exon'" in source, "target mode is not three_prime_exon")
    require("load_transcript_exons" in source, "transcript exon loader is missing")
    require("select_three_prime_exon" in source, "3-prime exon selector is missing")
    require("bw_mean_regions" in source, "multi-region bigWig mean function is missing")
    require("expression_regions" in source, "GeneRecord does not carry expression_regions")

    target_match = re.search(
        r"def _target\(self, sample: str, gene: GeneRecord\).*?return torch\.tensor",
        source,
        re.S,
    )
    require(target_match is not None, "_target method not found")
    target_source = target_match.group(0)
    require("bw_mean_regions" in target_source, "_target does not use exon-region bigWig mean")
    require("TARGET_HALF_WINDOW_BP" not in target_source, "_target still uses TSS +/- window")
    require("gene.tss - TARGET_HALF_WINDOW_BP" not in source, "old TSS-window target code remains")

    print(f"{path}: target definition checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
