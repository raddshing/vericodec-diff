from __future__ import annotations

import csv
import re
import sys
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.prompt_manifests import MANIFEST_SPECS, REQUIRED_COLUMNS, validate_manifest


MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"
QUOTE_RE = re.compile(r'"[^"\n]+"')


def load_rows(name: str) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with (MANIFESTS_DIR / name).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(reader.fieldnames or ()), list(reader)


class PromptManifestTests(unittest.TestCase):
    def test_schema_matches_exact_required_columns(self) -> None:
        for manifest_name in MANIFEST_SPECS:
            with self.subTest(manifest=manifest_name):
                fieldnames, _ = load_rows(manifest_name)
                self.assertEqual(fieldnames, REQUIRED_COLUMNS)

    def test_validator_accepts_all_manifests(self) -> None:
        for manifest_name in MANIFEST_SPECS:
            with self.subTest(manifest=manifest_name):
                summary = validate_manifest(MANIFESTS_DIR / manifest_name)
                self.assertEqual(summary["manifest"], manifest_name)
                self.assertEqual(summary["total"], MANIFEST_SPECS[manifest_name]["total"])

    def test_counts_ids_and_non_empty_values(self) -> None:
        for manifest_name, spec in MANIFEST_SPECS.items():
            with self.subTest(manifest=manifest_name):
                _, rows = load_rows(manifest_name)
                self.assertEqual(len(rows), spec["total"])
                self.assertEqual(Counter(row["category"] for row in rows), Counter(spec["category_counts"]))
                self.assertEqual(len(rows), len({row["prompt_id"] for row in rows}))
                self.assertTrue(all(row["prompt"].strip() for row in rows))
                self.assertTrue(all(row["seed"].strip() for row in rows))

    def test_text_heavy_prompts_include_explicit_quoted_text(self) -> None:
        for manifest_name in MANIFEST_SPECS:
            with self.subTest(manifest=manifest_name):
                _, rows = load_rows(manifest_name)
                text_rows = [row for row in rows if row["category"] == "text-heavy"]
                self.assertTrue(text_rows)
                self.assertTrue(all(QUOTE_RE.search(row["prompt"]) for row in text_rows))


if __name__ == "__main__":
    unittest.main()
