import csv
import json
from pathlib import Path


EXAMPLES = Path(__file__).parents[1] / "examples"


def test_expected_example_files_exist():
    expected = {
        "example_config.json",
        "example_manifest.jsonl",
        "example_preprocessed_tokens.txt",
        "example_phrased_tokens.txt",
        "example_vocabulary_counts.tsv",
        "example_neighbors.tsv",
        "example_displacement_summary.tsv",
    }
    assert expected.issubset({path.name for path in EXAMPLES.iterdir()})


def test_example_manifest_has_both_periods_and_source_types():
    rows = [
        json.loads(line)
        for line in (EXAMPLES / "example_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 12
    assert {row["period"] for row in rows} == {"pre2018", "post2018"}
    assert {row["source_type"] for row in rows} == {"pdf", "xml"}


def test_example_phraser_and_alignment_outputs_are_nonempty():
    phrased = (EXAMPLES / "example_phrased_tokens.txt").read_text(encoding="utf-8")
    assert "fracture_toughness" in phrased
    assert "resin_composite" in phrased

    with (EXAMPLES / "example_displacement_summary.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    all_row = next(row for row in rows if row["scope"] == "all")
    assert int(all_row["n_terms"]) == 4
