"""Tests for scientific PDF text cleanup."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'corpus'))

from text_normalization import cleanup_extracted_text, fix_pdf_hyphenation, normalize_scientific_unicode


def test_unicode_dash_and_subscript_normalization():
    assert normalize_scientific_unicode("H₂O") == "H2O"
    assert normalize_scientific_unicode("Ca(OH)₂") == "Ca(OH)2"
    assert normalize_scientific_unicode("3Y‑TZP") == "3Y-TZP"
    assert normalize_scientific_unicode("4Y–PSZ") == "4Y-PSZ"
    assert normalize_scientific_unicode("5Y—PSZ") == "5Y-PSZ"


def test_conservative_hyphenation():
    assert fix_pdf_hyphenation("infor-\nmation") == "information"
    assert fix_pdf_hyphenation("3Y-\nTZP") == "3Y-TZP"
    assert fix_pdf_hyphenation("4Y-\nPSZ") == "4Y-PSZ"
    assert fix_pdf_hyphenation("5Y-\nPSZ") == "5Y-PSZ"
    assert fix_pdf_hyphenation("short-\nfiber") == "short-fiber"


def test_inline_citation_removal_without_breaking_formulas():
    text = (
        "apicoectomy70,71), pulp capping72), radiopacity73,74), "
        "biocompatibility75-78). H₂O and Ca(OH)₂ were used with 3Y‑TZP."
    )
    cleaned = cleanup_extracted_text(text)
    assert "apicoectomy70" not in cleaned
    assert "70,71" not in cleaned
    assert "pulp capping72" not in cleaned
    assert "biocompatibility75" not in cleaned
    assert "H2O" in cleaned
    assert "Ca(OH)2" in cleaned
    assert "3Y-TZP" in cleaned


def test_cleanup_expected_examples():
    cases = {
        "3Y‑TZP": "3Y-TZP",
        "4Y–PSZ": "4Y-PSZ",
        "5Y—PSZ": "5Y-PSZ",
        "H₂O": "H2O",
        "Ca(OH)₂": "Ca(OH)2",
        "infor-\nmation": "information",
        "3Y-\nTZP": "3Y-TZP",
    }
    for raw, expected in cases.items():
        assert cleanup_extracted_text(raw) == expected

if __name__ == "__main__":
    test_unicode_dash_and_subscript_normalization()
    test_conservative_hyphenation()
    test_inline_citation_removal_without_breaking_formulas()
    test_cleanup_expected_examples()
    print("All text normalization tests passed.")
