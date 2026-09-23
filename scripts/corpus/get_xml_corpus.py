from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Literal

SplitMode = Literal["document", "section", "paragraph"]


def extract_publication_year(root: ET.Element) -> int | None:
    """
    Try to extract publication year from Elsevier XML.
    Returns year as int or None if year was not found.
    """

    year_candidate_tags = {
        "cover-date-year",
        "year-nav",
        "publication-year",
        "copyright-year",
    }

    date_candidate_tags = {
        "coverDate",
        "coverDisplayDate",
        "date-search-begin",
        "date-search-end",
        "cover-date-start",
        "cover-date-end",
    }

    # 1. Tags that usually contain only year, e.g. 2018
    for el in root.iter():
        name = local_name(el.tag)
        text = (el.text or "").strip()

        if name in year_candidate_tags and text:
            m = re.search(r"\b(19|20)\d{2}\b", text)
            if m:
                return int(m.group())

    # 2. Tags that contain dates, e.g. 2018-01-31 or January 2018
    for el in root.iter():
        name = local_name(el.tag)
        text = (el.text or "").strip()

        if name in date_candidate_tags and text:
            m = re.search(r"\b(19|20)\d{2}\b", text)
            if m:
                return int(m.group())

    # 3. Attributes, e.g. yyyymmdd="20170627"
    for el in root.iter():
        for value in el.attrib.values():
            m = re.search(r"\b(19|20)\d{2}\b", value)
            if m:
                return int(m.group())

    return None
    
    

def local_name(tag: str) -> str:
    """Remove XML namespace from tag name."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def children(el: ET.Element | None, name: str) -> list[ET.Element]:
    if el is None:
        return []
    return [c for c in list(el) if local_name(c.tag) == name]


def first_descendant(el: ET.Element | None, name: str) -> ET.Element | None:
    if el is None:
        return None
    for x in el.iter():
        if local_name(x.tag) == name:
            return x
    return None


def clean_text(text: str) -> str:
    """Light cleaning for raw corpus strings."""
    text = re.sub(r"https?://\S+|doi:\S+", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def text_without_tags(
    el: ET.Element | None,
    skip_tags: set[str] | None = None,
) -> str:
    """
    Extract text from an XML element, skipping citation/reference-like tags.
    """
    if el is None:
        return ""

    if skip_tags is None:
        skip_tags = {
            "label",        # section/figure/reference labels
            "cross-ref",    # citation markers like [1]
            "float-anchor", # figure/table anchors
            "link",         # DOI/URL links
        }

    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if local_name(node.tag) not in skip_tags:
            if node.text:
                parts.append(node.text)

            for child in list(node):
                walk(child)
                if child.tail:
                    parts.append(child.tail)

    walk(el)
    return clean_text(" ".join(parts))


def is_unwanted_abstract(abstract_el: ET.Element) -> bool:
    titles = [
        text_without_tags(t).lower()
        for t in abstract_el.iter()
        if local_name(t.tag) == "section-title"
    ]
    return any(t in {"graphical abstract", "highlights"} for t in titles)


def extract_section_texts(
    section_el: ET.Element,
    include_section_titles: bool = True,
) -> list[str]:
    """
    Extract one text chunk for this section, then recursively for subsections.
    Avoids duplicating subsection text in parent sections.
    """
    result: list[str] = []

    title_parts = []
    if include_section_titles:
        for title_el in children(section_el, "section-title"):
            title = text_without_tags(title_el)
            if title:
                title_parts.append(title)

    own_paragraphs = [
        text_without_tags(p)
        for p in children(section_el, "para")
    ]
    own_paragraphs = [p for p in own_paragraphs if p]

    section_text = clean_text(" ".join(title_parts + own_paragraphs))
    if section_text:
        result.append(section_text)

    for sub in children(section_el, "section"):
        result.extend(
            extract_section_texts(
                sub,
                include_section_titles=include_section_titles,
            )
        )

    return result


def extract_elsevier_xml_texts(
    xml_path: str | Path,
    split: SplitMode = "section",
    include_title: bool = True,
    include_abstract: bool = True,
    include_keywords: bool = True,
    include_section_titles: bool = True,
    min_chars: int = 30,
) -> list[str]:
    """
    Extract raw corpus strings from Elsevier / ScienceDirect full-text XML.

    split:
      - "document": one long string per XML file
      - "section": title, abstract, keywords, and each body section separately
      - "paragraph": title, abstract paragraphs, keywords, and body paragraphs
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    article = first_descendant(root, "article")

    if article is None:
        # Fallback for metadata-only XML.
        texts = []
        for el in root.iter():
            if local_name(el.tag) in {"title", "description", "subject"}:
                txt = text_without_tags(el)
                if len(txt) >= min_chars:
                    texts.append(txt)
        return texts

    head = next(iter(children(article, "head")), None)
    body = next(iter(children(article, "body")), None)

    pieces: list[str] = []

    if include_title and head is not None:
        for title_el in children(head, "title"):
            title = text_without_tags(title_el)
            if title:
                pieces.append(title)

    if include_abstract and head is not None:
        for abs_el in children(head, "abstract"):
            if is_unwanted_abstract(abs_el):
                continue

            paras = [
                text_without_tags(p)
                for p in abs_el.iter()
                if local_name(p.tag) == "simple-para"
            ]
            paras = [p for p in paras if p]

            if split == "paragraph":
                pieces.extend(paras)
            else:
                abstract_text = clean_text(" ".join(paras))
                if abstract_text:
                    pieces.append(abstract_text)

    if include_keywords and head is not None:
        keywords = [
            text_without_tags(k)
            for k in head.iter()
            if local_name(k.tag) == "keyword"
        ]
        keywords = [k for k in keywords if k]

        if keywords:
            pieces.append(" ".join(keywords))

    # Body only. Do not include acknowledgements, appendices, references, objects.
    sections_root = (
        next(iter(children(body, "sections")), None)
        if body is not None
        else None
    )

    if sections_root is not None:
        if split == "paragraph":
            for para_el in sections_root.iter():
                if local_name(para_el.tag) == "para":
                    para = text_without_tags(para_el)
                    if para:
                        pieces.append(para)

        else:
            body_sections = []
            for section_el in children(sections_root, "section"):
                body_sections.extend(
                    extract_section_texts(
                        section_el,
                        include_section_titles=include_section_titles,
                    )
                )

            if split == "section":
                pieces.extend(body_sections)

            elif split == "document":
                body_text = clean_text(" ".join(body_sections))
                if body_text:
                    pieces.append(body_text)

            else:
                raise ValueError(f"Unknown split mode: {split}")

    pieces = [clean_text(p) for p in pieces]
    pieces = [p for p in pieces if len(p) >= min_chars]

    if split == "document":
        doc = clean_text(" ".join(pieces))
        return [doc] if len(doc) >= min_chars else []

    return pieces


def build_xml_corpus(
    xml_paths,
    output_json_path=None,
    split="section",
    recursive=True,
    min_year: int = 1991,      # strictly after 1990
    skip_unknown_year: bool = True,
) -> list[str]:

    if isinstance(xml_paths, (str, Path)):
        xml_paths = Path(xml_paths)

        if xml_paths.is_dir():
            pattern = "**/*.xml" if recursive else "*.xml"
            paths = sorted(xml_paths.glob(pattern))
        else:
            paths = [xml_paths]
    else:
        paths = [Path(p) for p in xml_paths]

    corpus = []
    kept_files = 0
    skipped_old = 0
    skipped_unknown = 0

    for xml_path in paths:
        try:
            root = ET.parse(xml_path).getroot()
            year = extract_publication_year(root)

            if year is None:
                if skip_unknown_year:
                    skipped_unknown += 1
                    continue
            elif year < min_year:
                skipped_old += 1
                continue

            texts = extract_elsevier_xml_texts(
                xml_path,
                split=split,
            )

            corpus.extend(texts)
            kept_files += 1

        except Exception as e:
            print(f"[WARN] Failed to process {xml_path}: {e}")

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with output_json_path.open("w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)

        print(f"Saved XML corpus: {len(corpus)} texts -> {output_json_path}")

    print(
        f"XML files kept: {kept_files}, "
        f"skipped old: {skipped_old}, "
        f"skipped unknown year: {skipped_unknown}"
    )

    return corpus
    
    
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an auditable raw corpus from Elsevier XML files."
    )
    parser.add_argument(
        "--xml-inputs", nargs="+", required=True,
        help="XML files and/or directories. Directories are searched recursively by default.",
    )
    parser.add_argument("--output-json", required=True, help="Output JSON corpus path.")
    parser.add_argument(
        "--split", choices=["document", "section", "paragraph"], default="section"
    )
    parser.add_argument("--min-year", type=int, default=1991)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument(
        "--keep-unknown-year", action="store_true",
        help="Keep XML records whose publication year cannot be extracted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for item in args.xml_inputs:
        p = Path(item)
        if p.is_dir():
            pattern = "*.xml" if args.no_recursive else "**/*.xml"
            paths.extend(sorted(p.glob(pattern)))
        else:
            paths.append(p)

    build_xml_corpus(
        xml_paths=paths,
        output_json_path=args.output_json,
        split=args.split,
        recursive=not args.no_recursive,
        min_year=args.min_year,
        skip_unknown_year=not args.keep_unknown_year,
    )


if __name__ == "__main__":
    main()
