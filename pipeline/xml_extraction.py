from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

SplitMode = Literal["document", "section", "paragraph"]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def children(el: ET.Element | None, name: str) -> list[ET.Element]:
    if el is None:
        return []
    return [child for child in list(el) if local_name(child.tag) == name]


def first_descendant(el: ET.Element | None, name: str) -> ET.Element | None:
    if el is None:
        return None
    return next((node for node in el.iter() if local_name(node.tag) == name), None)


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|doi:\S+", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def text_without_tags(
    el: ET.Element | None,
    skip_tags: set[str] | None = None,
) -> str:
    if el is None:
        return ""
    skip_tags = skip_tags or {"label", "cross-ref", "float-anchor", "link"}
    parts: list[str] = []

    def walk(node: ET.Element) -> None:
        if local_name(node.tag) in skip_tags:
            return
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(el)
    return clean_text(" ".join(parts))


def extract_publication_year(root: ET.Element) -> int | None:
    year_tags = {"cover-date-year", "year-nav", "publication-year", "copyright-year"}
    date_tags = {
        "coverDate",
        "coverDisplayDate",
        "date-search-begin",
        "date-search-end",
        "cover-date-start",
        "cover-date-end",
    }
    for candidates in (year_tags, date_tags):
        for el in root.iter():
            if local_name(el.tag) in candidates:
                match = re.search(r"\b(19|20)\d{2}\b", (el.text or "").strip())
                if match:
                    return int(match.group())
    for el in root.iter():
        for value in el.attrib.values():
            match = re.search(r"\b(19|20)\d{2}\b", value)
            if match:
                return int(match.group())
    return None


def _is_unwanted_abstract(abstract_el: ET.Element) -> bool:
    titles = {
        text_without_tags(node).lower()
        for node in abstract_el.iter()
        if local_name(node.tag) == "section-title"
    }
    return bool(titles & {"graphical abstract", "highlights"})


def _extract_section_texts(
    section_el: ET.Element,
    include_section_titles: bool,
) -> list[str]:
    title_parts = (
        [text_without_tags(el) for el in children(section_el, "section-title")]
        if include_section_titles
        else []
    )
    paragraphs = [text_without_tags(el) for el in children(section_el, "para")]
    own_text = clean_text(" ".join(x for x in title_parts + paragraphs if x))
    result = [own_text] if own_text else []
    for subsection in children(section_el, "section"):
        result.extend(_extract_section_texts(subsection, include_section_titles))
    return result


def extract_elsevier_xml_texts(
    path: str | Path,
    split: SplitMode = "section",
    include_title: bool = True,
    include_abstract: bool = True,
    include_keywords: bool = True,
    include_section_titles: bool = True,
    min_chars: int = 30,
) -> tuple[list[str], int | None]:
    root = ET.parse(path).getroot()
    publication_year = extract_publication_year(root)
    article = first_descendant(root, "article")

    if article is None:
        texts = [
            text_without_tags(el)
            for el in root.iter()
            if local_name(el.tag) in {"title", "description", "subject"}
        ]
        return [text for text in texts if len(text) >= min_chars], publication_year

    head = next(iter(children(article, "head")), None)
    body = next(iter(children(article, "body")), None)
    pieces: list[str] = []

    if include_title and head is not None:
        pieces.extend(text_without_tags(el) for el in children(head, "title"))

    if include_abstract and head is not None:
        for abstract in children(head, "abstract"):
            if _is_unwanted_abstract(abstract):
                continue
            paragraphs = [
                text_without_tags(el)
                for el in abstract.iter()
                if local_name(el.tag) == "simple-para"
            ]
            if split == "paragraph":
                pieces.extend(paragraphs)
            else:
                pieces.append(clean_text(" ".join(paragraphs)))

    if include_keywords and head is not None:
        keywords = [
            text_without_tags(el)
            for el in head.iter()
            if local_name(el.tag) == "keyword"
        ]
        if keywords:
            pieces.append(" ".join(keywords))

    sections_root = (
        next(iter(children(body, "sections")), None) if body is not None else None
    )
    if sections_root is not None:
        if split == "paragraph":
            pieces.extend(
                text_without_tags(el)
                for el in sections_root.iter()
                if local_name(el.tag) == "para"
            )
        else:
            body_sections: list[str] = []
            for section in children(sections_root, "section"):
                body_sections.extend(
                    _extract_section_texts(section, include_section_titles)
                )
            if split == "section":
                pieces.extend(body_sections)
            elif split == "document":
                pieces.append(clean_text(" ".join(body_sections)))
            else:
                raise ValueError(f"Unknown XML split mode: {split}")

    pieces = [clean_text(piece) for piece in pieces]
    pieces = [piece for piece in pieces if len(piece) >= min_chars]
    if split == "document":
        document = clean_text(" ".join(pieces))
        pieces = [document] if len(document) >= min_chars else []
    return pieces, publication_year
