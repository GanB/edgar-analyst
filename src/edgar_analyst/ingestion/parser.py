"""Filing parser for SEC 10-K and 10-Q documents.

Strips presentation HTML, walks block-level elements, and detects
section starts ("Item 1A. Risk Factors", "Part I — Item 2.
Management's Discussion", etc.) to produce a list of structured
sections. The parser is deliberately pragmatic: a perfect SEC parser
is its own project, and for ingestion we only need section-aware
chunk boundaries.

If no sections can be detected, the entire document text is returned
as a single section with item_id="whole" so the pipeline still
makes progress.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning

# SEC 10-K/10-Q primary documents are XHTML; lxml's HTML mode parses them
# correctly but bs4 emits this warning. Silence it so ingestion logs stay clean.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Lines longer than this are unlikely to be a section header.
MAX_HEADER_LINE_CHARS = 200

# A "section" with less body than this is treated as a TOC entry and
# folded out. Real 10-K body sections run from a few thousand up to
# tens of thousands of characters.
MIN_SECTION_BODY_CHARS = 500

# Captures "Item 1", "Item 1A", optionally prefixed with a Part marker
# (10-Q uses "Part I — Item 1"). En-dash (–) and em-dash (—)
# are common SEC typography; matched explicitly here.
HEADER_RE = re.compile(
    r"^\s*"
    r"(?:Part\s+(?P<part>I{1,3}V?|IV|II)\s*[–—\-\.\:]?\s*)?"
    r"Item\s+(?P<item>\d{1,2}[A-Z]?)\b"
    r"\s*[\.\-–—:]?\s*"
    r"(?P<title>.*?)"
    r"\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Section:
    item_id: str
    title: str
    text: str


@dataclass(frozen=True)
class _HeaderMatch:
    item_id: str
    title: str
    line_start: int
    line_end: int


def _strip_presentation(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    # Inline style and class attributes carry no text and only add noise.
    for tag in soup.find_all(True):
        for attr in ("style", "class"):
            if attr in tag.attrs:
                del tag.attrs[attr]
    return soup


def _to_plain_text(soup: BeautifulSoup) -> str:
    raw = soup.get_text(separator="\n")
    # Collapse runs of whitespace within each line; keep line breaks.
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    # Drop empty lines but keep paragraph separation light.
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    return "\n".join(cleaned).strip()


def _normalize_title(raw_title: str) -> str:
    title = raw_title.strip()
    # Strip leading punctuation and trailing punctuation/whitespace runs.
    title = re.sub(r"^[\s\.\-–—:]+", "", title)
    title = re.sub(r"[\s]+", " ", title)
    return title.strip()


def _find_header_matches(text: str) -> list[_HeaderMatch]:
    matches: list[_HeaderMatch] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        line_no_break = line.rstrip("\n")
        if 0 < len(line_no_break) <= MAX_HEADER_LINE_CHARS:
            m = HEADER_RE.match(line_no_break)
            if m is not None:
                item_id = m.group("item").upper()
                title = _normalize_title(m.group("title") or "")
                matches.append(
                    _HeaderMatch(
                        item_id=item_id,
                        title=title,
                        line_start=pos,
                        line_end=pos + len(line_no_break),
                    )
                )
        pos += len(line)
    return matches


def parse_filing(html: str) -> list[Section]:
    """Parse filing HTML into a list of sections.

    Falls back to a single ``Section(item_id="whole", ...)`` when no
    section headers can be detected.
    """
    soup = _strip_presentation(html)
    text = _to_plain_text(soup)

    if not text:
        return []

    matches = _find_header_matches(text)
    if not matches:
        return [Section(item_id="whole", title="", text=text)]

    sections: list[Section] = []
    for i, header in enumerate(matches):
        end = matches[i + 1].line_start if i + 1 < len(matches) else len(text)
        body = text[header.line_end : end].strip()
        if len(body) < MIN_SECTION_BODY_CHARS:
            # Likely a TOC entry or a stub reference. Skip it; the next
            # header that actually has body text will absorb the right slice.
            continue
        title = header.title or _default_title_for(header.item_id)
        sections.append(Section(item_id=header.item_id, title=title, text=body))

    if not sections:
        return [Section(item_id="whole", title="", text=text)]
    return sections


_DEFAULT_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}


def _default_title_for(item_id: str) -> str:
    return _DEFAULT_TITLES.get(item_id.upper(), f"Item {item_id}")
