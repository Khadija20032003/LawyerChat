from __future__ import annotations

import re
from bs4 import BeautifulSoup

# Next section detector (Item 1B onward)
NEXT_ITEM_RE = re.compile(
    r"""(?imx)
    ^\s*
    (?:ITEM|Item)
    [\s\u00A0]*
    (1B|1C|2|3|4|5|6|7|7A|8|9|9A|9B|9C|10|11|12|13|14)
    [\s\u00A0]*[\.\:\-–—]?
    (?:\s*|\s*\n\s*)
    """
)

# Item 1A detector (robust)
ITEM_1A_RE = re.compile(
    r"""(?imx)
    ^\s*(?:ITEM|Item)[\s\u00A0]*1A(?:[\.\:\-–—]|\b)[\s\u00A0]*(?:Risk[\s\u00A0]+Factors\b)?
    """
)


def html_to_text(html: str) -> str:
    """
    Strips scripts/styles and returns a normalized plaintext.
    Tries XML parser first, then HTML.
    """
    try:
        soup = BeautifulSoup(html, "lxml-xml")
        if not soup.find(True):
            raise ValueError("not-xml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = text.replace("\u00A0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_risk_factors(text: str, min_chars: int = 800) -> str | None:
    """
    Extracts the best (longest) Item 1A Risk Factors section.
    Filters out Table of Contents-like matches and too-short sections.
    """
    t = text.replace("\u00A0", " ")
    starts = [m.start() for m in ITEM_1A_RE.finditer(t)]
    if not starts:
        return None

    sections: list[str] = []
    for start in starts:
        m_next = NEXT_ITEM_RE.search(t, pos=start + 1)
        end = m_next.start() if m_next else None
        sect = t[start:end].strip() if end else t[start:].strip()

        if len(sect) < min_chars:
            continue
        if re.search(r"\bTable of Contents\b", sect, flags=re.IGNORECASE):
            continue

        # remove stray page number lines
        sect = "\n".join(
            ln for ln in sect.splitlines()
            if not re.fullmatch(r"\d{1,3}", ln.strip())
        ).strip()

        sections.append(sect)

    return max(sections, key=len) if sections else None
