from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


TXT_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})__"
    r"(?P<cik>\d+)__"
    r"(?P<company>.+?)__"
    r"(?P<form>10-K(?:/A)?)__"
    r"(?P<acc>[^/\\]+)\.txt$"
)


def load_seen_accessions(manifest_path: Path) -> set[str]:
    seen: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    acc = j.get("accessionNumber")
                    if acc:
                        seen.add(acc)
                except Exception:
                    pass
    return seen


def safe_company_filename(company: str, limit: int = 80) -> str:
    return "".join(ch for ch in company if ch.isalnum() or ch in "._- ").strip()[:limit]


def write_artifact_txt(
    artifact_dir: Path,
    company_name: str,
    cik: str,
    form: str,
    filing_date: str,
    accession: str,
    url: str,
    risk_text: str,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_company = safe_company_filename(company_name)
    fname = f"{filing_date}__{cik}__{safe_company}__{form}__{accession}.txt"
    path = artifact_dir / fname
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{company_name} (CIK {cik}) | {form}\n")
        f.write(f"Filing Date: {filing_date} | Accession: {accession}\n")
        f.write(f"URL: {url}\n\n=== ITEM 1A. RISK FACTORS ===\n\n{risk_text}")
    return path


def append_manifest(manifest_path: Path, row: Dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_artifact_txt(path: Path) -> Optional[Tuple[Dict, str]]:
    """
    Reads your saved txt artifact and returns: (metadata, body)
    """
    m = TXT_RE.match(path.name)
    if not m:
        return None

    meta = m.groupdict()
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = text.split("=== ITEM 1A. RISK FACTORS ===", 1)
    body = parts[1].strip() if len(parts) > 1 else text

    metadata = {
        "cik": meta["cik"],
        "company": meta["company"],
        "form": meta["form"],
        "filingDate": meta["date"],
        "accessionNumber": meta["acc"],
        "section": "Item 1A",
        "fiscalYear": int(meta["date"][:4]),
    }
    return metadata, body


def parse_master_idx(text: str) -> list[dict]:
    """
    Parse SEC master index file for daily filings.
    
    Args:
        text: Raw text content of master index file
        
    Returns:
        List of dicts with keys: cik, company, form, date_filed, filename
    """
    rows = []
    started = False
    
    for line in text.splitlines():
        if not started:
            if line.strip().startswith("CIK|Company Name|Form Type|Date Filed|Filename"):
                started = True
            continue
            
        parts = line.split("|")
        if len(parts) != 5:
            continue
            
        cik, company, form, date_filed, filename = parts
        rows.append({
            "cik": cik.strip(),
            "company": company.strip(),
            "form": form.strip(),
            "date_filed": date_filed.strip(),
            "filename": filename.strip(),
        })
    
    return rows