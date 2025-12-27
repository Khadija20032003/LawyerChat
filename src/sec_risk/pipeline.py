from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .artifacts import append_manifest, write_artifact_txt
from .extractors import extract_risk_factors, html_to_text
from .sec_client import FilingRef, SecClient
from .store import make_chunk_ids, upsert_batched


@dataclass
class IngestConfig:
    top_n_per_cik: int = 3
    chunk_size: int = 1500
    chunk_overlap: int = 200
    min_item1a_chars: int = 800
    batch_size: int = 10_000


def ingest_one_filing(
    *,
    sec: SecClient,
    vectordb,
    cik: str,
    company_name: str,
    filing: FilingRef,
    seen_accessions: set[str],
    artifact_dir: Path,
    manifest_path: Path,
    cfg: IngestConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Ingests one filing and returns (record, skip_reason).
    Exactly one of record/skip_reason is non-None.
    """
    acc = filing.accessionNumber
    if acc in seen_accessions:
        return None, f"already_ingested:{acc}"

    primary = sec.pick_best_primary_doc(cik, acc, filing.primaryDocument)
    html, url = sec.fetch_filing_html(cik, acc, primary)
    text = html_to_text(html)
    risk = extract_risk_factors(text, min_chars=cfg.min_item1a_chars)

    if not risk:
        return None, f"{acc}:no_item_1a"

    splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    base_doc = Document(
        page_content=risk,
        metadata={
            "cik": cik,
            "company": company_name,
            "form": filing.form,
            "filingDate": filing.filingDate,
            "accessionNumber": acc,
            "url": url,
            "primary": primary,
            "section": "Item 1A",
            "fiscalYear": int(filing.filingDate[:4]) if filing.filingDate else None,
        },
    )

    docs = splitter.split_documents([base_doc])
    ids = make_chunk_ids(cik, acc, len(docs))

    # Use batched upsert to be safe on size limits
    upsert_batched(vectordb, docs, ids, batch_size=cfg.batch_size)

    # Artifacts + manifest
    write_artifact_txt(
        artifact_dir=artifact_dir,
        company_name=company_name,
        cik=cik,
        form=filing.form,
        filing_date=filing.filingDate,
        accession=acc,
        url=url,
        risk_text=risk,
    )
    append_manifest(
        manifest_path,
        {
            "cik": cik,
            "company": company_name,
            "form": filing.form,
            "filingDate": filing.filingDate,
            "accessionNumber": acc,
            "url": url,
            "primary": primary,
            "section": "Item 1A",
            "chunks": len(docs),
            "embedding_model": getattr(getattr(vectordb, "embedding_function", None), "model_name", None),
        },
    )

    seen_accessions.add(acc)

    record = {
        "company": company_name,
        "cik": cik,
        "filingDate": filing.filingDate,
        "form": filing.form,
        "accessionNumber": acc,
        "url": url,
        "chunks": len(docs),
    }
    return record, None


def ingest_cik(
    *,
    sec: SecClient,
    vectordb,
    cik: str,
    seen_accessions: set[str],
    artifact_dir: Path,
    manifest_path: Path,
    cfg: IngestConfig,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Returns (records, skips) where skips are (cik, reason).
    """
    subs = sec.fetch_submissions(cik)
    filings = sec.find_top_n_10k_or_10ka(subs, n=cfg.top_n_per_cik)
    if not filings:
        return [], [(cik, "no_10k_recent")]

    company_name = (subs.get("name") or f"CIK {cik}").strip()

    records: List[Dict[str, Any]] = []
    skips: List[Tuple[str, str]] = []

    for filing in filings:
        rec, reason = ingest_one_filing(
            sec=sec,
            vectordb=vectordb,
            cik=cik,
            company_name=company_name,
            filing=filing,
            seen_accessions=seen_accessions,
            artifact_dir=artifact_dir,
            manifest_path=manifest_path,
            cfg=cfg,
        )
        if reason:
            skips.append((cik, reason))
        else:
            records.append(rec)  # type: ignore[arg-type]

    return records, skips


def ingest_many_ciks(
    *,
    sec: SecClient,
    vectordb,
    ciks: Iterable[str],
    seen_accessions: set[str],
    artifact_dir: Path,
    manifest_path: Path,
    cfg: Optional[IngestConfig] = None,
    limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Main entry point for the "historical" style pipeline:
    takes CIKs, pulls submissions, selects top N 10-K/10-KA, ingests.
    """
    cfg = cfg or IngestConfig()

    all_records: List[Dict[str, Any]] = []
    all_skips: List[Tuple[str, str]] = []

    for idx, cik in enumerate(ciks):
        if limit is not None and idx >= limit:
            break
        try:
            recs, skips = ingest_cik(
                sec=sec,
                vectordb=vectordb,
                cik=cik,
                seen_accessions=seen_accessions,
                artifact_dir=artifact_dir,
                manifest_path=manifest_path,
                cfg=cfg,
            )
            all_records.extend(recs)
            all_skips.extend(skips)
        except Exception as e:
            all_skips.append((str(cik), f"error:{e}"))

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(
        columns=["company", "cik", "filingDate", "form", "accessionNumber", "url", "chunks"]
    )
    return df, all_skips


# ---- OPTIONAL: if your "recent pipeline" already has accessions ----
@dataclass
class ProvidedFiling:
    cik: str
    accessionNumber: str
    primaryDocument: str
    filingDate: str
    form: str = "10-K"
    company: Optional[str] = None


def ingest_provided_filings(
    *,
    sec: SecClient,
    vectordb,
    filings: Sequence[ProvidedFiling],
    seen_accessions: set[str],
    artifact_dir: Path,
    manifest_path: Path,
    cfg: Optional[IngestConfig] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Entry point for a "recent fetched data" pipeline if you already have a list of filings.
    """
    cfg = cfg or IngestConfig()

    all_records: List[Dict[str, Any]] = []
    all_skips: List[Tuple[str, str]] = []

    for f in filings:
        try:
            company = f.company
            if not company:
                # lightweight: only fetch submissions if company not provided
                subs = sec.fetch_submissions(f.cik)
                company = (subs.get("name") or f"CIK {f.cik}").strip()

            filing_ref = FilingRef(
                form=f.form,
                accessionNumber=f.accessionNumber,
                primaryDocument=f.primaryDocument,
                filingDate=f.filingDate,
            )
            rec, reason = ingest_one_filing(
                sec=sec,
                vectordb=vectordb,
                cik=f.cik,
                company_name=company,
                filing=filing_ref,
                seen_accessions=seen_accessions,
                artifact_dir=artifact_dir,
                manifest_path=manifest_path,
                cfg=cfg,
            )
            if reason:
                all_skips.append((f.cik, reason))
            else:
                all_records.append(rec)  # type: ignore[arg-type]
        except Exception as e:
            all_skips.append((f.cik, f"error:{e}"))

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame(
        columns=["company", "cik", "filingDate", "form", "accessionNumber", "url", "chunks"]
    )
    return df, all_skips
