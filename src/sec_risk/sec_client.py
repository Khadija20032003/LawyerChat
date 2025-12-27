from __future__ import annotations

import time
import datetime as dt
import requests
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def make_headers(host: str, user_agent: str) -> Dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


@dataclass
class FilingRef:
    form: str
    accessionNumber: str
    primaryDocument: str
    filingDate: str


class SecClient:
    """
    Small SEC HTTP client with rate limiting and helpers to locate/resolve filing documents.
    """

    def __init__(
        self,
        user_agent: str,
        rate_delay_sec: float = 0.15,
        timeout_sec: int = 60,
    ):
        if not user_agent or "@" not in user_agent:
            # SEC recommends a real contact in User-Agent
            raise ValueError("Provide a real SEC User-Agent contact like 'Name email@domain.com'")

        self.user_agent = user_agent
        self.rate_delay_sec = rate_delay_sec
        self.timeout_sec = timeout_sec

        self.sec_headers = make_headers("www.sec.gov", user_agent)
        self.data_headers = make_headers("data.sec.gov", user_agent)

    def _get(self, url: str, headers: Dict[str, str]) -> requests.Response:
        r = requests.get(url, headers=headers, timeout=self.timeout_sec)
        r.raise_for_status()
        time.sleep(self.rate_delay_sec)
        return r

    @staticmethod
    def cik10(cik: str) -> str:
        return str(cik).zfill(10)

    def get_all_ciks(self) -> List[str]:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = self._get(url, self.sec_headers).json()
        return [str(v["cik_str"]) for v in data.values()]

    def fetch_submissions(self, cik: str) -> Dict[str, Any]:
        url = f"https://data.sec.gov/submissions/CIK{self.cik10(cik)}.json"
        return self._get(url, self.data_headers).json()

    @staticmethod
    def build_archive_base(cik: str, accession_number: str) -> str:
        acc_no = accession_number.replace("-", "")
        cik_nolead = str(int(cik))
        return f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{acc_no}"

    def fetch_index_json(self, cik: str, accession_number: str) -> Dict[str, Any]:
        base = self.build_archive_base(cik, accession_number)
        return self._get(base + "/index.json", self.sec_headers).json()

    def pick_best_primary_doc(self, cik: str, accession_number: str, default_primary: str) -> str:
        """
        Prefer an HTML 10-K doc when default is XML/XBRL.
        """
        base = self.build_archive_base(cik, accession_number)
        try:
            data = self._get(base + "/index.json", self.sec_headers).json()
            html_candidates = []
            for f in data.get("directory", {}).get("item", []):
                name = (f.get("name") or "").lower()
                if name.endswith((".htm", ".html")) and ("10-k" in name or "10k" in name):
                    html_candidates.append(f["name"])
            if html_candidates:
                # short names tend to be the main doc
                return sorted(html_candidates, key=len)[0]
        except Exception:
            pass
        return default_primary

    def fetch_filing_html(self, cik: str, accession_number: str, primary_doc: str) -> Tuple[str, str]:
        """
        Returns: (html_text, url)
        """
        base = self.build_archive_base(cik, accession_number)
        url = f"{base}/{primary_doc}"
        return self._get(url, self.sec_headers).text, url

    @staticmethod
    def find_top_n_10k_or_10ka(submissions: Dict[str, Any], n: int = 3) -> List[FilingRef]:
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession = recent.get("accessionNumber", [])
        primary = recent.get("primaryDocument", [])
        filing_date = recent.get("filingDate", [])

        rows: List[FilingRef] = []
        for i, form in enumerate(forms):
            if form in ("10-K", "10-K/A"):
                rows.append(
                    FilingRef(
                        form=form,
                        accessionNumber=accession[i],
                        primaryDocument=primary[i],
                        filingDate=filing_date[i],
                    )
                )
        rows.sort(key=lambda r: r.filingDate, reverse=True)
        return rows[:n]

    @staticmethod
    def master_index_url(d: dt.date) -> str:
        """Generate URL for SEC daily master index"""
        qtr = (d.month - 1) // 3 + 1
        return f"https://www.sec.gov/Archives/edgar/daily-index/{d.year}/QTR{qtr}/master.{d.strftime('%Y%m%d')}.idx"

    def fetch_daily_10k_rows(self, d: dt.date) -> Tuple[Optional[str], List[Dict]]:
        """
        Fetch filings for a specific date from SEC daily index.
        
        Returns:
            Tuple of (url, rows) where url is the index URL and rows is list of filing dicts.
            Returns (None, []) if no data available.
        """
        from .artifacts import parse_master_idx
        
        url = self.master_index_url(d)
        
        try:
            logger.info(f"Fetching filings for: {d}")
            r = requests.get(url, headers=self.sec_headers, timeout=self.timeout_sec)
            r.raise_for_status()
            time.sleep(self.rate_delay_sec)
            
            all_rows = parse_master_idx(r.text)
            logger.info(f"Total filings in index: {len(all_rows)}")
            
            rows = [x for x in all_rows if x["form"] in ("10-K", "10-K/A", "10-Q", "10-Q/A")]

            
            if rows:
                logger.info(f"✓ Found {len(rows)} 10-K/10-K/A filings")
                return url, rows
            else:
                logger.info(f"No 10-K/10-K/A filings found for {d}")
                return None, []
                
        except requests.HTTPError as e:
            if e.response.status_code in (404, 403):
                logger.info(f"No index file available for {d}. This is likely due to a weekend or federal holiday.")
                return None, []
            else:
                logger.error(f"HTTP error: {e}")
                return None, []
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None, []