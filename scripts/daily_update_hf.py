"""
Daily SEC Filings Ingestion & Hugging Face Sync
Fetches new 10-K filings and updates the Chroma vector database on HF
"""

import datetime as dt
import logging
import shutil
from pathlib import Path
from huggingface_hub import HfApi, login, snapshot_download
import os
import sys

# Add src/ directory to Python path if it exists
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if os.path.exists(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import from sec_risk
try:
    from sec_risk_factors import SecClient, init_chroma, load_seen_accessions, check_date_in_vectordb
    from sec_risk_factors.pipeline import ingest_provided_filings, ProvidedFiling, IngestConfig
except ModuleNotFoundError as e:
    print("=" * 60)
    print("ERROR: sec_risk package not found!")
    print("=" * 60)
    print(f"\nError: {e}")
    print("\nChecked paths:")
    print(f"  - {src_path}")
    print(f"  - Current sys.path: {sys.path[:3]}...")
    print("\nMake sure 'sec_risk' folder exists in:")
    print("  your-repo/src/sec_risk/")
    print("=" * 60)
    raise

# ============================================================
# CONFIGURATION
# ============================================================

# Hugging Face Settings
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this as environment variable or GitHub secret
HF_DATASET_REPO = "FatimaZh/sec-10k-chroma"

# SEC Settings
USER_AGENT = "fati.bnr15@gmail.com"

# Local paths
BASE = Path("./data")
PERSIST_DIR = str(BASE / "chroma_sec")
ARTIFACT_DIR = BASE / "retrieved"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.jsonl"

# Ingestion config
CONFIG = IngestConfig(chunk_size=1500, chunk_overlap=200, batch_size=10_000)

# Logging
LOG_FILE = "sec_daily_ingestion.log"


# ============================================================
# SETUP LOGGING
# ============================================================

def setup_logging():
    """Configure logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================
# HUGGING FACE FUNCTIONS
# ============================================================

def download_chroma_from_hf(hf_api, repo_id, local_dir):
    """Download existing Chroma database from Hugging Face"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📥 Downloading Chroma database from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            token=HF_TOKEN,
            allow_patterns=["chroma_sec/**", "retrieved/**"]
        )
        logger.info("✓ Download complete")
        return True
    except Exception as e:
        logger.warning(f"Could not download from HF (might be first run): {e}")
        return False


def upload_chroma_to_hf(hf_api, repo_id, local_dir):
    """Upload updated Chroma database to Hugging Face"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📤 Uploading Chroma database to {repo_id}...")
        
        # Upload the chroma directory
        hf_api.upload_folder(
            folder_path=str(Path(local_dir) / "chroma_sec"),
            path_in_repo="chroma_sec",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        
        # Upload the retrieved artifacts
        hf_api.upload_folder(
            folder_path=str(Path(local_dir) / "retrieved"),
            path_in_repo="retrieved",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        
        logger.info("✓ Upload complete")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to HF: {e}")
        return False


def create_readme(repo_id, total_vectors, last_update, processed_count):
    """Create/update README for the dataset"""
    readme_content = f"""---
license: mit
tags:
- sec-filings
- 10-k
- financial-data
- rag
- chroma
---

# SEC 10-K Risk Factors Vector Database

This dataset contains a Chroma vector database of SEC 10-K filing risk factors, automatically updated daily.

## 📊 Statistics

- **Total vectors**: {total_vectors:,}
- **Last updated**: {last_update}
- **Latest batch**: {processed_count} filings

## 🔄 Usage

### Loading in your Gradio app:

```python
from huggingface_hub import snapshot_download
import chromadb

# Download the database
snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="./data",
    allow_patterns=["chroma_sec/**"]
)

# Initialize Chroma
client = chromadb.PersistentClient(path="./data/chroma_sec")
collection = client.get_collection("sec_10k_risk_factors")

# Query
results = collection.query(
    query_texts=["What are the cybersecurity risks?"],
    n_results=5
)
```

## 🤖 Automated Updates

This dataset is automatically updated daily with new SEC 10-K filings via GitHub Actions.

## 📝 Data Source

Data sourced from the SEC EDGAR database.
"""
    
    return readme_content


# ============================================================
# MAIN INGESTION LOGIC
# ============================================================

def run_daily_ingestion(target_date=None):
    """
    Main function to fetch and ingest SEC filings
    
    Args:
        target_date: Specific date to process (defaults to yesterday)
    """
    logger = logging.getLogger(__name__)
    
    # Setup HF API
    if not HF_TOKEN:
        logger.error("❌ HF_TOKEN environment variable not set!")
        return False
    
    login(token=HF_TOKEN)
    hf_api = HfApi()
    
    # Download existing database from HF
    download_chroma_from_hf(hf_api, HF_DATASET_REPO, str(BASE))
    
    # Ensure directories exist
    BASE.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Initialize SEC client and Chroma
    sec = SecClient(user_agent=USER_AGENT)
    vectordb = init_chroma(PERSIST_DIR, collection_name="sec_10k_risk_factors")
    seen = load_seen_accessions(MANIFEST_PATH)
    
    # Determine target date
    if target_date is None:
        target_date = dt.date.today() - dt.timedelta(days=1)
    
    logger.info("="*60)
    logger.info("Starting SEC daily ingestion")
    logger.info("="*60)
    logger.info(f"Target date: {target_date}")
    
    # Fetch filings
    idx_url, rows = sec.fetch_daily_10k_rows(target_date)
    
    if not rows:
        logger.info("No 10-K filings found in the index.")
        
        has_data = check_date_in_vectordb(vectordb, target_date)
        
        if has_data:
            logger.info(f"✓ Filings from {target_date} already exist in the vector database.")
            logger.info("Reason: Already ingested previously")
            return False  # No update needed
        else:
            logger.info("ℹ No filings from this date in the vector database.")
            logger.info("Reason: Weekend, federal holiday, or genuinely no 10-K filings on this date")
            return False  # No update needed
    
    logger.info(f"Daily index: {idx_url}")
    logger.info(f"10-K/10-K/A rows: {len(rows)}")
    
    # Build ProvidedFiling list
    provided = []
    missed = []
    
    logger.info("Fetching submission details for each filing...")
    for idx, x in enumerate(rows):
        if idx % 10 == 0:
            logger.info(f"Processing {idx}/{len(rows)}...")
        
        cik = x["cik"]
        filing_date = x["date_filed"]
        form = x["form"]
        company = x["company"]
        
        try:
            subs = sec.fetch_submissions(cik)
            recent = subs.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accession = recent.get("accessionNumber", [])
            primary = recent.get("primaryDocument", [])
            filing_dates = recent.get("filingDate", [])
            
            found = False
            for i, f in enumerate(forms):
                if f == form and filing_dates[i] == filing_date:
                    acc_num = accession[i]
                    
                    if acc_num in seen:
                        logger.debug(f"Skipping {company} - already ingested")
                        missed.append((cik, company, form, filing_date, "already_seen"))
                        found = True
                        break
                    
                    provided.append(ProvidedFiling(
                        cik=cik,
                        accessionNumber=acc_num,
                        primaryDocument=primary[i],
                        filingDate=filing_date,
                        form=form,
                        company=company,
                    ))
                    found = True
                    break
            
            if not found:
                missed.append((cik, company, form, filing_date, "not_found_in_submissions_recent"))
        
        except Exception as e:
            logger.error(f"Error processing {company} (CIK: {cik}): {e}")
            missed.append((cik, company, form, filing_date, f"error:{e}"))
    
    logger.info(f"Prepared filings: {len(provided)}")
    logger.info(f"Missed/Skipped: {len(missed)}")
    
    if not provided:
        logger.info("All filings already ingested or no new filings to process.")
        return False  # No update needed
    
    # Ingest
    logger.info("Starting ingestion...")
    df, skips = ingest_provided_filings(
        sec=sec,
        vectordb=vectordb,
        filings=provided,
        seen_accessions=seen,
        artifact_dir=ARTIFACT_DIR,
        manifest_path=MANIFEST_PATH,
        cfg=CONFIG,
    )
    
    # Get stats
    total_vectors = vectordb._collection.count()
    processed_count = len(df)
    
    logger.info("="*60)
    logger.info("INGESTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Date processed: {target_date}")
    logger.info(f"Total filings found: {len(rows)}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped during ingestion: {len(skips)}")
    logger.info(f"Total vectors in DB: {total_vectors:,}")
    
    if processed_count > 0:
        logger.info("\nProcessed companies:")
        for company in df['company'].head(10):
            logger.info(f"  ✓ {company}")
    
    # Upload to HF
    if processed_count > 0:
        logger.info("\n" + "="*60)
        logger.info("Uploading to Hugging Face...")
        logger.info("="*60)
        
        # Create/update README
        readme_content = create_readme(
            HF_DATASET_REPO,
            total_vectors,
            dt.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            processed_count
        )
        
        readme_path = BASE / "README.md"
        readme_path.write_text(readme_content)
        
        # Upload README
        hf_api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        
        # Upload database
        success = upload_chroma_to_hf(hf_api, HF_DATASET_REPO, str(BASE))
        
        if success:
            logger.info("✓ Successfully uploaded to Hugging Face!")
            return True
        else:
            logger.error("❌ Failed to upload to Hugging Face")
            return False
    
    return False


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logger = setup_logging()
    
    try:
        success = run_daily_ingestion()
        
        if success:
            logger.info("\n🎉 Daily update completed successfully!")
        else:
            logger.info("\n💤 No updates needed today")
            
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        raise