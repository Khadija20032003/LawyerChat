"""
Helper functions for SEC Risk Factors RAG Chatbot
"""

import logging
from huggingface_hub import snapshot_download, login
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import os

logger = logging.getLogger(__name__)

# ============================================================
# AUTHENTICATION
# ============================================================

def setup_hf_authentication(token: str = None) -> None:
    """
    Setup HuggingFace authentication
    
    Args:
        token: HuggingFace token (optional)
    """
    if token:
        logger.info("🔐 Using HF_TOKEN for authentication")
        login(token=token)
    else:
        logger.info("ℹ️ No HF_TOKEN found - dataset must be public")


# ============================================================
# DATABASE DOWNLOAD
# ============================================================

def download_chroma_database(
    repo_id: str,
    local_dir: str,
    token: str = None
) -> bool:
    """
    Download Chroma database from HuggingFace
    
    Args:
        repo_id: HuggingFace dataset repository ID
        local_dir: Local directory to save the database
        token: HuggingFace token (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"📥 Downloading Chroma database from {repo_id}...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=["chroma_sec/**"],
            token=token
        )
        logger.info("✓ Database downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download database: {e}")
        logger.error("\n" + "="*60)
        logger.error("SOLUTION:")
        logger.error("1. Make your dataset public at:")
        logger.error(f"   https://huggingface.co/datasets/{repo_id}/settings")
        logger.error("   → Click 'Make public'")
        logger.error("\nOR")
        logger.error("\n2. Add HF_TOKEN to Space secrets:")
        logger.error("   Space Settings → Variables and secrets → New secret")
        logger.error("   Name: HF_TOKEN")
        logger.error("   Value: your_hf_token")
        logger.error("="*60)
        raise


# ============================================================
# PDF PROCESSING
# ============================================================

# def extract_text_from_pdf(pdf_path: str) -> str:
#     """
#     Extract text from a PDF file
    
#     Args:
#         pdf_path: Path to the PDF file
    
#     Returns:
#         str: Extracted text from the PDF
#     """
#     try:
#         import PyPDF2
        
#         with open(pdf_path, 'rb') as file:
#             pdf_reader = PyPDF2.PdfReader(file)
#             text = ""
            
#             for page in pdf_reader.pages:
#                 text += page.extract_text() + "\n\n"
        
#         logger.info(f"✓ Extracted {len(text)} characters from PDF")
#         return text.strip()
        
#     except Exception as e:
#         logger.error(f"Error extracting text from PDF: {e}")
#         return ""

import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text.strip()


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Save uploaded file to temporary location
    
    Args:
        uploaded_file: Gradio uploaded file object
    
    Returns:
        str: Path to saved file, or None if failed
    """
    if uploaded_file is None:
        return None
    
    try:
        # uploaded_file is already a file path from Gradio
        if isinstance(uploaded_file, str):
            logger.info(f"✓ Using uploaded file at {uploaded_file}")
            return uploaded_file
        else:
            # Fallback for file objects
            temp_dir = tempfile.gettempdir()
            file_path = os.path.join(temp_dir, "uploaded_document.pdf")
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            logger.info(f"✓ Saved uploaded file to {file_path}")
            return file_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return None
