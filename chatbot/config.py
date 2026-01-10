"""
Configuration settings for SEC Risk Factors RAG Chatbot
"""

import os
# import torch

# ============================================================
# HUGGING FACE SETTINGS
# ============================================================

HF_DATASET_REPO = "FatimaZh/sec-10k-chroma"
HF_TOKEN = os.getenv("HF_TOKEN")

# ============================================================
# MODEL SETTINGS
# ============================================================

LLM_MODEL = "unsloth/Llama-3.2-1B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================
# PATHS
# ============================================================

LOCAL_DIR = "./data"
CHROMA_PATH = f"{LOCAL_DIR}/chroma_sec"
COLLECTION_NAME = "sec_10k_risk_factors"

# ============================================================
# GENERATION SETTINGS
# ============================================================

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
RETRIEVAL_K = 5  # Number of documents to retrieve

# ============================================================
# DEVICE SETTINGS
# ============================================================

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
USE_FP16 = DEVICE == "cuda"

# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about SEC 10-K filings. 

Use the following context from SEC filings to answer the question accurately and concisely. If you cannot find the answer in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
