"""
Model loading functions for SEC Risk Factors RAG Chatbot
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# ============================================================
# EMBEDDINGS
# ============================================================

def load_embeddings(model_name: str, device: str) -> HuggingFaceEmbeddings:
    """
    Load embeddings model
    
    Args:
        model_name: Name of the embeddings model
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        HuggingFaceEmbeddings: Loaded embeddings model
    """
    logger.info(f"📥 Loading embeddings model: {model_name}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}
    )
    
    logger.info("✓ Embeddings loaded successfully")
    return embeddings


# ============================================================
# VECTOR DATABASE
# ============================================================

def load_vector_database(
    chroma_path: str,
    collection_name: str,
    embeddings: HuggingFaceEmbeddings
) -> Chroma:
    """
    Load Chroma vector database
    
    Args:
        chroma_path: Path to Chroma database
        collection_name: Name of the collection
        embeddings: Embeddings model
    
    Returns:
        Chroma: Loaded vector database
    """
    logger.info(f"📥 Loading vector database from {chroma_path}")
    
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chroma_path
    )
    
    count = vectordb._collection.count()
    logger.info(f"✓ Loaded collection with {count:,} vectors")
    
    return vectordb


# ============================================================
# LLM MODEL
# ============================================================

def load_llm(
    model_name: str,
    device: str,
    use_fp16: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> HuggingFacePipeline:
    """
    Load LLM model and create LangChain pipeline
    
    Args:
        model_name: Name of the LLM model
        device: Device to use ('cuda' or 'cpu')
        use_fp16: Whether to use FP16 precision
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
    
    Returns:
        HuggingFacePipeline: LangChain-wrapped LLM pipeline
    """
    logger.info(f"📥 Loading LLM model: {model_name}")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    # Create pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        device=0 if device == "cuda" else -1,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    
    # Wrap in LangChain
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    logger.info("✓ LLM loaded successfully")
    return llm