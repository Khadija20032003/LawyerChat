"""
LangChain chains setup for SEC Risk Factors RAG Chatbot
"""

import logging
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

logger = logging.getLogger(__name__)

# Global variable to store uploaded PDF vector store
uploaded_pdf_vectordb = None

# ============================================================
# RAG CHAIN SETUP
# ============================================================

def create_rag_chain(
    llm,
    vectordb,
    prompt_template: str,
    retrieval_k: int = 5
):
    """
    Create a simple RAG chain for question answering
    
    Args:
        llm: LangChain LLM instance
        vectordb: LangChain Chroma vector database
        prompt_template: Template string for prompts
        retrieval_k: Number of documents to retrieve
    
    Returns:
        dict: Configuration for RAG chain with retriever and prompt
    """
    logger.info("🔗 Creating RAG chain...")
    
    # Create prompt template
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": retrieval_k}
    )
    
    # Return a simple chain configuration
    chain_config = {
        "llm": llm,
        "retriever": retriever,
        "prompt": prompt
    }
    
    logger.info("✓ RAG chain created successfully")
    return chain_config


# ============================================================
# PDF VECTOR STORE
# ============================================================

def create_pdf_vectorstore(
    pdf_text: str,
    embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Optional[Chroma]:
    """
    Create a temporary vector store from PDF text
    
    Args:
        pdf_text: Extracted text from PDF
        embeddings: Embeddings model
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        Chroma: Temporary vector store, or None if failed
    """
    global uploaded_pdf_vectordb
    
    if not pdf_text or not pdf_text.strip():
        logger.warning("No text provided for PDF vector store")
        uploaded_pdf_vectordb = None
        return None
    
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(pdf_text)
        logger.info(f"Split PDF into {len(chunks)} chunks")
        
        # Create temporary vector store
        uploaded_pdf_vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name="uploaded_pdf_temp"
        )
        
        logger.info("✓ Created vector store from uploaded PDF")
        return uploaded_pdf_vectordb
        
    except Exception as e:
        logger.error(f"Error creating PDF vector store: {e}")
        uploaded_pdf_vectordb = None
        return None


def clear_pdf_vectorstore():
    """Clear the uploaded PDF vector store"""
    global uploaded_pdf_vectordb
    uploaded_pdf_vectordb = None
    logger.info("Cleared uploaded PDF vector store")


# ============================================================
# QUERY PROCESSING
# ============================================================

def process_query(chain_config: dict, query: str) -> str:
    """
    Process a query through the RAG chain (main database only)
    
    Args:
        chain_config: Chain configuration dict
        query: User's question
    
    Returns:
        str: Generated answer
    """
    if not query.strip():
        return "Please enter a question."
    
    try:
        logger.info(f"Query: {query}")
        
        # Get components from chain config
        llm = chain_config["llm"]
        retriever = chain_config["retriever"]
        prompt = chain_config["prompt"]
        
        # Retrieve relevant documents
        # docs = retriever.get_relevant_documents(query)
        docs = retriever.invoke(query)

        
        if not docs:
            return "I couldn't find relevant information. Please try rephrasing your question."
        
        # Format context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format the prompt
        formatted_prompt = prompt.format(context=context, question=query)
        
        # Generate answer using LLM
        try:
            # For HuggingFacePipeline
            result = llm.invoke(formatted_prompt)
            answer = result.strip()
                
        except Exception as e:
            logger.error(f"Error with pipeline: {e}")
            # Try alternative method
            answer = str(llm(formatted_prompt))
        
        # Clean up the answer
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return "I encountered an error processing your question. Please try again."


def process_query_with_pdf(chain_config: dict, main_vectordb, query: str) -> str:
    """
    Process a query searching both main database and uploaded PDF
    
    Args:
        chain_config: Chain configuration dict
        main_vectordb: Main vector database
        query: User's question
    
    Returns:
        str: Generated answer
    """
    global uploaded_pdf_vectordb
    
    if not query.strip():
        return "Please enter a question."
    
    # If no PDF uploaded, use standard query
    if uploaded_pdf_vectordb is None:
        return process_query(chain_config, query)
    
    try:
        logger.info(f"Query (with PDF): {query}")
        
        # Search both sources
        all_docs = []
        
        # Search main database
        try:
            main_results = main_vectordb.similarity_search(query, k=3)
            for doc in main_results:
                all_docs.append({
                    'content': doc.page_content,
                    'source': 'SEC Database'
                })
            logger.info(f"Found {len(main_results)} docs from SEC database")
        except Exception as e:
            logger.error(f"Error searching main database: {e}")
        
        # Search uploaded PDF
        try:
            pdf_results = uploaded_pdf_vectordb.similarity_search(query, k=3)
            for doc in pdf_results:
                all_docs.append({
                    'content': doc.page_content,
                    'source': 'Your PDF'
                })
            logger.info(f"Found {len(pdf_results)} docs from uploaded PDF")
        except Exception as e:
            logger.error(f"Error searching PDF: {e}")
        
        if not all_docs:
            return "I couldn't find relevant information. Please try rephrasing your question."
        
        # Format context
        context_parts = []
        for doc in all_docs[:6]:
            source = doc['source']
            content = doc['content'][:500]
            context_parts.append(f"[From {source}]\n{content}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Get prompt template from chain config
        prompt_template = chain_config["prompt"]
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=combined_context, question=query)
        
        # Get LLM and generate
        llm = chain_config["llm"]
        
        try:
            result = llm.invoke(formatted_prompt)
            answer = result.strip()

                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = str(llm(formatted_prompt))
        
        # Clean answer
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"Error processing query with PDF: {e}")
        return "I encountered an error. Please try again."