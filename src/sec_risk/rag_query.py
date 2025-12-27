"""
RAG Query functions - extracted from notebook
"""
from pathlib import Path
import requests
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ask_ollama(prompt: str, model: str = "llama3.1"):
    """
    Ask Ollama (local LLM) with error handling
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.7
                }
            },
            timeout=120
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        if 'response' in response_data:
            return response_data['response']
        else:
            print(f"❌ Unexpected response format: {response_data}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama.")
        print("Make sure Ollama is running: 'ollama serve'")
        return None
    except requests.exceptions.Timeout:
        print("❌ Request timed out.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def ask_with_rag(vectordb, question: str, n_context: int = 5):
    """
    Ask a question using RAG with Ollama
    
    This function is extracted from notebooks/03_rag_with_ollama.ipynb
    """
    print(f"🔍 Searching for relevant context...")
    
    # 1. Retrieve context
    results = vectordb.similarity_search_with_score(question, k=n_context)
    
    if not results:
        print("❌ No relevant documents found")
        return None
    
    # 2. Build context
    context_parts = []
    sources = []
    
    for i, (doc, score) in enumerate(results, 1):
        company = doc.metadata['company']
        date = doc.metadata['filingDate']
        content = doc.page_content[:500]
        
        context_parts.append(f"[Source {i}] {company} ({date})\n{content}")
        sources.append({
            "company": company,
            "date": date,
            "score": score,
            "url": doc.metadata.get('url', 'N/A'),
            "content": doc.page_content
        })
    
    context = "\n\n".join(context_parts)
    print(f"✓ Retrieved {len(results)} sources")
    
    # 3. Ask Ollama
    prompt = f"""You are a legal and financial analyst specializing in SEC filings.

Based on these risk factor disclosures from SEC 10-K filings, answer the question.

Context from SEC 10-K Risk Factors:
{context}

Question: {question}

Provide a comprehensive answer that:
1. Synthesizes information from the sources
2. Cites specific companies
3. Highlights key risks
4. Remains objective and evidence-based

Answer:"""

    print(f"💭 Asking Ollama (this may take 20-30 seconds)...")
    answer = ask_ollama(prompt, model="llama3.1")
    
    if not answer:
        return {
            "question": question,
            "answer": "⚠️ Ollama did not respond. Check the connection.",
            "sources": sources
        }
    
    print(f"✓ Answer generated!")
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "context_length": len(context)
    }