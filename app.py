"""
Streamlit app for SEC Risk Factors Query System
"""
import streamlit as st
from pathlib import Path
from sec_risk import init_chroma
import requests
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
st.set_page_config(
    page_title="SEC Risk Factors AI",
    page_icon="⚖️",
    layout="wide"
)

# Initialize
@st.cache_resource
def load_vectordb():
    BASE = Path("./data")
    PERSIST_DIR = str(BASE / "chroma_sec")
    return init_chroma(PERSIST_DIR, collection_name="sec_10k_risk_factors")

vectordb = load_vectordb()

# Helper functions
def ask_ollama(prompt: str, model: str = "llama3.1"):
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 250, "temperature": 0.7}
            },
            timeout=90
        )
        return response.json().get('response')
    except:
        return None

def search_risks(question: str, n_results: int = 3):
    results = vectordb.similarity_search_with_score(question, k=n_results)
    
    context_parts = []
    sources = []
    
    for i, (doc, score) in enumerate(results, 1):
        company = doc.metadata['company']
        date = doc.metadata['filingDate']
        content = doc.page_content[:300]
        
        context_parts.append(f"[{i}] {company}: {content}")
        sources.append({
            "company": company,
            "date": date,
            "score": score,
            "content": doc.page_content
        })
    
    context = "\n".join(context_parts)
    
    prompt = f"""Based on these SEC risk factors:

{context}

Question: {question}

Answer in 2-3 concise sentences:"""

    answer = ask_ollama(prompt)
    
    return {"answer": answer, "sources": sources}

# UI
st.title("⚖️ SEC Risk Factors AI Assistant")
st.markdown("Query corporate risk factors from 10-K filings")

# Sidebar
with st.sidebar:
    st.header("📊 Database Info")
    total = vectordb._collection.count()
    st.metric("Total Documents", total)
    
    st.header("⚙️ Settings")
    n_sources = st.slider("Number of sources", 1, 5, 3)
    
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("- Ollama (llama3.1)")
    st.markdown("- ChromaDB")
    st.markdown("- Sentence Transformers")

# Main area
tab1, tab2 = st.tabs(["💬 Ask Question", "🔍 Browse"])

with tab1:
    st.header("Ask about risk factors")
    
    # Example questions
    st.markdown("**Try these examples:**")
    examples = [
        "What are the main cybersecurity risks?",
        "What supply chain issues do companies face?",
        "How do companies describe AI-related risks?",
        "What regulatory risks are mentioned?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"ex_{i}"):
            st.session_state.question = example
    
    # Question input
    question = st.text_area(
        "Your question:",
        value=st.session_state.get('question', ''),
        height=100,
        placeholder="Ask about risk factors from SEC filings..."
    )
    
    if st.button("🚀 Get Answer", type="primary"):
        if question:
            with st.spinner("🔍 Searching and analyzing..."):
                result = search_risks(question, n_results=n_sources)
            
            if result['answer']:
                st.success("✓ Answer generated!")
                
                st.markdown("### 🤖 Answer")
                st.write(result['answer'])
                
                st.markdown("### 📚 Sources")
                for i, source in enumerate(result['sources'], 1):
                    with st.expander(f"📄 {source['company']} ({source['date']}) - Score: {source['score']:.3f}"):
                        st.write(source['content'][:500] + "...")
            else:
                st.error("❌ Could not generate answer. Is Ollama running?")
        else:
            st.warning("Please enter a question")

with tab2:
    st.header("Browse database")
    
    search_term = st.text_input("Search term:", placeholder="cybersecurity, climate, supply chain...")
    
    if st.button("Search"):
        if search_term:
            results = vectordb.similarity_search_with_score(search_term, k=5)
            
            st.write(f"Found {len(results)} results")
            
            for i, (doc, score) in enumerate(results, 1):
                with st.expander(f"{i}. {doc.metadata['company']} (Score: {score:.3f})"):
                    st.write(f"**Filing Date:** {doc.metadata['filingDate']}")
                    st.write(f"**Form:** {doc.metadata['form']}")
                    st.write(f"**URL:** {doc.metadata.get('url', 'N/A')}")
                    st.markdown("---")
                    st.text_area("Content:", doc.page_content, height=200, key=f"content_{i}")