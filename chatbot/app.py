"""
SEC Risk Factors RAG Chatbot - Fixed Version
Compatible with LangChain 0.3+
"""
import os  

import gradio as gr
import logging
from typing import List, Tuple

# Import our modules
import config
from helpers import (
    setup_hf_authentication,
    download_chroma_database,
    extract_text_from_pdf,
    save_uploaded_file
)
from models import (
    load_embeddings,
    load_vector_database,
    load_llm
)
from chains import (
    create_rag_chain,
    create_pdf_vectorstore,
    clear_pdf_vectorstore,
    process_query
)

# ============================================================
# SETUP LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL VARIABLES FOR STATE
# ============================================================

embeddings = None
vectordb = None
llm = None
qa_chain = None
pdf_vectordb = None

# ============================================================
# INITIALIZATION FUNCTIONS
# ============================================================
def initialize_system():
    global embeddings, vectordb, llm, qa_chain

    setup_hf_authentication(config.HF_TOKEN)

    # download only if needed
    import os
    if (not os.path.exists(config.CHROMA_PATH)) or (not os.listdir(config.CHROMA_PATH)):
        download_chroma_database(
            repo_id=config.HF_DATASET_REPO,
            local_dir=config.LOCAL_DIR,
            token=config.HF_TOKEN
        )

    embeddings = load_embeddings(config.EMBEDDING_MODEL, config.DEVICE)
    vectordb = load_vector_database(config.CHROMA_PATH, config.COLLECTION_NAME, embeddings)
    llm = load_llm(config.LLM_MODEL, config.DEVICE, config.USE_FP16,
                   config.MAX_NEW_TOKENS, config.TEMPERATURE, config.TOP_P)
    qa_chain = create_rag_chain(llm, vectordb, config.PROMPT_TEMPLATE, config.RETRIEVAL_K)


def handle_pdf_upload(pdf_file, progress=gr.Progress()):
    """Handle PDF upload and create vector store"""
    global pdf_vectordb, embeddings
    if embeddings is None:
        initialize_system()

    if pdf_file is None:
        clear_pdf_vectorstore()
        pdf_vectordb = None
        return "📄 No PDF uploaded. Using SEC database only."
    
    try:
        progress(0, desc="💾 Saving PDF...")
        pdf_path = save_uploaded_file(pdf_file)
        if not pdf_path:
            return "❌ Failed to save PDF file."
        
        progress(0.3, desc="📖 Extracting text...")
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return "❌ Could not extract text from PDF. Make sure it's a text-based PDF."
        
        progress(0.6, desc="🔨 Creating vector store...")
        pdf_vectordb = create_pdf_vectorstore(pdf_text, embeddings)
        if not pdf_vectordb:
            return "❌ Failed to process PDF."
        
        progress(1.0, desc="✅ Done!")
        return f"✅ PDF processed successfully!\n\n📊 {len(pdf_text):,} characters extracted\n\nYou can now ask questions about both your PDF and the SEC database."
        
    except Exception as e:
        logger.error(f"Error handling PDF: {e}")
        return f"❌ Error: {str(e)}"


# ============================================================
# CHAT FUNCTION
# ============================================================

def process_message(message, history):
    """
    Process user message and generate response
    
    Args:
        message: User's message
        history: Chat history (list of [user_msg, bot_msg] pairs)
    
    Returns:
        tuple: (updated_history, empty_string_for_textbox)
    """
    global qa_chain, vectordb, pdf_vectordb, llm
    
    if not message.strip():
        return history, ""
    
    if qa_chain is None:
        initialize_system()

    
    try:
        logger.info(f"Processing query: {message}")
        
        # Determine which databases to search
        if pdf_vectordb is None:
            # Search only SEC database using the process_query function
            answer = process_query(qa_chain, message)
        else:
            # Search both databases
            all_docs = []
            
            # Search SEC database
            try:
                main_results = vectordb.similarity_search(message, k=3)
                for doc in main_results:
                    all_docs.append({
                        'content': doc.page_content,
                        'source': '📊 SEC Database'
                    })
                logger.info(f"Found {len(main_results)} docs from SEC database")
            except Exception as e:
                logger.error(f"Error searching SEC database: {e}")
            
            # Search uploaded PDF
            try:
                pdf_results = pdf_vectordb.similarity_search(message, k=4)
                for doc in pdf_results:
                    all_docs.append({
                        'content': doc.page_content,
                        'source': '📄 Your PDF'
                    })
                logger.info(f"Found {len(pdf_results)} docs from PDF")
            except Exception as e:
                logger.error(f"Error searching PDF: {e}")
            
            if not all_docs:
                answer = "I couldn't find relevant information. Please try rephrasing your question."
            else:
                # Format context
                context_parts = []
                for doc in all_docs[:6]:
                    source = doc['source']
                    content = doc['content'][:500]
                    context_parts.append(f"[{source}]\n{content}")
                
                combined_context = "\n\n".join(context_parts)
                
                # Get prompt template from chain config
                prompt_template = qa_chain["prompt"]
                
                # Format the prompt
                formatted_prompt = prompt_template.format(context=combined_context, question=message)
                
                # Generate answer
                try:
                    result = llm.invoke(formatted_prompt)
                    answer = result.strip()

                except Exception as e:
                    logger.error(f"Error generating answer: {e}")
                    # Fallback
                    answer = str(llm(formatted_prompt))
        
        # Clean up the answer
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Add to history
        new_history = history + [[message, answer]]
        return new_history, ""

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Error processing message: {e}\n{tb}")
        bot_response = f"❌ Error:\n{e}\n\nTraceback:\n{tb}"
        new_history = history + [[message, bot_response]]
        return new_history, ""



# ============================================================
# GRADIO INTERFACE
# ============================================================

def create_demo():
    """Create the Gradio interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="SEC Risk Factors Assistant") as demo:
        
        gr.HTML("<center><h1>📊 SEC Risk Factors Assistant</h1></center>")
        gr.Markdown("""
        <center>Ask questions about SEC filings and your own documents</center>
        """)
        
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=1):
                gr.Markdown("### 🚀 System Setup")
                
                gr.Markdown("---")
                gr.Markdown("### 📄 PDF Upload (Optional)")
                
                pdf_upload = gr.File(
                    label="Upload Your PDF",
                    type="filepath",
                    file_types=[".pdf"]
                )
                pdf_status = gr.Textbox(
                    label="PDF Status",
                    value="📄 No PDF uploaded",
                    interactive=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### 💡 Example Questions")
                gr.Markdown("""
                - What are the main cybersecurity risks?
                - Which companies mention climate change?
                - What regulatory risks do financial institutions report?
                - What are supply chain risks?
                - Tell me about emerging technology risks
                """)
            
            # Right column - Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_label=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Ask a question about SEC filings or your PDF...",
                        show_label=False,
                        scale=9
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        
        pdf_upload.change(
            fn=handle_pdf_upload,
            inputs=[pdf_upload],
            outputs=[pdf_status]
        )
        
        msg.submit(
            fn=process_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )

        send_btn.click(
            fn=process_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )

        
        clear_btn.click(
            fn=lambda: ([], ""),
            inputs=[],
            outputs=[chatbot, msg]
        )
        demo.queue(status_update_rate="auto", default_concurrency_limit=1)


    
    return demo


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    # After logger setup and global vars:

    logger.info("🚀 Preloading system on startup...")
    try:
        initialize_system()  
        logger.info("✅ Preload complete")
    except Exception as e:
        logger.error(f"❌ Preload failed: {e}")

    logger.info("🚀 Starting SEC Risk Factors RAG Chatbot...")
    demo = create_demo()
    demo.launch(share=True)