import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from loader import load_transcript
from chunker import chunk_documents
from vector_store import create_vector_store
from retriever import get_retriever
from rag_chain import create_rag_chain

# Minimal, modern styling setup
st.set_page_config(page_title="YT Chat", page_icon="🎬", layout="centered")

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "current_url" not in st.session_state:
        st.session_state.current_url = ""

def process_video(url):
    if url == st.session_state.current_url and st.session_state.chain is not None:
        return True # Already processed

    with st.spinner("📥 Loading video transcript..."):
        docs = load_transcript(url)
        if not docs:
            st.error("Failed to load transcript. Please check if the URL is valid and the video has subtitles.")
            return False

    with st.spinner("⚙️ Chunking documents..."):
        chunks = chunk_documents(docs)

    with st.spinner("🗂️ Storing in Vector Database..."):
        vector_store = create_vector_store(chunks, force_reload=True)

    with st.spinner("🧠 Preparing Knowledge Base..."):
        retriever = get_retriever(vector_store, chunks)
        st.session_state.chain = create_rag_chain(retriever)
        st.session_state.current_url = url
        st.session_state.messages = [] # Reset chat for new video
        
    return True

def main():
    init_session_state()

    # Custom CSS for a cleaner, modern look
    st.markdown("""
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 800px;
            }
            .stButton>button {
                border-radius: 8px;
                font-weight: 500;
            }
            .stTextInput>div>div>input {
                border-radius: 8px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for controls
    with st.sidebar:
        st.title("🎬 YT Chat")
        st.markdown("Ask questions about any YouTube video with AI.")
        
        youtube_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
        
        if st.button("Process Video", type="primary", use_container_width=True):
            if youtube_url:
                success = process_video(youtube_url)
                if success:
                    st.success("Ready! You can now chat with the video.")
            else:
                st.warning("Please enter a valid YouTube URL.")
                
        st.markdown("---")
        st.caption("Powered by LangChain, Qdrant & Groq")

    # Main Chat Interface
    st.title("Chat")
    
    if not st.session_state.chain:
        st.info("👈 Enter a YouTube URL in the sidebar and click **Process Video** to start.")
        return

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response with loading indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                
                # Format chat history for LangChain (keep last 5 interactions)
                chat_history = []
                for msg in st.session_state.messages[:-1][-10:]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    else:
                        chat_history.append(AIMessage(content=msg["content"]))

                response = st.session_state.chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                
                # The retrieval chain returns a dict with 'answer'
                answer = response.get("answer", "Error: No answer returned.")
                message_placeholder.markdown(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
