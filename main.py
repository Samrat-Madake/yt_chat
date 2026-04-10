from loader import load_transcript
from chunker import chunk_documents
from vector_store import create_vector_store
from retriever import get_retriever
from rag_chain import create_rag_chain
import sys

def main():
    print("🎬 Welcome to YT_Chat CLI")
    
    # Prompt for URL
    url = input("Enter YouTube URL (or press Enter to use default demo): ").strip()
    if not url:
        url = "https://www.youtube.com/watch?v=jjp3WC8Unj8"
        print(f"Using default URL: {url}")

    print("\n[1/5] Loading transcript...")
    docs = load_transcript(url)
    if not docs:
        print("Failed to load transcript. Exiting.")
        sys.exit(1)

    print("\n[2/5] Chunking text...")
    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print("\n[3/5] Setting up Vector DB...")
    # Add force_reload=True so giving a new URL doesn't load old cache blindly for now
    # We will improve this later with multi-video support
    vector_store = create_vector_store(chunks, force_reload=True)

    print("\n[4/5] Preparing Retriever...")
    retriever = get_retriever(vector_store)

    print("\n[5/5] Creating RAG Chain...")
    chain = create_rag_chain(retriever)

    print("\n✅ System Ready!\n")

    while True:
        query = input("Ask a question about the video (or type 'exit'): ")
        
        if query.strip().lower() == "exit":
            break
        if not query.strip():
            continue

        print("\n⏳ Thinking...")
        result = chain.invoke({"question": query})

        print("\n💬 Answer:")
        print(result)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()