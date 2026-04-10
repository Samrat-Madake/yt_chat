from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import COLLECTION_NAME, EMBEDDING_MODEL

def create_vector_store(chunks, force_reload=False):
    # embedding model
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # create client
    client = QdrantClient(path="./qdrant_db")
    
    collections = [c.name for c in client.get_collections().collections]
    
    if force_reload and COLLECTION_NAME in collections:
        print(f"Clearing old collection '{COLLECTION_NAME}' for fresh ingestion...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        collections.remove(COLLECTION_NAME)

    # Check if collection exists to avoid recreation errors
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,  # all-MiniLM-L6-v2 outputs 384-dimensional vectors
                distance=models.Distance.COSINE
            )
        )
        
    # instantiate wrapper
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding
    )
    
    # Check if we have documents in it already
    collection_info = client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0 or force_reload:
        if chunks: # Only add if there are chunks
            print(f"Injecting {len(chunks)} chunks into vector store...")
            vector_store.add_documents(chunks)
    else:
        print(f"Vector store already contains {collection_info.points_count} chunks. Skipping ingestion.")

    return vector_store