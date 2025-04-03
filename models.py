import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def load_models():
    """Loads the embedding model and QA model."""
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    qa_model_name = "deepset/roberta-base-squad2"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

    return embedding_model, qa_pipeline

def embed_chunks(chunks, embedding_model):
    """Embeds text chunks and stores them in a FAISS index."""
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, embeddings

def search_faiss(query, chunks, embedding_model, index, k=3):
    """Finds the top-k most relevant text chunks for a query."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]
