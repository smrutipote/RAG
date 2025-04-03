import wikipedia
from transformers import pipeline
from utils import get_wikipedia_content, split_text
from models import load_models, embed_chunks, search_faiss

# Get user input
topic = input("Enter a topic to learn about: ")
document = get_wikipedia_content(topic)

if not document:
    print("Could not retrieve information.")
    exit()

# Split the document into chunks
chunks = split_text(document)

# Load models
embedding_model, qa_pipeline = load_models()

# Embed the chunks
index, chunk_embeddings = embed_chunks(chunks, embedding_model)

# Get user query
query = input("Ask a question about the topic: ")

# Search for relevant chunks
retrieved_chunks = search_faiss(query, chunks, embedding_model, index)

# Extract answer using QA model
context = " ".join(retrieved_chunks)
answer = qa_pipeline(question=query, context=context)

print(f"\nRetrieved Chunks:\n")
for chunk in retrieved_chunks:
    print("- " + chunk)

print(f"\nAnswer: {answer['answer']}")