Wikipedia-Based Question Answering System using RAG
check out this project deployed on hugging face : https://huggingface.co/spaces/smrup/RAG_using_Wikipedia_based_QA


This project retrieves Wikipedia content, processes it into smaller text chunks, and allows users to query information using FAISS-based retrieval and a Question Answering (QA) model.

📌 Features
✅ Fetches Wikipedia content for any given topic.
✅ Splits the content into manageable text chunks.
✅ Uses Sentence Transformers to generate vector embeddings.
✅ Implements FAISS for fast similarity search.
✅ Answers user questions using Roberta-based QA model.

📂 Project Structure
wikipedia-qa/
│── app.py                # Main script to run the project
│── utils.py              # Utility functions (text processing, Wikipedia retrieval)
│── models.py             # Model handling (loading, embedding, FAISS search)
│── requirements.txt      # Dependencies
│── README.md             # Project documentation


🛠 How It Works
1️⃣ User enters a Wikipedia topic → Wikipedia content is retrieved.
2️⃣ Text is split into smaller chunks → Helps with efficient search.
3️⃣ Chunks are embedded using Sentence Transformers → Converts text into vectors.
4️⃣ FAISS index is created → Enables fast retrieval of relevant text.
5️⃣ User asks a question → The most relevant chunks are retrieved.
6️⃣ QA model extracts the answer → Using Roberta-based SQuAD2 model.

📌 Example Usage
Enter a topic to learn about: Artificial Intelligence
Ask a question about the topic: What is AI?
🔹 Retrieved Chunks:

AI is the simulation of human intelligence in machines.

It includes learning, reasoning, and self-correction.

AI is used in various applications like chatbots, robotics, etc.

🔹 Answer:

"AI is the simulation of human intelligence in machines."

📦 Dependencies
The following libraries are required:
wikipedia-api
transformers
sentence-transformers
faiss-cpu
numpy


