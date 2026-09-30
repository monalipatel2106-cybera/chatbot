import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

st.set_page_config(
    page_title="Mini RAG Q&A",
    page_icon="🤖"
)

st.title("🤖 Mini RAG Q&A")

st.write(
    "Paste a document, store it in ChromaDB, and ask questions about it."
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="documents"
)


document = st.text_area(
    "Paste your document here:",
    height=300,
    placeholder="Paste your notes, textbook content, PDF text, etc."
)


if st.button("📥 Add Document"):

    if not document.strip():
        st.warning("Please paste a document first.")

    else:
        words = document.split()

        chunks = []

        for i in range(0, len(words), 500):
            chunk = " ".join(words[i:i + 500])
            chunks.append(chunk)

        embeddings = embedding_model.encode(
            chunks
        ).tolist()

        ids = []

        for i in range(len(chunks)):
            ids.append(f"chunk_{i}")

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

        st.success(
            f"Added {len(chunks)} chunks successfully!"
        )


st.subheader("Ask a Question")

question = st.text_input(
    "Enter your question"
)


if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        question_embedding = embedding_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=3
        )

        relevant_chunks = results["documents"][0]

        context = "\n\n".join(
            relevant_chunks
        )

        prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        st.subheader("Answer")

        st.write(
            response["message"]["content"]
        )