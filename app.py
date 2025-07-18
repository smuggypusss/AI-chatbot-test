import streamlit as st
import openai
import json
import faiss
import numpy as np

# --- CONFIG ---
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4o"
openai.api_key = st.secrets['openai_key']  # Secure in production

# --- LOAD VECTOR INDEX AND METADATA ---
index = faiss.read_index("vector_index.faiss")
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)


# --- UTILS ---
def get_embedding(text):
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return np.array(response.data[0].embedding).astype("float32")


def retrieve_chunks(question, top_k=10):
    query_vector = get_embedding(question)
    D, I = index.search(np.array([query_vector]), top_k)
    return [metadata[i] for i in I[0] if i < len(metadata)]


def rerank_with_gpt(question, chunks, top_n=4):
    prompt = "Given the following question and document chunks, rank the chunks by their relevance (1 to 5).\n\n"
    prompt += f"Question: {question}\n\n"
    for i, chunk in enumerate(chunks):
        prompt += f"Chunk {i}:\n{chunk['full_text']}\n\n"
    prompt += "Return a JSON array of chunk indices in order of relevance, most relevant first."

    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are a document ranking assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        ranked_indices = json.loads(response.choices[0].message.content.strip())
        return [chunks[i] for i in ranked_indices[:top_n] if i < len(chunks)]
    except Exception as e:
        #st.error(f"GPT reranking failed: {e}")
        return chunks[:top_n]


def build_context(chunks):
    return "\n\n---\n\n".join([f"{chunk['full_text']}" for chunk in chunks])


def generate_answer(context, question):
    prompt = f"""You are a helpful assistant. Answer the user's question strictly using the provided context.

Context:
{context}

Question:
{question}
"""
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "Only use the provided context to answer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


# --- STREAMLIT UI ---
st.set_page_config(page_title="🚑 ResQ Chatbot", layout="centered")
st.title("🚑 ResQ Emergency Document Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["ai"])

# Chat input
user_input = st.chat_input("Ask your emergency-related question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🔍 Searching relevant context..."):
        top_chunks = retrieve_chunks(user_input, top_k=10)
        reranked_chunks = rerank_with_gpt(user_input, top_chunks, top_n=4)
        context = build_context(reranked_chunks)

    with st.spinner("🧠 Thinking..."):
        response = generate_answer(context, user_input)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append({
        "user": user_input,
        "ai": response
    })

st.button("🔄 Clear Chat", on_click=lambda: st.session_state.chat_history.clear())
