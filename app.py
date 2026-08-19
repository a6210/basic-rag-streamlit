import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(".env", override=True)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)
prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Answer the question using only the context below.

If the answer cannot be found in the provided context, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")
rag_chain = prompt | llm
def ask_rag(question):
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    if isinstance(response.content, list):
        answer = response.content[0]["text"]
    else:
        answer = response.content

    sources = []

    for doc in retrieved_docs:
        sources.append({
            "source": doc.metadata.get("source_file"),
            "page": doc.metadata.get("page")
        })

    return answer, sources
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 PDF RAG Assistant")

st.write(
    "Ask questions about the PDF documents stored in the knowledge base."
)

question = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):
    if question.strip():

        with st.spinner("Searching documents and generating answer..."):
            answer, sources = ask_rag(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        for source in sources:
            page = source["page"]

            if page is not None:
                page = page + 1

            st.write(
                f"📄 {source['source']} — Page {page}"
            )

    else:
        st.warning("Please enter a question.")