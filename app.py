import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------
# Streamlit Page Config
# -------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# -------------------------
# Load API Key
# -------------------------

load_dotenv(".env", override=True)

google_api_key = None

try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("GOOGLE_API_KEY is missing.")
    st.stop()


# -------------------------
# Embedding Model
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------
# Chroma Vector Database
# -------------------------

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)


# -------------------------
# Gemini LLM
# -------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=google_api_key,
    temperature=0
)


# -------------------------
# Prompt
# -------------------------

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


# -------------------------
# RAG Function
# -------------------------

def ask_rag(question):

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    try:
        response = rag_chain.invoke({
            "context": context,
            "question": question
        })

    except Exception as e:

        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return (
                "Gemini API quota has been reached. Please try again later.",
                []
            )

        return (
            "An error occurred while generating the answer.",
            []
        )

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


# -------------------------
# Process Uploaded PDF
# -------------------------

def process_uploaded_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)

    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = uploaded_file.name

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(
        documents
    )

    vector_db.add_documents(
        chunks
    )

    return len(documents), len(chunks)


# -------------------------
# Streamlit UI
# -------------------------

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF, process it, and then ask questions about the documents."
)


# -------------------------
# Session State
# -------------------------

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# -------------------------
# Upload PDF
# -------------------------

st.subheader("1️⃣ Upload PDF")

uploaded_file = st.file_uploader(
    "Upload a PDF to add to the knowledge base",
    type=["pdf"]
)

if uploaded_file is not None:

    if uploaded_file.name in st.session_state.processed_files:

        st.info(
            "This PDF has already been processed."
        )

    elif st.button("Process PDF"):

        with st.spinner(
            "Processing PDF..."
        ):

            pages, chunks = process_uploaded_pdf(
                uploaded_file
            )

        st.session_state.processed_files.add(
            uploaded_file.name
        )

        st.success(
            f"PDF added successfully! "
            f"{pages} pages and "
            f"{chunks} chunks processed."
        )


# -------------------------
# Ask Question
# -------------------------

st.subheader("2️⃣ Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is PyTorch?"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            answer, sources = ask_rag(
                question
            )

        st.subheader("Answer")

        st.write(
            answer
        )


        # -------------------------
        # Display Sources
        # -------------------------

        if sources:

            st.subheader("Sources")

            unique_sources = set()

            for source in sources:

                page = source["page"]

                if page is not None:
                    page = page + 1

                source_name = source["source"]

                source_key = (
                    source_name,
                    page
                )

                if source_key not in unique_sources:

                    unique_sources.add(
                        source_key
                    )

                    st.markdown(
                        f"📄 **{source_name}** — Page {page}"
                    )

    else:

        st.warning(
            "Please enter a question."
        )