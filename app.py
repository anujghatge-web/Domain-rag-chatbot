import os

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    DEFAULT_EMBEDDING_MODEL,
    MAX_FILE_SIZE_MB,
    PROVIDER_MODELS,
    build_vector_store,
    extract_documents,
    generate_answer,
    retrieve_chunks,
)

load_dotenv()

st.set_page_config(
    page_title="AI Batch - RAG Application",
    page_icon="📚",
    layout="wide",
)

PROVIDER = "Gemini"
GEMINI_MODELS = PROVIDER_MODELS[PROVIDER]


def initialise_session_state() -> None:
    defaults = {
        "vector_store": None,
        "chunks": [],
        "processed_files": [],
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_application() -> None:
    st.session_state.vector_store = None
    st.session_state.chunks = []
    st.session_state.processed_files = []
    st.session_state.messages = []


def clear_chat() -> None:
    st.session_state.messages = []


def main() -> None:
    initialise_session_state()

    st.title("📚 RAG-Based Document Question Answering")
    st.caption(
        "Upload PDF or TXT documents, process them, and ask questions "
        "using Retrieval-Augmented Generation, powered by Gemini."
    )
    

    with st.expander("How this RAG application works"):
        st.markdown(
            """
            **1. Load:** Read text from PDF or TXT files.  
            **2. Split:** Break text into overlapping chunks.  
            **3. Embed:** Convert chunks into numerical vectors.  
            **4. Store:** Save vectors in a FAISS index.  
            **5. Retrieve:** Find chunks related to the question.  
            **6. Generate:** Give the retrieved context to the Gemini LLM.
            """
        )

    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox("Gemini model", GEMINI_MODELS)
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            st.error("GEMINI_API_KEY is not set. Add it to your .env file.")

        top_k = st.slider("Chunks to retrieve", 1, 8, 4)
        chunk_size = st.slider("Chunk size", 300, 1500, 700, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, 20)
        st.caption(f"Embedding model: `{DEFAULT_EMBEDDING_MODEL}`")

        st.divider()
        st.header("Documents")
        uploaded_files = st.file_uploader(
            "Upload one or more documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help=f"Max {MAX_FILE_SIZE_MB}MB per file. PDF and TXT only.",
        )

        if st.button(
            "Process Documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        ):
            if chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
                st.stop()

            try:
                with st.spinner("Building the RAG knowledge base..."):
                    documents = extract_documents(uploaded_files)
                    vector_store, chunks = build_vector_store(
                        documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                st.session_state.vector_store = vector_store
                st.session_state.chunks = chunks
                st.session_state.processed_files = [
                    f.name for f in uploaded_files
                ]
                st.session_state.messages = []
                st.success(
                    f"Processed {len(uploaded_files)} file(s) into "
                    f"{len(chunks)} chunks."
                )
            except Exception as error:
                st.error(f"Document processing failed: {error}")
                st.stop()

        col_a, col_b = st.columns(2)
        if col_a.button("Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()
        if col_b.button("Reset all", use_container_width=True):
            reset_application()
            st.rerun()

    if st.session_state.vector_store is None:
        st.info("Upload files and click **Process Documents**.")
        st.markdown(
            """
            ### Students learn
            - PDF/TXT loading
            - Text chunking and overlap
            - Sentence-transformer embeddings
            - FAISS vector retrieval
            - Prompt construction
            - Grounded LLM answers with sources
            """
        )
        return

    st.success("Knowledge base ready: " + ", ".join(st.session_state.processed_files))

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(st.session_state.processed_files))
    col2.metric("Chunks", len(st.session_state.chunks))
    col3.metric("Vector search", "FAISS")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['source']} — chunk {source['chunk_id']}**"
                        )
                        st.caption(f"Similarity: {source['score']:.3f}")
                        st.write(source["text"])

    question = st.chat_input("Ask a question about the documents")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    sources = retrieve_chunks(
                        question,
                        st.session_state.vector_store,
                        st.session_state.chunks,
                        top_k=top_k,
                    )
                    answer = generate_answer(
                        question,
                        sources,
                        provider=PROVIDER,
                        api_key=api_key,
                        model_name=model_name,
                    )
                st.markdown(answer)
                with st.expander("Retrieved sources"):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']} — chunk {source['chunk_id']}**"
                        )
                        st.caption(f"Similarity: {source['score']:.3f}")
                        st.write(source["text"])

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as error:
            st.error(f"Unable to answer: {error}")


if __name__ == "__main__":
    main()
