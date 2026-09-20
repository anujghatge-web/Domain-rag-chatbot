# Domain-Specific RAG Chatbot for PDF Question Answering

A Streamlit chatbot that answers questions from uploaded PDF/TXT documents
(course notes, company policies, manuals, legal documents, training
material) using Retrieval-Augmented Generation (RAG).

## Architecture / Workflow

```
Upload PDF/TXT files
        |
Extract text (pypdf) + keep source/page metadata
        |
Split text into overlapping chunks
        |
Embed chunks (Sentence-Transformers, all-MiniLM-L6-v2)
        |
Store embeddings in a FAISS index
        |
User asks a question
        |
Embed the question -> retrieve top-k similar chunks (FAISS)
        |
Send retrieved context + question to the Groq LLM
        |
Display grounded answer with source document, page, and similarity score
```

## Project Structure

```
domain_rag_chatbot/
|-- app.py              # Streamlit UI (upload, chat, sources)
|-- rag_engine.py        # Extraction, chunking, embeddings, retrieval, generation
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
|-- documents/           # (optional) sample PDFs for demo/testing
|-- tests/
|   |-- test_questions.csv
```

> Note: this implementation keeps the pipeline in a single `rag_engine.py`
> module instead of splitting it into `rag_pipeline.py` /
> `document_loader.py` / `vector_store.py` / `prompt.py` as in the
> suggested folder structure. The functionality is equivalent -- feel free
> to split it into those files if you want extra practice with modular
> code organization.

## Setup

1. **Clone and enter the project**
   ```bash
   git clone <your-repo-url>
   cd domain_rag_chatbot
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Add your API key (only for Groq or Gemini)**
   ```bash
   cp .env.example .env
   # then edit .env and set GROQ_API_KEY or GEMINI_API_KEY
   ```
   You can also paste the key directly into the sidebar at runtime. If you
   pick the **Ollama (local, no key)** provider, skip this step entirely —
   see below instead.

   | Provider | Key needed? | Where to get one |
   |---|---|---|
   | Groq | Yes | https://console.groq.com/keys |
   | Gemini | Yes (free tier) | https://aistudio.google.com/apikey |
   | Ollama (local) | **No** | Install from https://ollama.com |

   To use Ollama: install it, then in a terminal run
   ```bash
   ollama serve            # starts the local server (may already be running)
   ollama pull llama3.1    # downloads a model once, ~4-5GB
   ```
   Then just pick **Ollama (local, no key)** in the sidebar — everything
   runs on your machine, so it isn't affected by Groq or any other
   provider being down.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage

1. In the sidebar, upload one or more PDF/TXT files (max 20MB each).
2. Click **Process Documents** to extract, chunk, embed, and index them.
3. Ask questions in the chat box at the bottom.
4. Expand **Retrieved sources** under any answer to see which chunks,
   documents, and pages were used, along with similarity scores.
5. Use **Clear chat** to reset the conversation while keeping the
   knowledge base, or **Reset all** to remove documents and start over.

## Configuration options (sidebar)

| Setting | Purpose |
|---|---|
| LLM provider | Groq, Gemini, or local Ollama — switch freely between them |
| Model | Which model to use, options depend on the chosen provider |
| Chunks to retrieve (top-k) | How many chunks are sent as context (1-8) |
| Chunk size / overlap | Controls how text is split before embedding |

## Responsible AI & Security

This project follows the guardrails required by the assignment:

- **No hardcoded keys.** The Groq API key is read from `.env` (git-ignored)
  or entered by the user at runtime; it is never written into source code.
- **Prompt-injection guard.** The system prompt explicitly instructs the
  model to treat retrieved document text as untrusted data, not as
  instructions, and to ignore any embedded attempts to change its rules
  (see the "prompt-injection" row in `tests/test_questions.csv`).
- **Grounded answers only.** The model is instructed to answer solely from
  retrieved context and to return a fixed fallback message when the answer
  isn't present, rather than inventing facts.
- **Verification disclaimer.** The UI displays a persistent notice asking
  users to verify important or high-stakes answers against the source
  documents.
- **Upload limits.** Only `.pdf` and `.txt` files are accepted, and each
  file is capped at 20MB (`MAX_FILE_SIZE_MB` in `rag_engine.py`).
- **Confidential documents.** Do not upload confidential or sensitive
  documents to a shared/deployed instance without permission from their
  owner -- uploaded content lives in memory only and is cleared on
  **Reset all**, but nothing is encrypted at rest by default.

## Testing & Evaluation

`tests/test_questions.csv` is a starter sheet with 16 questions covering:
- **In-document** questions the chatbot should answer correctly with the
  right source/page.
- **Unavailable** questions (not covered by the uploaded documents) where
  the chatbot should return the fallback message instead of guessing.
- A **prompt-injection** question to confirm the chatbot refuses to follow
  instructions embedded in retrieved text.

Fill in the `Retrieved Source` and `Correct?` columns after running each
question against your own uploaded documents, and expand the sheet to 15+
rows as required by the assignment deliverables.

## Suggested Viva Questions

- What is RAG and why is it used?
- Why do we split documents into chunks?
- What is an embedding, and what does a vector database store?
- How does cosine similarity (or inner product on normalized vectors)
  help retrieval?
- Why can a RAG chatbot still produce an incorrect answer?
- How will you test whether retrieval is working correctly?
- What happens when the answer is not present in the documents?
- How does this project defend against prompt injection embedded in a
  document?

## Deployment

The app can be deployed as-is to **Streamlit Community Cloud**, **Render**,
or inside a **Docker** container. Set `GROQ_API_KEY` as an environment
variable / secret on whichever platform you use -- do not commit it.
