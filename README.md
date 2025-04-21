# F1 Regulations RAG Chatbot

A Retrieval‑Augmented Generation chatbot that lets you query the 2025 Formula 1 technical regulations in seconds—no more manual PDF searches, just ask and get concise, citation‑backed answers.

---

## 🚀 Problem Statement & Solution

**Problem:**\
Formula 1 regulations are published as lengthy PDFs. Manually searching them is slow and error‑prone.

**Solution:**\
We built a RAG (Retrieval‑Augmented Generation) chatbot that:

1. **Ingests** regulation PDFs from `data/`
2. **Splits** them into 1 000‑char overlapping chunks with LangChain
3. **Embeds** chunks into a ChromaDB + FAISS vector store
4. **Retrieves** the top‑K relevant sections for each user query
5. **Generates** structured JSON answers via Google Gen‑AI or Zephyr‑7B
6. **Serves** everything in a Gradio ChatInterface, hosted on Hugging Face Spaces

---

## 🔧 Tech Stack

- **Language & Parsing:** Python, PyMuPDF
- **Chunking:** LangChain’s RecursiveCharacterTextSplitter
- **Vector DB:** ChromaDB (PersistentClient) + FAISS-CPU
- **Embeddings:** `text-embedding-004` via `google-genai` or HF’s `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** Google Gemini (Gen‑AI SDK) or Zephyr‑7B (Hugging Face InferenceClient)
- **UI & Deployment:** Gradio 4.x on Hugging Face Spaces
- **Data Storage:** Git LFS for PDF files

---

## ⚙️ Installation & Local Setup

1. **Clone & venv**

   ```bash
   git clone https://github.com/<your-org>/f1-regulations-rag-chatbot.git
   cd f1-regulations-rag-chatbot
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add regulation PDFs**

   ```bash
   mkdir -p data
   # Copy FIA regulation PDFs into data/
   git lfs install
   git lfs track "*.pdf"
   git add .gitattributes data/*.pdf
   git commit -m "Add regulation PDFs via LFS"
   ```

4. **Set environment variables**

   ```bash
   export GOOGLE_API_KEY="sk-..."      # for google-genai
   export PDF_DIR="data"               # default data folder
   export CHROMA_DIR="chroma_db"       # default vector store path
   ```

5. **Run the app**

   ```bash
   python app.py
   ```

   Open your browser at [http://127.0.0.1:7860](http://127.0.0.1:7860) and ask a question!

---

## ☁️ Deploy to Hugging Face Spaces

1. Ensure PDFs are committed with Git LFS.
2. In Space → Settings → Secrets, add `GOOGLE_API_KEY`.
3. Push all code and data:
   ```bash
   git add .
   git commit -m "Deploy chatbot to HF Space"
   git push origin main
   ```

Build takes \~3–5 min. Your Space URL will be [https://huggingface.co/spaces/](https://huggingface.co/spaces/)/f1-regulations-rag-chatbot

---

## 💬 Usage Examples

- “What are the tire specifications for 2025?”
- “Explain the rules on rear bodywork sidepods.”
- “What’s the minimum car weight?”

Each answer includes:

- **Answer:** A short, concise paragraph
- **Relevant Sections:** Bullet list of articles or pages
- **Clauses:** Specific sub‑clauses cited

---

## 🌱 Future Work

- **Voice‑enabled queries** for hands‑free interaction
- **Multi‑lingual support** to serve a global audience
- **Function‑calling** for structured outputs (tables, exact references)

---

## 🎓 Acknowledgements

Part of the **5‑day Google × Kaggle Gen AI Intensive Course**, where I refreshed my knowledge of:

- Large Language Models & Prompt Engineering
- Embeddings & AI Agents
- Domain‑specific LLMs & MLOps via Gemini APIs

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

