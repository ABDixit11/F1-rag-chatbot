# app.py  –  Gradio front‑end for the F1‑Regulation RAG bot
import json
import gradio as gr
from rag_core import retrieve_regulation   # ← comes from the file you just cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Helper: turn the structured JSON answer into readable Markdown
# ──────────────────────────────────────────────────────────────────────────────
def render_answer(resp: dict) -> str:
    md = f"**Answer:**  {resp['response']}\n\n"
    if resp.get("relevant_sections"):
        md += "### Relevant Sections\n"
        for sec in resp["relevant_sections"]:
            md += f"- {sec}\n"
    if resp.get("clauses"):
        md += "\n### Clauses\n"
        for c in resp["clauses"]:
            md += f"- {c}\n"
    return md


# ──────────────────────────────────────────────────────────────────────────────
# Main callback expected by gr.ChatInterface
# ──────────────────────────────────────────────────────────────────────────────
def respond(message, history, system_message, max_tokens, temperature, top_p):
    # We ignore the sliders for max_tokens etc. because Gemini JSON mode is
    # handled inside rag_core, but they’re left in the UI for flexibility.
    resp_json = retrieve_regulation(message)
    yield render_answer(resp_json)


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI definition
# ──────────────────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value="You are a helpful assistant that cites the FIA regulations.",
            label="System message",
        ),
        gr.Slider(1, 2048, 512, step=1, label="Max new tokens"),
        gr.Slider(0.1, 4.0, 0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, 0.95, step=0.05, label="Top‑p"),
    ],
    title="F1 Regulations Chatbot",
    description=(
        "Ask anything about the 2025 Formula 1 technical regulations.  "
        "Answers cite relevant sections and clauses."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # `share=True` not needed on Hugging Face Spaces, 
    # but harmless if you run locally:
    demo.launch()
