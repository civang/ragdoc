# ragdoc 📄🔍

**Chat with your documents.** Ask questions about a PDF or text file and get answers grounded in its actual contents — with the source passages shown so you can verify every answer.

Built from scratch in 7 small steps to learn how **RAG (Retrieval-Augmented Generation)** — the technique behind "chat with your docs" products — actually works under the hood. No frameworks hiding the magic: every piece (chunking, embeddings, semantic search, retrieval) is plain, readable Python.

---

## What it looks like

```
$ py ask.py handbook.pdf

Building search index for: handbook.pdf
Indexed 214 chunks. Ask away! Type 'exit' to quit.

You: How many vacation days do new employees get?

ragdoc: New employees receive 15 paid vacation days in their first year,
increasing to 20 days after two years of service (Chunk 2).

  Sources (retrieved passages):
    [Chunk 2 | similarity 0.612] Full-time employees accrue 15 days of paid
    vacation in year one. After two years of continuous service this increases...
    [Chunk 7 | similarity 0.341] Time-off requests must be submitted at least...
```

The answer comes with **receipts** — the exact passages it was based on.

---

## How it works

RAG solves a simple problem: language models don't know the contents of *your* documents, and you can't paste a 500-page manual into every question. So instead of sending the whole document, ragdoc finds just the few passages relevant to your question and sends those.

```mermaid
flowchart LR
    A[Your PDF / text] --> B[Split into chunks]
    B --> C[Embed each chunk<br/>into a vector]
    C --> D[(Search index)]
    Q[Your question] --> E[Embed the question]
    E --> F{Find the most<br/>similar chunks}
    D --> F
    F --> G[Send top chunks<br/>+ question to Claude]
    G --> H[Grounded answer<br/>+ cited sources]
```

1. **Chunk** — split the document into small overlapping passages (`chunk.py`)
2. **Embed** — turn each chunk into a vector that captures its *meaning* (`embed.py`)
3. **Search** — embed the question and find the closest chunks by cosine similarity (`search.py`)
4. **Answer** — send only those chunks to Claude, which answers and cites them (`ask.py`)

---

## Quick start

```bash
# 1. Install dependencies
py -m pip install -r requirements.txt

# 2. Set your Anthropic API key (get one at console.anthropic.com)
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"

# 3. Chat with a document
py ask.py path/to/your/document.pdf
```

Supports `.pdf`, `.txt`, and `.md`. Type `exit` to quit.

> **Tip:** To test cheaply while experimenting, set `MODEL = "claude-haiku-4-5"` in `ask.py`.

The embedding model runs **locally** (via [model2vec](https://github.com/MinishLab/model2vec)) — no second API key, and it works offline after a one-time ~30 MB download.

---

## Project structure

Each file is one step of the pipeline and can be run on its own to see what it does:

| File | What it does | Run it standalone |
|------|-------------|-------------------|
| `chunk.py` | Split a document into overlapping chunks | `py chunk.py <file>` |
| `embed.py` | Turn chunks into meaning-vectors | `py embed.py <file>` |
| `search.py` | Find the chunks most relevant to a question | `py search.py <file>` |
| `ask.py` | **The full RAG app** — retrieve, answer, cite | `py ask.py <file>` |
| `ragdoc.py` | Day-1 baseline: sends the *whole* doc each time | `py ragdoc.py <file>` |

`ask.py` vs `ragdoc.py` is a nice before/after: the baseline sends the entire document every time (simple, but doesn't scale); the real app sends only what's relevant (cheap, fast, works on huge docs).

---

## Built in 7 days, one step at a time

| Day | Step | Skill |
|-----|------|-------|
| 1 | Chat with a whole document | LLM + prompting |
| 2 | Split documents into overlapping chunks | text chunking |
| 3 | Embed chunks into vectors | embeddings |
| 4 | Semantic search over chunks | similarity search |
| 5 | Retrieve-then-answer (real RAG) | retrieval |
| 6 | Show the source behind every answer | grounding / citations |
| 7 | Polish, document, ship | 🎉 |

See the [commit history](../../commits) — each day is one focused commit.

---

## Concepts demonstrated

- **Retrieval-Augmented Generation** end to end, with no black-box framework
- **Text chunking** with overlap to keep ideas intact across boundaries
- **Embeddings** and **cosine similarity** for meaning-based search
- **Grounding & citations** so answers are verifiable, not hallucinated
- **Prompt design** that constrains the model to answer only from provided context

---

## Ideas for going further

- Cache the index to disk so large PDFs don't re-embed on every launch
- Support a whole folder of documents at once
- Swap the local embedder for a hosted one (Voyage, OpenAI) and compare quality
- Add a small web UI (upload + chat box)
- Add evals: a set of question/expected-answer pairs to measure accuracy

---

## License

MIT — see [LICENSE](LICENSE).
