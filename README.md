# ragdoc

Chat with your documents. Ask questions about a PDF or text file and get answers grounded in its contents.

A learning project built one small step at a time — the kind of thing AI engineers get hired to build.

---

## Quick start

**1. Install the dependency**

```bash
py -m pip install -r requirements.txt
```

**2. Set your Anthropic API key**

Get a key at [console.anthropic.com](https://console.anthropic.com/settings/keys), then in PowerShell:

```bash
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```

(That sets it for the current terminal session. See `.env.example` for the persistent option.)

**3. Run it**

```bash
py ragdoc.py path/to/your/document.pdf
```

Then ask questions. Type `exit` to quit.

> Tip: to test cheaply while learning, change `MODEL` in `ragdoc.py` to `claude-haiku-4-5`.

---

## The roadmap

This project grows one stage at a time. Each stage is a real AI-engineering skill.

| Stage | What you add | Skill you prove | Status |
|-------|-------------|-----------------|--------|
| **Day 1** | Load one doc, ask questions in the terminal | LLM + prompting | ✅ done |
| **Day 2** | Split the document into overlapping chunks | text chunking | ✅ done |
| **Day 3** | Embed the chunks (turn text into vectors) | embeddings | ⬜ next |
| **Day 4** | Search: find the top chunks for a question | similarity search | ⬜ |
| **Day 5** | Wire search into the answer (real RAG!) | retrieval | ⬜ |
| **Day 6** | Show which chunk the answer came from | citations | ⬜ |
| **Day 7** | Clean up + write it up | polish | ⬜ |

**Why Day 1 doesn't use embeddings yet:** right now we send the whole document to the model every question. That's simple and works for small files, but it's slow and costs more as documents grow. Week 1 fixes that — instead of sending everything, we'll find just the few relevant chunks and send those. That's what "RAG" (Retrieval-Augmented Generation) means.

---

## How Day 1 works

```
your document  ──►  sent to Claude  ──►  answer grounded in the document
   (PDF/text)        (whole thing)         (says "not in the document" if it can't find it)
```

The one clever bit: `cache_control` on the document tells Claude to reuse it across your questions, so you only pay full price to read it once per session.
