"""
ragdoc — web UI.

The same RAG pipeline as ask.py, but in a browser: a chat box, streamed-in
answers, and the retrieved source passages shown under each one.

Usage:
    py web.py path/to/document.pdf      # one document
    py web.py path/to/folder            # every .pdf/.txt/.md inside

Then open http://localhost:5000

Requires an Anthropic API key (see README.md).
"""

import os
import sys

import anthropic
from flask import Flask, jsonify, render_template_string, request

from ask import MAX_TOKENS, MODEL, TOP_K, build_context
from search import build_index, search

app = Flask(__name__)

# Populated at startup by main(); the index is built once and reused for
# every request, which is why the server is snappy after the first load.
INDEX = {"chunks": None, "vectors": None, "name": ""}
CLIENT: anthropic.Anthropic | None = None


PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ragdoc</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280;
    --card: #f7f7f8; --border: #e5e7eb; --accent: #2563eb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --fg: #e8e8ea; --muted: #9ca3af;
      --card: #1a1d23; --border: #2a2e37; --accent: #60a5fa;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1rem; background: var(--bg); color: var(--fg);
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 780px; margin: 0 auto; }
  h1 { margin: 0 0 .25rem; font-size: 1.6rem; }
  .sub { color: var(--muted); margin: 0 0 1.5rem; font-size: .9rem; }
  #log { display: flex; flex-direction: column; gap: 1rem; margin-bottom: 1.5rem; }
  .msg { padding: .85rem 1rem; border-radius: 10px; border: 1px solid var(--border); }
  .you { background: transparent; }
  .you .who { color: var(--accent); }
  .bot { background: var(--card); }
  .who { font-weight: 600; font-size: .8rem; text-transform: uppercase;
         letter-spacing: .04em; color: var(--muted); margin-bottom: .35rem; }
  .sources { margin-top: .85rem; border-top: 1px solid var(--border); padding-top: .7rem; }
  .sources h4 { margin: 0 0 .5rem; font-size: .75rem; text-transform: uppercase;
                letter-spacing: .04em; color: var(--muted); font-weight: 600; }
  .src { font-size: .82rem; color: var(--muted); margin-bottom: .5rem; }
  .src .meta { color: var(--accent); font-weight: 600; }
  form { display: flex; gap: .5rem; }
  input {
    flex: 1; padding: .75rem 1rem; font-size: 1rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg); color: var(--fg);
  }
  input:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button {
    padding: .75rem 1.4rem; font-size: 1rem; font-weight: 600; cursor: pointer;
    border: none; border-radius: 8px; background: var(--accent); color: #fff;
  }
  button:disabled { opacity: .5; cursor: not-allowed; }
</style>
</head>
<body>
<div class="wrap">
  <h1>ragdoc</h1>
  <p class="sub">Asking about <strong>{{ name }}</strong> &middot; {{ count }} chunks indexed</p>

  <div id="log"></div>

  <form id="f">
    <input id="q" autocomplete="off" placeholder="Ask a question about the document..." autofocus>
    <button id="go" type="submit">Ask</button>
  </form>
</div>

<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const box = document.getElementById('q');
const go = document.getElementById('go');

function addMessage(who, cls) {
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  el.innerHTML = '<div class="who"></div><div class="body"></div>';
  el.querySelector('.who').textContent = who;
  log.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return el;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = box.value.trim();
  if (!question) return;

  box.value = '';
  go.disabled = true;

  addMessage('You', 'you').querySelector('.body').textContent = question;
  const botEl = addMessage('ragdoc', 'bot');
  const body = botEl.querySelector('.body');
  body.textContent = 'Thinking...';

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await res.json();

    if (data.error) {
      body.textContent = 'Error: ' + data.error;
    } else {
      body.textContent = data.answer;

      // Show the receipts: which passages the answer came from.
      const box2 = document.createElement('div');
      box2.className = 'sources';
      box2.innerHTML = '<h4>Sources</h4>';
      data.sources.forEach((s, i) => {
        const d = document.createElement('div');
        d.className = 'src';
        const meta = document.createElement('span');
        meta.className = 'meta';
        meta.textContent = `[${i + 1}] ${s.source} (${s.score.toFixed(3)})`;
        d.appendChild(meta);
        d.appendChild(document.createTextNode(' ' + s.preview));
        box2.appendChild(d);
      });
      botEl.appendChild(box2);
    }
  } catch (err) {
    body.textContent = 'Request failed: ' + err;
  }

  go.disabled = false;
  box.focus();
  botEl.scrollIntoView({ behavior: 'smooth', block: 'end' });
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(
        PAGE, name=INDEX["name"], count=len(INDEX["chunks"])
    )


@app.route("/ask", methods=["POST"])
def ask_endpoint():
    """Retrieve the best chunks, ask Claude, return the answer plus sources."""
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400

    results = search(question, INDEX["chunks"], INDEX["vectors"], top_k=TOP_K)
    context = build_context(results)

    try:
        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=(
                "You answer questions using ONLY the provided context chunks. "
                "If the answer is not in the context, say you couldn't find it "
                "in the document. Be concise and cite which chunk(s) you used."
            ),
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
        )
    except anthropic.APIError as e:
        return jsonify({"error": str(e)}), 502

    answer = "".join(b.text for b in response.content if b.type == "text")

    sources = [
        {
            "source": chunk["source"],
            "score": float(score),
            "preview": " ".join(chunk["text"][:220].split()) + "...",
        }
        for score, chunk in results
    ]

    return jsonify({"answer": answer, "sources": sources})


def main() -> None:
    global CLIENT

    if len(sys.argv) < 2:
        print("Usage: py web.py <path-to-document-or-folder>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Not found: {path}")
        sys.exit(1)

    CLIENT = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    print(f"Building search index for: {path}")
    chunks, vectors = build_index(path)
    INDEX["chunks"] = chunks
    INDEX["vectors"] = vectors
    INDEX["name"] = os.path.basename(os.path.abspath(path))
    print(f"Indexed {len(chunks)} chunks.\n")
    print("  ragdoc is running at  http://localhost:5000")
    print("  Press Ctrl+C to stop.\n")

    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
