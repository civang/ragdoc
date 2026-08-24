"""Tests for conversational follow-up handling.

A follow-up like "where is it?" carries no subject of its own. These tests
check that the retrieval query carries the subject forward from the previous
question, so vague follow-ups still land on the right chunks.
"""

import pytest

from ask import build_context, build_search_query
from embed import embed_texts
from search import search


# --- query building --------------------------------------------------------

def test_first_question_is_used_as_is():
    assert build_search_query("How tall is Everest?", []) == "How tall is Everest?"


def test_follow_up_borrows_the_previous_question():
    history = [
        {"role": "user", "content": "How tall is Everest?"},
        {"role": "assistant", "content": "It is 8,849 meters."},
    ]
    query = build_search_query("Where is it?", history)

    assert "Everest" in query
    assert "Where is it?" in query


def test_only_the_most_recent_question_is_borrowed():
    history = [
        {"role": "user", "content": "Tell me about photosynthesis."},
        {"role": "assistant", "content": "Plants convert sunlight."},
        {"role": "user", "content": "How tall is Everest?"},
        {"role": "assistant", "content": "8,849 meters."},
    ]
    query = build_search_query("Where is it?", history)

    assert "Everest" in query
    assert "photosynthesis" not in query.lower()


def test_assistant_answers_are_not_used_as_query_text():
    history = [
        {"role": "user", "content": "How tall is Everest?"},
        {"role": "assistant", "content": "Zebras live in Africa."},
    ]
    assert "Zebras" not in build_search_query("Where is it?", history)


# --- retrieval actually improves ------------------------------------------

@pytest.fixture(scope="module")
def index():
    chunks = [
        {"text": "Mount Everest rises 8,849 meters above sea level. It sits on "
                 "the border between Nepal and Tibet.", "source": "geography.txt"},
        {"text": "Quarterly revenue rose 12 percent on strong software sales.",
         "source": "finance.txt"},
        {"text": "Photosynthesis lets plants turn sunlight into chemical energy.",
         "source": "biology.txt"},
    ]
    return chunks, embed_texts([c["text"] for c in chunks])


def test_history_improves_follow_up_retrieval(index):
    """The whole point: a vague follow-up should retrieve more confidently."""
    chunks, vectors = index
    history = [
        {"role": "user", "content": "How tall is Everest?"},
        {"role": "assistant", "content": "It is 8,849 meters."},
    ]

    bare_score, _ = search("Where is it?", chunks, vectors, top_k=1)[0]
    aware_score, aware_chunk = search(
        build_search_query("Where is it?", history), chunks, vectors, top_k=1
    )[0]

    assert aware_chunk["source"] == "geography.txt"
    assert aware_score > bare_score


# --- context formatting ----------------------------------------------------

def test_build_context_labels_every_chunk(index):
    chunks, vectors = index
    results = search("How tall is the mountain?", chunks, vectors, top_k=2)
    context = build_context(results)

    assert "[Chunk 1" in context
    assert "[Chunk 2" in context
    # Each label names the source document it came from.
    for _score, chunk in results:
        assert chunk["source"] in context


def test_build_context_on_empty_results():
    assert build_context([]) == ""
