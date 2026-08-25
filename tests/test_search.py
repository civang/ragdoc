"""Unit tests for embeddings and retrieval.

These use the real embedding model (it's small and local), so they verify the
actual behaviour rather than a mock.
"""

import numpy as np
import pytest

from embed import cosine_similarity, embed_texts
from search import search


@pytest.fixture(scope="module")
def index():
    """A tiny corpus indexed once and shared across tests in this module."""
    chunks = [
        {"text": "Photosynthesis lets plants turn sunlight into chemical energy.",
         "source": "biology.txt"},
        {"text": "Quarterly revenue rose 12 percent on strong software sales.",
         "source": "finance.txt"},
        {"text": "Mount Everest rises 8,849 meters above sea level.",
         "source": "geography.txt"},
    ]
    vectors = embed_texts([c["text"] for c in chunks])
    return chunks, vectors


# --- cosine similarity -----------------------------------------------------

def test_identical_vectors_have_similarity_one():
    v = embed_texts(["hello world"])[0]
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_zero_vector_does_not_divide_by_zero():
    zero = np.zeros(8)
    other = np.ones(8)
    assert cosine_similarity(zero, other) == 0.0


def test_related_text_scores_higher_than_unrelated():
    a, b, c = embed_texts([
        "The cat sat on the mat.",
        "A kitten rested on the rug.",
        "Quarterly revenue grew by 12 percent.",
    ])
    assert cosine_similarity(a, b) > cosine_similarity(a, c)


# --- embeddings ------------------------------------------------------------

def test_embed_returns_one_row_per_text():
    vectors = embed_texts(["one", "two", "three"])
    assert vectors.shape[0] == 3


def test_embed_handles_empty_list():
    """An empty document produces no chunks, so this input is legitimate.

    Regression test: model.encode([]) raises, which crashed folder indexing
    whenever any file in the folder was empty.
    """
    vectors = embed_texts([])

    assert vectors.shape[0] == 0
    # Still 2-D with the right width, so np.vstack against real vectors works.
    assert vectors.ndim == 2
    assert vectors.shape[1] == embed_texts(["probe"]).shape[1]


def test_empty_vectors_can_be_stacked_with_real_ones():
    stacked = np.vstack([embed_texts([]), embed_texts(["one", "two"])])
    assert stacked.shape[0] == 2


def test_embedding_is_deterministic():
    first = embed_texts(["stable input"])[0]
    second = embed_texts(["stable input"])[0]
    assert cosine_similarity(first, second) == pytest.approx(1.0, abs=1e-6)


# --- retrieval -------------------------------------------------------------

def test_search_respects_top_k(index):
    chunks, vectors = index
    assert len(search("anything", chunks, vectors, top_k=2)) == 2
    assert len(search("anything", chunks, vectors, top_k=1)) == 1


def test_search_returns_results_in_descending_score_order(index):
    chunks, vectors = index
    results = search("How do plants make food?", chunks, vectors, top_k=3)
    scores = [score for score, _chunk in results]
    assert scores == sorted(scores, reverse=True)


def test_search_results_carry_their_source(index):
    chunks, vectors = index
    for _score, chunk in search("plants", chunks, vectors, top_k=3):
        assert "source" in chunk
        assert "text" in chunk


def test_top_k_larger_than_corpus_is_safe(index):
    chunks, vectors = index
    results = search("anything", chunks, vectors, top_k=99)
    assert len(results) == len(chunks)


@pytest.mark.parametrize(
    "question,expected_source",
    [
        ("How do plants make their own food?", "biology.txt"),
        ("Did the business have a good quarter?", "finance.txt"),
        ("How tall is the world's highest peak?", "geography.txt"),
    ],
)
def test_questions_retrieve_the_right_document(index, question, expected_source):
    """Meaning-based routing: these questions share few keywords with targets."""
    chunks, vectors = index
    _score, top_chunk = search(question, chunks, vectors, top_k=1)[0]
    assert top_chunk["source"] == expected_source
