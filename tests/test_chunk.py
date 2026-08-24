"""Unit tests for the chunker.

These lock in the guarantees chunking must never break:
  - no text is silently dropped
  - chunks respect chunk_size
  - chunks start and end on sentence boundaries
Everything downstream (embeddings, retrieval, answers) depends on these.
"""

import pytest

from chunk import chunk_text, split_sentences


# --- basic behaviour -------------------------------------------------------

def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("Hello world. This is short.")
    assert len(chunks) == 1
    assert chunks[0] == "Hello world. This is short."


def test_long_text_produces_multiple_chunks():
    text = " ".join(f"This is sentence number {i}." for i in range(80))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Some text.", chunk_size=100, overlap=100)


# --- the guarantees that actually matter -----------------------------------

def test_no_sentence_is_lost():
    """Every sentence in the source must survive into at least one chunk."""
    sentences = [f"Unique sentence {i} about topic {i}." for i in range(50)]
    text = " ".join(sentences)

    chunks = chunk_text(text, chunk_size=180, overlap=40)
    joined = " ".join(chunks)

    missing = [s for s in sentences if s not in joined]
    assert missing == [], f"chunker dropped {len(missing)} sentences: {missing[:3]}"


def test_chunks_never_exceed_chunk_size():
    text = " ".join(f"Sentence {i} has some words in it." for i in range(100))
    chunk_size = 250

    for chunk in chunk_text(text, chunk_size=chunk_size, overlap=50):
        assert len(chunk) <= chunk_size


def test_chunks_start_and_end_on_sentence_boundaries():
    """The whole point of sentence-aware chunking: no mid-word slicing."""
    text = " ".join(
        "The mitochondria is the powerhouse of the cell. "
        "It generates the chemical energy that powers reactions."
        for _ in range(15)
    )

    chunks = chunk_text(text, chunk_size=200, overlap=40)

    for chunk in chunks:
        # Starts with a capital letter (a sentence start), not mid-word.
        assert chunk[0].isupper(), f"chunk starts mid-sentence: {chunk[:40]!r}"
        # Ends with sentence-ending punctuation.
        assert chunk.rstrip()[-1] in ".!?", f"chunk ends mid-sentence: {chunk[-40:]!r}"


def test_oversized_sentence_is_split_not_dropped():
    """A single sentence longer than chunk_size must still be chunked."""
    giant = "word " * 300  # no sentence breaks at all
    chunks = chunk_text(giant, chunk_size=200, overlap=20)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert "word" in chunks[0]


def test_overlap_repeats_content_between_chunks():
    """Consecutive chunks should share text, so boundary thoughts survive."""
    text = " ".join(f"Sentence {i} carries meaning." for i in range(40))
    chunks = chunk_text(text, chunk_size=200, overlap=60)

    assert len(chunks) >= 2
    # Some sentence from chunk 0 should reappear in chunk 1.
    first_sentences = set(split_sentences(chunks[0]))
    second_sentences = set(split_sentences(chunks[1]))
    assert first_sentences & second_sentences, "no overlap between adjacent chunks"


# --- sentence splitting ----------------------------------------------------

def test_split_sentences_handles_all_terminators():
    text = "First one. Second one! Third one? Fourth one."
    assert len(split_sentences(text)) == 4


def test_paragraph_breaks_are_hard_boundaries():
    text = "Para one sentence A. Para one sentence B.\n\nPara two sentence C."
    sentences = split_sentences(text)

    assert len(sentences) == 3
    # No sentence should span the paragraph break.
    assert not any("\n" in s for s in sentences)


def test_split_sentences_on_empty_input():
    assert split_sentences("") == []
    assert split_sentences("\n\n  \n") == []
