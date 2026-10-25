"""Tests for building an index from a real-world folder.

Folders in the wild are messy: nested directories, files saved in older
encodings, the occasional corrupt PDF. None of that should take down the
whole index or silently drop content.
"""

import pytest

from chunk import load_text
from search import build_index


@pytest.fixture
def docs(tmp_path):
    """A messy but realistic document folder."""
    (tmp_path / "top.txt").write_text("Mountains rise above the treeline.", encoding="utf-8")

    nested = tmp_path / "subdir"
    nested.mkdir()
    (nested / "nested.txt").write_text("Rivers carve valleys over time.", encoding="utf-8")

    # A file saved in an older encoding.
    (tmp_path / "latin1.txt").write_bytes("Café résumé notes.".encode("latin-1"))

    return tmp_path


# --- reading files ---------------------------------------------------------

def test_utf8_file_round_trips(tmp_path):
    path = tmp_path / "u.txt"
    path.write_text("Café — naïve résumé", encoding="utf-8")
    assert "Café" in load_text(str(path))


def test_non_utf8_file_does_not_raise(tmp_path):
    """Regression: a latin-1 file used to crash indexing for the whole folder."""
    path = tmp_path / "l.txt"
    path.write_bytes("Café résumé".encode("latin-1"))

    text = load_text(str(path))  # must not raise
    assert "Caf" in text


# --- folder traversal ------------------------------------------------------

def test_subdirectories_are_indexed(docs):
    """Regression: nested files used to be silently skipped."""
    chunks, _vectors = build_index(str(docs), use_cache=False)
    sources = {c["source"] for c in chunks}

    assert "subdir/nested.txt" in sources
    assert "top.txt" in sources


def test_nested_files_keep_their_relative_path(tmp_path):
    """Same basename in two folders must stay distinguishable."""
    for folder in ("a", "b"):
        d = tmp_path / folder
        d.mkdir()
        (d / "notes.txt").write_text(f"Notes from folder {folder}.", encoding="utf-8")

    chunks, _vectors = build_index(str(tmp_path), use_cache=False)
    sources = {c["source"] for c in chunks}

    assert sources == {"a/notes.txt", "b/notes.txt"}


def test_hidden_directories_are_skipped(tmp_path):
    (tmp_path / "real.txt").write_text("Real content here.", encoding="utf-8")
    hidden = tmp_path / ".ragdoc_cache"
    hidden.mkdir()
    (hidden / "junk.txt").write_text("Cache junk.", encoding="utf-8")

    chunks, _vectors = build_index(str(tmp_path), use_cache=False)
    assert {c["source"] for c in chunks} == {"real.txt"}


def test_docx_paragraphs_and_tables_are_read(tmp_path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Glaciers shape the landscape.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Leave"
    table.rows[0].cells[1].text = "15 days"
    path = tmp_path / "policy.docx"
    doc.save(path)

    text = load_text(str(path))

    assert "Glaciers shape the landscape." in text
    assert "Leave | 15 days" in text


def test_docx_files_in_a_folder_are_indexed(docs):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Deserts receive very little rain.")
    doc.save(docs / "notes.docx")

    chunks, _vectors = build_index(str(docs), use_cache=False)

    assert "notes.docx" in {c["source"] for c in chunks}


# --- resilience ------------------------------------------------------------

def test_one_unreadable_file_does_not_break_the_index(docs):
    """A corrupt file should be skipped, not fatal."""
    (docs / "corrupt.pdf").write_bytes(b"this is definitely not a pdf")

    chunks, _vectors = build_index(str(docs), use_cache=False)
    sources = {c["source"] for c in chunks}

    assert "corrupt.pdf" not in sources
    assert "top.txt" in sources          # the good files still made it
    assert "subdir/nested.txt" in sources


def test_empty_file_contributes_nothing_but_does_not_crash(docs):
    (docs / "empty.txt").write_text("", encoding="utf-8")
    (docs / "blank.txt").write_text("   \n\n  ", encoding="utf-8")

    chunks, vectors = build_index(str(docs), use_cache=False)

    assert len(chunks) == vectors.shape[0]
    assert "empty.txt" not in {c["source"] for c in chunks}


def test_folder_with_no_supported_files_raises(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    with pytest.raises(ValueError, match="No supported documents"):
        build_index(str(tmp_path), use_cache=False)


def test_folder_where_everything_is_unreadable_raises(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"not a pdf")
    (tmp_path / "b.pdf").write_bytes(b"also not a pdf")

    with pytest.raises(ValueError, match="No readable content"):
        build_index(str(tmp_path), use_cache=False)
