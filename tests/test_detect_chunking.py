"""The shared chunker: word windows, token-limit fitting, chunk labels."""

import re

import pytest

from injbench.detect.chunking import (MAX_WORDS, OVERLAP_WORDS, Chunk, chunk_label, chunk_text,
                                      fit_chunks, segment_chunks)


def words(text):
    return re.findall(r"\S+", text)


def test_short_text_is_one_chunk():
    text = "  It worked fine until last week.\n\nThen it broke.  "
    (c,) = chunk_text(text)
    assert c.text == text.strip() and text[c.start:c.end] == c.text


def test_empty_text_is_one_chunk():
    assert chunk_text("") == [Chunk(0, 0, "")]
    assert len(chunk_text("   ")) == 1


def test_long_text_windows_overlap_and_cover_every_word():
    text = " ".join(f"w{i}" for i in range(475))
    chunks = chunk_text(text)
    assert all(len(words(c.text)) <= MAX_WORDS for c in chunks)
    assert all(text[c.start:c.end] == c.text for c in chunks)
    covered = [w for c in chunks for w in words(c.text)]
    assert set(covered) == set(words(text))
    for a, b in zip(chunks, chunks[1:]):
        assert words(a.text)[-OVERLAP_WORDS:] == words(b.text)[:OVERLAP_WORDS]
    assert words(chunks[-1].text)[-1] == "w474"


def test_bad_parameters():
    with pytest.raises(ValueError):
        chunk_text("a b", max_words=2, overlap=2)
    with pytest.raises(ValueError):
        chunk_text("a b", max_words=0)


def test_fit_chunks_respects_a_token_limit_without_truncating():
    text = " ".join(f"w{i}" for i in range(300)) + " " + "x" * 5000

    def count(t):  # a fake tokenizer: 3 tokens per word, 1 per 10 characters, 2 specials
        return 2 + sum(max(3, len(w) // 10) for w in words(t))

    chunks = segment_chunks(text, count, max_tokens=128)
    assert all(count(c.text) <= 128 for c in chunks)
    assert all(text[c.start:c.end] == c.text for c in chunks)
    assert "".join(c.text for c in chunks if c.text.startswith("x")) == "x" * 5000
    assert {w for c in chunks for w in words(c.text) if w.startswith("w")} == {
        f"w{i}" for i in range(300)}


def test_fit_chunks_keeps_chunks_that_fit():
    chunks = chunk_text("a b c")
    assert fit_chunks(chunks, lambda t: 5) == chunks


def test_chunk_labels():
    c = Chunk(0, 100, "x" * 100)
    assert chunk_label(c, None) == 0
    assert chunk_label(c, (200, 300)) == 0  # no overlap
    assert chunk_label(c, (60, 140)) == 1  # half of the payload
    assert chunk_label(c, (0, 1000)) == 1  # the chunk is all payload
    assert chunk_label(c, (95, 300)) is None  # a sliver: left out of training
