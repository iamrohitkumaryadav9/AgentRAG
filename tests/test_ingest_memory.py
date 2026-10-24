import io
from pathlib import Path

import pytest

from agentrag.ingest import (
    UnreadableDocument,
    build_corpus,
    build_corpus_from_files,
    chunk_text,
    extract_text,
)


def make_pdf(text: str) -> bytes:
    """A minimal valid one-page PDF whose text layer is ``text`` (no third-party PDF writer needed)."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET" if text else ""
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def sentences(n: int, tag: str = "x") -> bytes:
    """About n short sentences; with a 100-character chunk size each sentence is roughly its own chunk."""
    return " ".join(f"Sentence number {i} about {tag} stands alone here." for i in range(n)).encode()


# -- extraction ---------------------------------------------------------------------------------


def test_extracts_text_from_markdown_plain_text_and_pdf():
    assert extract_text("notes.md", b"# Title\n\nSome *markdown* body.") == "# Title\n\nSome *markdown* body."
    assert extract_text("a.TXT", "  plain text  ".encode()) == "plain text"  # extension match is case-insensitive
    assert "Hello PDF world" in extract_text("paper.pdf", make_pdf("Hello PDF world"))


def test_non_utf8_text_is_decoded_leniently_rather_than_failing():
    assert "caf" in extract_text("latin.txt", "café au lait".encode("latin-1"))


@pytest.mark.parametrize(
    ("name", "data", "reason"),
    [
        ("empty.txt", b"", "empty"),
        ("blank.md", b"   \n\t  ", "no text"),
        ("notes.docx", b"PK\x03\x04", "unsupported file type '.docx'"),
        ("noext", b"hello", "unsupported file type 'none'"),
        ("broken.pdf", b"this is not a pdf at all", "could not read this PDF"),
        ("image.txt", b"\x00\x01\x02binary\x00data", "binary file"),
        ("scanned.pdf", make_pdf(""), "no extractable text"),
    ],
)
def test_unreadable_files_raise_a_user_presentable_reason(name, data, reason):
    with pytest.raises(UnreadableDocument) as err:
        extract_text(name, data)
    assert reason in err.value.reason and err.value.name == name


def test_a_password_protected_pdf_is_reported_not_crashed():
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(UnreadableDocument, match="password-protected"):
        extract_text("locked.pdf", buffer.getvalue())


# -- building a corpus with a chunk cap -------------------------------------------------------------


def test_in_memory_ingest_matches_disk_ingest_on_the_sample_docs():
    root = Path("data/sample_docs")
    files = [(p.name, p.read_bytes()) for p in sorted(root.glob("*.md"))]
    from_memory = build_corpus_from_files(files, chunk_size=500, chunk_overlap=80)
    from_disk = build_corpus(root, chunk_size=500, chunk_overlap=80)
    assert [(c.source, c.position, c.text) for c in from_memory.chunks] == [
        (c.source, c.position, c.text) for c in from_disk
    ]
    assert not from_memory.truncated and from_memory.cap is None


def test_the_same_chunker_is_used():
    text = sentences(40).decode()
    result = build_corpus_from_files([("a.txt", text.encode())], chunk_size=200, chunk_overlap=20)
    assert [c.text for c in result.chunks] == chunk_text(text, chunk_size=200, chunk_overlap=20)


def test_the_cap_indexes_exactly_up_to_it_and_reports_what_was_left_out():
    files = [(f"{tag}.txt", sentences(30, tag)) for tag in ("one", "two", "three")]
    uncapped = build_corpus_from_files(files, chunk_size=120, chunk_overlap=10)
    per_file = [r.chunks_available for r in uncapped.files]
    assert all(n > 5 for n in per_file)

    cap = per_file[0] + 7  # file one fits whole, file two is cut after 7 chunks, file three is never read
    capped = build_corpus_from_files(files, chunk_size=120, chunk_overlap=10, max_chunks=cap)
    assert capped.indexed == cap and capped.truncated and capped.cap == cap
    one, two, three = capped.files
    assert (one.status, one.chunks_indexed, one.chunks_available) == ("indexed", per_file[0], per_file[0])
    assert (two.status, two.chunks_indexed, two.chunks_available) == ("partial", 7, per_file[1])
    assert (three.status, three.chunks_indexed, three.chunks_available) == ("skipped", 0, None)
    assert [c.position for c in capped.chunks if c.source == "two.txt"] == list(range(7))  # a clean prefix

    summary = capped.summary()
    assert f"Indexed {cap:,} chunks" in summary and f"cap of {cap:,} chunks was reached" in summary
    assert f"two.txt: indexed 7 of {per_file[1]:,} chunks" in summary
    assert "three.txt" in summary and "Not read because the cap was already reached" in summary


def test_a_cap_that_is_not_reached_is_not_reported_as_truncation():
    result = build_corpus_from_files([("a.txt", sentences(5))], chunk_size=120, chunk_overlap=10, max_chunks=1000)
    assert not result.truncated and result.files[0].status == "indexed"
    assert "cap" not in result.summary()


def test_a_file_that_exactly_fills_the_cap_is_complete_not_partial():
    n = build_corpus_from_files([("a.txt", sentences(20))], chunk_size=120, chunk_overlap=10).indexed
    exact = build_corpus_from_files([("a.txt", sentences(20)), ("b.txt", sentences(3))], chunk_size=120,
                                    chunk_overlap=10, max_chunks=n)
    assert exact.files[0].status == "indexed" and exact.files[1].status == "skipped" and exact.truncated


def test_unreadable_files_are_reported_and_do_not_stop_the_others():
    result = build_corpus_from_files(
        [("bad.pdf", b"nope"), ("good.txt", sentences(4)), ("empty.md", b"")], chunk_size=120, chunk_overlap=10
    )
    statuses = {f.name: f.status for f in result.files}
    assert statuses == {"bad.pdf": "unreadable", "good.txt": "indexed", "empty.md": "unreadable"}
    assert result.indexed > 0
    assert "bad.pdf: skipped, could not read this PDF" in result.summary()
    assert "empty.md: skipped, the file is empty" in result.summary()


def test_all_files_unreadable_gives_an_empty_result_with_reasons():
    result = build_corpus_from_files([("a.pdf", b"x"), ("b.txt", b"")])
    assert result.indexed == 0 and not result.truncated and len(result.files) == 2


def test_duplicate_file_names_stay_distinguishable_in_citations():
    result = build_corpus_from_files([("same.txt", b"First file text."), ("same.txt", b"Second file text.")])
    assert [f.name for f in result.files] == ["same.txt", "same.txt (2)"]
    assert {c.source for c in result.chunks} == {"same.txt", "same.txt (2)"}


def test_progress_is_reported_in_order_and_finishes_at_one():
    seen = []
    files = [("a.txt", sentences(3)), ("b.txt", sentences(3))]
    build_corpus_from_files(files, progress=lambda fraction, message: seen.append((fraction, message)))
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == 1.0
    assert any("a.txt" in m for _, m in seen) and any("b.txt" in m for _, m in seen)


def test_building_from_memory_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    build_corpus_from_files([("a.txt", sentences(10)), ("p.pdf", make_pdf("some pdf text"))])
    assert list(tmp_path.iterdir()) == []
