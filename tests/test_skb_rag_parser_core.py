from __future__ import annotations

from app.skb.models import WikiPage
from app.skb.parser import chunk_page, clean_dokuwiki_markup, parse_sections


def _page(raw: str) -> WikiPage:
    return WikiPage(
        page_id="spay:faq:test",
        title="Payroll test",
        source_url="http://skb.uniconsults.mu/doku.php?id=spay%3Afaq%3Atest",
        module="Payroll",
        raw_text=raw,
        content_hash="a" * 64,
    )


def test_parser_preserves_heading_hierarchy_and_cleans_wiki_markup():
    page = _page(
        """
====== Payroll test ======
Intro with **bold** and [[spay:other|a useful link]].

===== Setup =====
Use {{:screen.png|the payroll screen}}.\\\\
Then save.

==== Validation ====
  * Check the employee
  * Check the period
"""
    )

    sections = parse_sections(page)

    assert [section.heading for section in sections] == [
        "Payroll test",
        "Setup",
        "Validation",
    ]
    assert sections[-1].heading_path == ("Payroll test", "Setup", "Validation")
    assert "**" not in sections[0].text
    assert "a useful link" in sections[0].text
    assert "the payroll screen" in sections[1].text
    assert "- Check the employee" in sections[-1].text


def test_markup_cleaner_does_not_damage_http_urls():
    cleaned = clean_dokuwiki_markup("See http://skb.uniconsults.mu/path and //important//.")
    assert "http://skb.uniconsults.mu/path" in cleaned
    assert "important" in cleaned
    assert "//important//" not in cleaned


def test_chunking_is_deterministic_section_aware_and_bounded():
    body = " ".join(f"sentence-{number}." for number in range(250))
    page = _page(f"====== Payroll test ======\n{body}")

    first = chunk_page(page, chunk_size=300, chunk_overlap=50, min_chunk_size=30)
    second = chunk_page(page, chunk_size=300, chunk_overlap=50, min_chunk_size=30)

    assert len(first) > 2
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.page_id == page.page_id for chunk in first)
    assert all(chunk.source_url == page.source_url for chunk in first)
    assert all(chunk.embedding_text.startswith("Module: Payroll\nPage: Payroll test") for chunk in first)
    assert all(len(chunk.text) <= 330 for chunk in first[:-1])
    assert all(len(chunk.text) >= 30 for chunk in first)


def test_short_sections_are_merged_and_isolated_media_noise_is_dropped():
    combined = _page(
        """
====== Payroll test ======
Short note.

===== Procedure =====
This is a sufficiently detailed procedure that contains useful grounding text
for a user who needs to complete the payroll operation correctly and safely.
"""
    )
    chunks = chunk_page(combined, chunk_size=300, chunk_overlap=30, min_chunk_size=80)

    assert len(chunks) == 1
    assert "Short note." in chunks[0].text
    assert len(chunks[0].text) >= 80

    media_only = _page("====== Screenshot ======\n{{:spay:images:x.png}}")
    assert chunk_page(media_only, min_chunk_size=80) == []
