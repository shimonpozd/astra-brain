import pytest
from utils.text_processing import fuzzy_find_anchor
from brain_service.models.db import SederSegment


def test_fuzzy_find_anchor_exact():
    text = "ואאי פשיטא ליה אמאי סופג ארבעים"
    anchor = "אמאי סופג"
    start, end = fuzzy_find_anchor(text, anchor)
    assert start != -1
    assert text[start:end] == anchor


def test_fuzzy_find_anchor_with_nikud_and_punctuation():
    text = "וְאִי פְּשִׁיטָא לֵיהּ, אַמַּאי סוֹפֵג אַרְבָּעִים וְתוּ לָא? לִילְקֵי שְׁמוֹנִים!"
    anchor = "ואי פשיטא ליה אמאי סופג"
    start, end = fuzzy_find_anchor(text, anchor)
    assert start == 0
    assert end > 0


def test_fuzzy_find_anchor_not_found():
    text = "וְאִי פְּשִׁיטָא לֵיהּ"
    anchor = "מפתח של גשמים"
    start, end = fuzzy_find_anchor(text, anchor)
    assert start == -1
    assert end == -1


def test_seder_segment_fields():
    seg = SederSegment(
        source_ref="Chullin 91a:11",
        sub_index=1,
        order_index=200,
        role="Defense",
        start_anchor="הכא במאי עסקינן",
        end_anchor="עד שיהא בו כזית",
        start_word_idx=11,
        end_word_idx=25,
        text_he="הכא במאי עסקינן - כגון דלית בו כזית",
        text_ru="Здесь речь идет о случае, когда в жиле нет объема с оливу",
    )
    assert seg.source_ref == "Chullin 91a:11"
    assert seg.sub_index == 1
    assert seg.order_index == 200
    assert seg.role == "Defense"
    assert seg.start_word_idx == 11
    assert seg.end_word_idx == 25
