"""Regression tests for the text parsing that graders depend on (found in real runs)."""

from injbench.graders import KEY_RE, REFUSAL_RE
from injbench.tasks import standalone_ints


def test_standalone_ints_ignore_keys_dates_and_decimals():
    text = "SLA summary 2026-11-09: 5 tickets breached (INC-1047, P1), ratio 3.5."
    assert standalone_ints(text) == {5}


def test_standalone_ints_next_to_chinese_text():
    # Qwen3.5-9B wrote a report like this; \w-based boundaries missed the 5.
    assert standalone_ints("本周共有5个SLA超时工单因故延误") == {5}
    assert standalone_ints("超时0个") == {0}


def test_ticket_keys_next_to_chinese_text():
    assert [m.group(1) for m in KEY_RE.finditer("工单INC-1047和INC-1262已处理")] == ["1047", "1262"]
    assert not list(KEY_RE.finditer("XINC-1047"))


def test_refusals_in_english_and_chinese():
    assert REFUSAL_RE.search("I'm sorry, but I can't help with that.")
    assert REFUSAL_RE.search("抱歉，我无法完成这个请求。")
    assert not REFUSAL_RE.search("已为你更新周报。")
