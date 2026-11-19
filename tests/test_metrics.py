from injbench.metrics import METRICS, cluster_bootstrap, paired_diff, rate, summarize, to_markdown


def rec(template, condition="benign", success=True, defense="none", **kw):
    base = {"status": "ok", "template": template, "condition": condition, "task_success": success,
            "safe_completion": success, "defense": defense, "model": "m", "goal_class": None,
            "attack_success": None, "n_realized": 0, "n_attempted": 0, "out_of_scope_reads": 0,
            "wrongful_refusal": False, "exposed": None}
    base.update(kw)
    return base


def test_rate_and_bootstrap_are_deterministic():
    records = [rec(f"t{i % 4}", success=(i % 3 != 0)) for i in range(40)]
    filt, val = METRICS["benign_completion"]
    v, n = rate(records, filt, val)
    assert n == 40 and 0 < v < 1
    fn = lambda s: rate(s, filt, val)[0]  # noqa: E731
    ci1 = cluster_bootstrap(records, fn, n_boot=200, seed=1)
    assert ci1 == cluster_bootstrap(records, fn, n_boot=200, seed=1)
    assert ci1[0] <= v <= ci1[1]
    assert cluster_bootstrap([rec("only")], fn) is None  # one cluster: no CI


def test_paired_diff():
    a = [rec(f"t{i}", success=False) for i in range(5)]
    b = [rec(f"t{i}", success=True, defense="permissions") for i in range(5)]
    filt, val = METRICS["benign_completion"]
    res = paired_diff(a, b, filt, val, n_boot=100)
    assert res["diff"] == 1.0 and res["ci"] == [1.0, 1.0]


def test_summary_and_markdown():
    records = [rec("t1"), rec("t2"),
               rec("t1", condition="attack", attack_success=True, goal_class="within_permission",
                   exposed=True, success=False),
               rec("t1", defense="permissions"), {"status": "error", "episode_id": "e"}]
    s = summarize(records, n_boot=50)
    assert {(a["model"], a["defense"]) for a in s["arms"]} == {("m", "none"), ("m", "permissions")}
    none = next(a for a in s["arms"] if a["defense"] == "none")
    assert none["metrics"]["asr_within_permission"]["value"] == 1.0
    assert s["status_counts"] == {"ok": 4, "error": 1}
    assert s["notes"]
    md = to_markdown(s)
    assert "Safe completion under attack" in md
