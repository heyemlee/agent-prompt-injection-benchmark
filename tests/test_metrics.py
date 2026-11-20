from injbench.metrics import (METRICS, cluster_bootstrap, comparison_baselines, paired_diff, rate,
                              summarize, to_markdown)


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


def test_stacked_arms_are_paired_with_the_arm_they_add_to():
    arms = ["none", "permissions", "spotlight_datamark", "permissions+spotlight_datamark"]
    assert comparison_baselines("none", arms) == []
    assert comparison_baselines("permissions", arms) == ["none"]
    assert comparison_baselines("spotlight_datamark", arms) == ["none"]
    assert comparison_baselines("permissions+spotlight_datamark", arms) == ["none", "permissions"]
    # Without a `none` arm, a stacked arm is still compared with its base.
    assert comparison_baselines("permissions+modernbert",
                                ["permissions", "permissions+modernbert"]) == ["permissions"]

    records = [rec(f"t{i}", condition="attack", success=(d != "none"), defense=d)
               for i in range(3) for d in ("none", "permissions", "permissions+x")]
    s = summarize(records, n_boot=50)
    pairs = {(p["baseline"], p["defense"]) for p in s["paired"]}
    assert pairs == {("none", "permissions"), ("none", "permissions+x"),
                     ("permissions", "permissions+x")}
    safe = {(p["baseline"], p["defense"]): p["diff"] for p in s["paired"]
            if p["metric"] == "safe_completion_under_attack"}
    assert safe[("none", "permissions+x")] == 1.0 and safe[("permissions", "permissions+x")] == 0.0
    assert "| m | permissions+x | permissions |" in to_markdown(s)


def test_attack_success_rates_are_paired_by_goal_class():
    # Two templates; permissions+x stops the within-permission successes that permissions lets
    # through, and no arm has an out-of-permission success.
    records = []
    for t in ("t1", "t2"):
        for d in ("permissions", "permissions+x"):
            records.append(rec(t, condition="attack", defense=d, goal_class="within_permission",
                               attack_success=(d == "permissions"), success=(d != "permissions")))
            records.append(rec(t, condition="attack", defense=d, goal_class="out_of_permission",
                               attack_success=False))
    s = summarize(records, n_boot=50)
    diff = {p["metric"]: p for p in s["paired"]
            if (p["baseline"], p["defense"]) == ("permissions", "permissions+x")}
    assert diff["asr_within_permission"]["diff"] == -1.0
    assert diff["asr_within_permission"]["ci"] == [-1.0, -1.0]
    assert diff["asr_out_of_permission"]["diff"] == 0.0
    assert diff["asr"]["diff"] == -0.5
    assert "| ASR, within-permission goals | -1.00 [-1.00, -1.00] |" in to_markdown(s)
