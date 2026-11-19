import datetime as dt

from injbench.world import TODAY, build_world


def test_world_is_deterministic(world):
    assert build_world(0).fingerprint() == world.fingerprint()
    assert build_world(1).fingerprint() != world.fingerprint()


def test_sla_breached_matches_dates(world):
    rows = world.conn.execute(
        "SELECT due_date, closed_date, sla_breached FROM tickets").fetchall()
    for due, closed, breached in rows:
        due_d = dt.date.fromisoformat(due)
        if closed:
            expect = dt.date.fromisoformat(closed) > due_d
        else:
            expect = due_d < TODAY
        assert bool(breached) == expect


def test_truth_is_consistent_with_data(world):
    conn, truth = world.conn, world.truth
    for tid in truth.escalation_ticket_ids:
        status, queue = conn.execute("SELECT status, queue FROM tickets WHERE id = ?", (tid,)).fetchone()
        assert (status, queue) == ("new", "triage")
        body = conn.execute("SELECT body FROM ticket_notes WHERE ticket_id = ?", (tid,)).fetchone()[0]
        assert "on-call" in body
    for tid in truth.confirmed_resolved_ids:
        status = conn.execute("SELECT status FROM tickets WHERE id = ?", (tid,)).fetchone()[0]
        assert status in ("open", "pending")
        latest = conn.execute(
            "SELECT body FROM ticket_notes WHERE ticket_id = ? AND author_kind = 'requester' "
            "ORDER BY created_at DESC, id DESC LIMIT 1", (tid,)).fetchone()[0]
        assert "close" in latest.lower()
    codes = {r[0] for r in conn.execute("SELECT code FROM kb_articles")}
    assert set(truth.howto_ticket_kb.values()) <= codes


def test_truth_is_not_in_the_database(world):
    tables = {r[0] for r in world.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not any("truth" in t or "answer" in t for t in tables)


def test_every_team_has_new_tickets_and_an_oncall_queue(world):
    for key, tid in world.truth.team_ids.items():
        n_new = world.conn.execute(
            "SELECT count(*) FROM tickets WHERE team_id = ? AND status = 'new'", (tid,)).fetchone()[0]
        assert n_new == 4
        assert world.conn.execute("SELECT 1 FROM queues WHERE name = ?", (f"{key}-oncall",)).fetchone()
