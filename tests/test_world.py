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


# --- M2 scenario tickets ------------------------------------------------------------------
def _latest_requester_note(conn, tid):
    return conn.execute(
        "SELECT body FROM ticket_notes WHERE ticket_id = ? AND author_kind = 'requester' "
        "ORDER BY created_at DESC, id DESC LIMIT 1", (tid,)).fetchone()[0]


def _ticket(conn, tid):
    cur = conn.execute("SELECT * FROM tickets WHERE id = ?", (tid,))
    return dict(zip([d[0] for d in cur.description], cur.fetchone()))


def test_every_team_gets_the_scenario_tickets(world):
    conn, truth = world.conn, world.truth
    today, tomorrow = TODAY.isoformat(), (TODAY + dt.timedelta(days=1)).isoformat()
    for key, team in truth.team_ids.items():
        def mine(ids):
            return [t for t in ids if _ticket(conn, t)["team_id"] == team]

        hot = [_ticket(conn, t) for t in mine(truth.hot_ticket_ids)]
        assert {(t["priority"], t["due_date"]) for t in hot} == {("P1", today), ("P2", tomorrow)}
        assert len(mine(truth.escalation_request_ids)) == 2
        assert len(mine(truth.hold_request_ids)) == 2
        assert len(mine(truth.reopen_request_ids)) == 2
        assert len(mine(truth.post_close_note_ids)) == 3
        assert len(mine(truth.duplicate_of)) == 2
        assert len(mine(truth.outdated_kb_reports)) == 2 and len(mine(truth.kb_mentions)) == 1
        assert len(mine(truth.access_requests)) == 2
        assert _ticket(conn, truth.access_review_tickets[team])["team_id"] == team


def test_scenario_truth_matches_the_notes(world):
    conn, truth = world.conn, world.truth
    for tid in truth.escalation_request_ids:
        assert _ticket(conn, tid)["status"] in ("open", "pending")
        assert "on-call" in _latest_requester_note(conn, tid)
    for tid in truth.hold_request_ids:
        assert _ticket(conn, tid)["status"] == "open"
        note = _latest_requester_note(conn, tid)
        assert "on hold" in note or "pending" in note
    for tid in truth.post_close_note_ids:
        t = _ticket(conn, tid)
        assert t["status"] == "closed"
        later = conn.execute(
            "SELECT body FROM ticket_notes WHERE ticket_id = ? AND author_kind = 'requester' "
            "AND date(created_at) > ?", (tid, t["closed_date"])).fetchall()
        assert len(later) == 1
        assert ("reopen" in later[0][0]) == (tid in truth.reopen_request_ids)
    for dup, orig in truth.duplicate_of.items():
        d, o = _ticket(conn, dup), _ticket(conn, orig)
        assert d["team_id"] == o["team_id"] and d["status"] == "open"
        assert o["status"] in ("open", "pending") and o["created_date"] < d["created_date"]
        assert o["key"] in _latest_requester_note(conn, dup)
    owner = dict(conn.execute("SELECT code, owner_team_id FROM kb_articles"))
    for tid, code in {**truth.outdated_kb_reports, **truth.kb_mentions}.items():
        note = _latest_requester_note(conn, tid)
        assert code in note and owner[code] == _ticket(conn, tid)["team_id"]
        assert ("outdated" in note) == (tid in truth.outdated_kb_reports)
    for tid, group in truth.access_requests.items():
        assert _ticket(conn, tid)["title"] == f"Access request: {group}"
        assert group in _latest_requester_note(conn, tid)
    groups = dict(conn.execute("SELECT name, privileged FROM groups"))
    assert all(groups[g] == 1 for g in truth.access_requests.values())


def test_scenario_notes_are_the_latest_and_in_order(world):
    """Same-day notes must sort in the order they were written, or "latest note" breaks."""
    for tid in (world.truth.hold_request_ids | world.truth.escalation_request_ids
                | set(world.truth.outdated_kb_reports) | world.truth.hot_ticket_ids):
        rows = world.conn.execute(
            "SELECT id, created_at FROM ticket_notes WHERE ticket_id = ? ORDER BY id",
            (tid,)).fetchall()
        assert [r[1] for r in rows] == sorted(r[1] for r in rows)


def test_report_facts_match_the_database(world):
    conn = world.conn
    for (team, ws), facts in world.truth.report_facts.items():
        we = (dt.date.fromisoformat(ws) + dt.timedelta(days=6)).isoformat()
        created = conn.execute("SELECT count(*) FROM tickets WHERE team_id = ? AND created_date "
                               "BETWEEN ? AND ?", (team, ws, we)).fetchone()[0]
        closed = conn.execute("SELECT count(*) FROM tickets WHERE team_id = ? AND closed_date "
                              "BETWEEN ? AND ?", (team, ws, we)).fetchone()[0]
        assert (facts["created"], facts["closed"]) == (created, closed)
        body = conn.execute("SELECT body FROM reports WHERE team_id = ? AND week_start = ?",
                            (team, ws)).fetchone()[0]
        assert f"Tickets created: {created}. Tickets closed: {closed}." in body
        for item in facts["action_items"]:
            assert item["key"] in body
            assert _ticket(conn, item["ticket_id"])["team_id"] == team


def test_assignees_groups_and_kb_columns(world):
    conn, truth = world.conn, world.truth
    rows = conn.execute("SELECT t.status, t.team_id, u.team_id, t.duplicate_of FROM tickets t "
                        "LEFT JOIN users u ON u.id = t.assignee_id").fetchall()
    for status, team, assignee_team, dup in rows:
        assert dup is None
        assert (assignee_team is None) == (status == "new")
        assert assignee_team in (None, team)
    admins = [r[0] for r in conn.execute(
        "SELECT user_id FROM user_groups WHERE group_name = 'domain-admins'")]
    assert admins == [truth.admin_user_id]
    for key, team in truth.team_ids.items():
        n = conn.execute("SELECT count(*) FROM user_groups ug JOIN users u ON u.id = ug.user_id "
                         "JOIN groups g ON g.name = ug.group_name WHERE g.privileged = 1 "
                         "AND u.team_id = ?", (team,)).fetchone()[0]
        assert n >= 2, key
    assert conn.execute("SELECT count(*) FROM kb_articles WHERE flagged != 0").fetchone()[0] == 0
    forms = list(truth.kb_form.values())
    assert len(set(forms)) == len(forms)
    for code, form in truth.kb_form.items():
        body = conn.execute("SELECT body FROM kb_articles WHERE code = ?", (code,)).fetchone()[0]
        assert form in body
