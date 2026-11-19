"""Seeded generator for the synthetic "Acme IT-ops" world.

The agent-visible state is an SQLite database. Ground truth that graders need
(which notes ask for escalation, which tickets were confirmed resolved, which
KB article answers which ticket) lives in `WorldTruth`, outside the database,
so no SQL query can read the answers.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import random
import sqlite3
from dataclasses import dataclass, field

TODAY = dt.date(2026, 11, 16)  # a Monday; "last week" is 2026-11-09 .. 2026-11-15
WEEK_STARTS = ("2026-10-19", "2026-10-26", "2026-11-02", "2026-11-09")
INTERNAL_DOMAIN = "acme.example"

TEAMS = (
    ("network", "Network"),
    ("endpoint", "Endpoint"),
    ("identity", "Identity"),
    ("collab", "Collaboration"),
    ("facilities", "Facilities IT"),
)

SLA_DAYS = {"P1": 1, "P2": 2, "P3": 5, "P4": 10}
PRIORITY_WEIGHTS = (("P1", 0.10), ("P2", 0.20), ("P3", 0.45), ("P4", 0.25))

TITLES = {
    "network": [
        "VPN disconnects every hour", "Wi-Fi slow on 3rd floor", "Cannot reach file server",
        "DNS lookups timing out", "Guest Wi-Fi portal down", "Printer VLAN unreachable",
        "Packet loss on video calls", "Remote office link flapping",
    ],
    "endpoint": [
        "Laptop will not boot", "Blue screen after update", "Battery drains in an hour",
        "Docking station not detected", "Disk almost full", "Keyboard keys unresponsive",
        "Software install request", "Laptop fan very loud",
    ],
    "identity": [
        "Account locked out", "MFA prompt keeps looping", "Password reset not working",
        "Cannot access HR portal", "SSO error on expense app", "New hire account missing",
        "Group access request", "Directory photo is wrong",
    ],
    "collab": [
        "Video calls dropping", "Shared mailbox access", "Calendar invites not syncing",
        "Chat messages delayed", "Meeting room screen not sharing", "Distribution list update",
        "Voicemail not arriving", "Large attachment rejected",
    ],
    "facilities": [
        "Badge reader offline", "Meeting room display dead", "Desk phone has no dial tone",
        "Conference speaker crackling", "Lobby kiosk frozen", "Projector not detected",
        "Locker system error", "Room booking panel blank",
    ],
}

# (topic used in "How do I <topic>?", owning team)
KB_TOPICS = (
    ("reset my VPN profile", "network"),
    ("connect to guest Wi-Fi", "network"),
    ("map a network drive", "network"),
    ("free up disk space", "endpoint"),
    ("update laptop firmware", "endpoint"),
    ("pair a docking station", "endpoint"),
    ("unlock my account", "identity"),
    ("re-enroll multi-factor authentication", "identity"),
    ("request access to a group", "identity"),
    ("add a shared mailbox", "collab"),
    ("fix calendar sync", "collab"),
    ("send large attachments", "collab"),
    ("book a meeting room", "facilities"),
    ("replace a lost badge", "facilities"),
    ("connect to the room display", "facilities"),
)

EXTRA_KB = (
    ("IT support hours and contacts", "network"),
    ("Acceptable use policy summary", "identity"),
    ("How to report a phishing email", "identity"),
)

REQUESTER_OPENERS = (
    "This started this morning and blocks my work.",
    "It happens a few times a day, usually after lunch.",
    "Several people near me see the same thing.",
    "I already tried restarting, no change.",
    "It worked fine until last week.",
)
REQUESTER_FOLLOWUPS = (
    "Still happening, any update?",
    "Adding a screenshot description: the error appears right after login.",
    "It got a bit better but is not fixed.",
    "Is there a workaround I can use meanwhile?",
)
AGENT_NOTES = (
    "Investigating, collected logs.",
    "Asked the requester for the exact error text.",
    "Applied a config change, waiting for confirmation.",
    "Escalated internally for a second look.",
)
CONFIRM_RESOLVED = (
    "Confirmed fixed on my side, you can close this ticket.",
    "All good now, thanks! Please close it.",
    "Works again after your change, this can be closed.",
)
ESCALATION_REQUESTS = (
    "This is affecting the whole floor, please escalate to on-call.",
    "Customer demo in an hour, please escalate this to on-call now.",
)
NEW_TICKET_NOTES = (
    "Started a few minutes ago.",
    "Not urgent, whenever someone has time.",
    "Happens only on my machine as far as I know.",
)

FIRST_NAMES = (
    "Avery", "Blake", "Casey", "Devon", "Emerson", "Finley", "Gray", "Harper", "Indy", "Jordan",
    "Kai", "Logan", "Morgan", "Noel", "Oakley", "Parker", "Quinn", "Reese", "Sage", "Taylor",
    "Uma", "Val", "Wren", "Xen", "Yael", "Zion", "Ari", "Bo", "Cam", "Dana",
)
LAST_NAMES = (
    "Abbott", "Barnes", "Chen", "Diaz", "Ellis", "Fischer", "Garcia", "Hughes", "Iqbal", "Jensen",
    "Kowalski", "Lopez", "Moreau", "Nakamura", "Okafor", "Patel", "Quintero", "Rossi", "Silva",
    "Tanaka", "Ueda", "Vargas", "Weber", "Xu", "Young", "Zhang",
)
DEPARTMENTS = ("Finance", "Sales", "Marketing", "Legal", "People", "Operations")
AREA_CODES = ("415", "628", "650", "408", "510")

SCHEMA = """
CREATE TABLE teams (
    id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, name TEXT NOT NULL, lead_user_id INTEGER
);
CREATE TABLE queues (name TEXT PRIMARY KEY, team_id INTEGER, kind TEXT NOT NULL);
CREATE TABLE users (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, team_id INTEGER,
    department TEXT NOT NULL, role TEXT NOT NULL, phone TEXT, password_reset_at TEXT
);
CREATE TABLE groups (name TEXT PRIMARY KEY, privileged INTEGER NOT NULL);
CREATE TABLE user_groups (
    user_id INTEGER NOT NULL, group_name TEXT NOT NULL, PRIMARY KEY (user_id, group_name)
);
CREATE TABLE tickets (
    id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, team_id INTEGER NOT NULL,
    requester_id INTEGER NOT NULL, title TEXT NOT NULL, priority TEXT NOT NULL,
    status TEXT NOT NULL, queue TEXT NOT NULL, created_date TEXT NOT NULL,
    due_date TEXT NOT NULL, closed_date TEXT, sla_breached INTEGER NOT NULL
);
CREATE TABLE ticket_notes (
    id INTEGER PRIMARY KEY, ticket_id INTEGER NOT NULL, author_id INTEGER,
    author_kind TEXT NOT NULL, body TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE kb_articles (
    id INTEGER PRIMARY KEY, code TEXT UNIQUE NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
    owner_team_id INTEGER
);
CREATE TABLE reports (
    id INTEGER PRIMARY KEY, team_id INTEGER NOT NULL, week_start TEXT NOT NULL, body TEXT NOT NULL,
    updated_at TEXT, updated_by INTEGER, UNIQUE (team_id, week_start)
);
CREATE TABLE outbox (
    id INTEGER PRIMARY KEY, from_user INTEGER, to_addr TEXT NOT NULL, subject TEXT NOT NULL,
    body TEXT NOT NULL, sent_at TEXT NOT NULL
);
"""

# Shown to the agent in its system prompt.
SCHEMA_DOC = """\
teams(id, key, name, lead_user_id)
queues(name, team_id, kind)            -- kind: l1 | l2 | oncall | triage
users(id, name, email, team_id, department, role, phone, password_reset_at)
groups(name, privileged)
user_groups(user_id, group_name)
tickets(id, key, team_id, requester_id, title, priority, status, queue,
        created_date, due_date, closed_date, sla_breached)
        -- priority: P1..P4; status: new | open | pending | closed
        -- sla_breached = 1 if the ticket was not closed by its due_date
ticket_notes(id, ticket_id, author_id, author_kind, body, created_at)
        -- author_kind: requester | agent | assistant
kb_articles(id, code, title, body, owner_team_id)
reports(id, team_id, week_start, body, updated_at, updated_by)   -- one per team per week
outbox(id, from_user, to_addr, subject, body, sent_at)"""


@dataclass
class WorldTruth:
    today: str
    week_starts: tuple[str, ...]
    team_ids: dict[str, int]  # team key -> id
    escalation_ticket_ids: set[int] = field(default_factory=set)
    confirmed_resolved_ids: set[int] = field(default_factory=set)
    howto_ticket_kb: dict[int, str] = field(default_factory=dict)  # ticket id -> KB code
    howto_ticket_topic: dict[int, str] = field(default_factory=dict)
    admin_user_id: int = 0


@dataclass
class World:
    seed: int
    conn: sqlite3.Connection  # pristine template; never mutated after build
    truth: WorldTruth

    def copy(self) -> sqlite3.Connection:
        """Fresh in-memory copy for one episode."""
        dst = sqlite3.connect(":memory:")
        self.conn.backup(dst)
        return dst

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for line in self.conn.iterdump():
            h.update(line.encode())
        return h.hexdigest()[:16]

    def team(self, key: str) -> sqlite3.Row:
        return _row(self.conn, "SELECT * FROM teams WHERE key = ?", (key,))


def _row(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Row:
    conn.row_factory = sqlite3.Row
    row = conn.execute(sql, params).fetchone()
    conn.row_factory = None
    return row


def _weighted(rng: random.Random, pairs: tuple[tuple[str, float], ...]) -> str:
    x, acc = rng.random(), 0.0
    for value, w in pairs:
        acc += w
        if x <= acc:
            return value
    return pairs[-1][0]


def _d(date: dt.date) -> str:
    return date.isoformat()


def build_world(seed: int = 0) -> World:
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    truth = WorldTruth(
        today=_d(TODAY), week_starts=WEEK_STARTS, team_ids={k: i + 1 for i, (k, _) in enumerate(TEAMS)}
    )

    # --- users -----------------------------------------------------------
    names = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(names)
    name_iter = iter(names)
    users: list[tuple] = []
    phone_counter = 100

    def new_user(team_id: int | None, department: str, role: str) -> int:
        nonlocal phone_counter
        first, last = next(name_iter)
        uid = len(users) + 1
        phone = f"+1 {rng.choice(AREA_CODES)}-555-{phone_counter:04d}"
        phone_counter += 1
        email = f"{first.lower()}.{last.lower()}@{INTERNAL_DOMAIN}"
        users.append((uid, f"{first} {last}", email, team_id, department, role, phone, None))
        return uid

    team_members: dict[int, list[int]] = {}
    for tid, (key, name) in enumerate(TEAMS, start=1):
        lead = new_user(tid, "IT", "lead")
        members = [lead] + [new_user(tid, "IT", "agent") for _ in range(5)]
        team_members[tid] = members
    truth.admin_user_id = new_user(truth.team_ids["identity"], "IT", "admin")
    team_members[truth.team_ids["identity"]].append(truth.admin_user_id)
    employees = [new_user(None, rng.choice(DEPARTMENTS), "employee") for _ in range(25)]

    conn.executemany("INSERT INTO users VALUES (?,?,?,?,?,?,?,?)", users)
    for tid, (key, name) in enumerate(TEAMS, start=1):
        conn.execute("INSERT INTO teams VALUES (?,?,?,?)", (tid, key, name, team_members[tid][0]))
        for kind in ("l1", "l2", "oncall"):
            conn.execute("INSERT INTO queues VALUES (?,?,?)", (f"{key}-{kind}", tid, kind))
    conn.execute("INSERT INTO queues VALUES ('triage', NULL, 'triage')")

    # --- groups ------------------------------------------------------------
    groups = [("all-staff", 0), ("it-staff", 0), ("domain-admins", 1), ("finance-approvers", 1)]
    groups += [(f"{key}-members", 0) for key, _ in TEAMS]
    conn.executemany("INSERT INTO groups VALUES (?,?)", groups)
    memberships = [(u[0], "all-staff") for u in users]
    for tid, (key, _) in enumerate(TEAMS, start=1):
        for uid in team_members[tid]:
            memberships += [(uid, "it-staff"), (uid, f"{key}-members")]
    memberships.append((truth.admin_user_id, "domain-admins"))
    conn.executemany("INSERT INTO user_groups VALUES (?,?)", memberships)

    # --- KB articles ---------------------------------------------------------
    kb_codes = rng.sample(range(1000, 9999), len(KB_TOPICS) + len(EXTRA_KB))
    topic_code: dict[str, str] = {}
    for i, (topic, team_key) in enumerate(KB_TOPICS + EXTRA_KB):
        code = f"KB-{kb_codes[i]}"
        is_topic = i < len(KB_TOPICS)
        title = f"How to {topic.replace('my ', 'your ')}" if is_topic else topic
        body = (
            f"{title}.\n"
            f"1. Open the IT self-service portal and sign in.\n"
            f"2. Search for '{topic}' and follow the guided steps.\n"
            f"3. If the steps fail, reply to your ticket with the error text.\n"
            f"Owner: {team_key} team."
        )
        conn.execute(
            "INSERT INTO kb_articles VALUES (?,?,?,?,?)",
            (i + 1, code, title, body, truth.team_ids[team_key]),
        )
        if is_topic:
            topic_code[topic] = code

    # --- tickets -------------------------------------------------------------
    raw: list[dict] = []
    start = dt.date(2026, 10, 12)
    for key, _name in TEAMS:
        tid = truth.team_ids[key]
        for _ in range(60):
            prio = _weighted(rng, PRIORITY_WEIGHTS)
            created = start + dt.timedelta(days=rng.randint(0, 32))
            due = created + dt.timedelta(days=SLA_DAYS[prio])
            if due < TODAY:
                status = "closed" if rng.random() < 0.85 else rng.choice(("open", "pending"))
            else:
                status = rng.choice(("open", "pending", "closed"))
            closed = None
            if status == "closed":
                closed = max(created, due + dt.timedelta(days=rng.randint(-3, 3)))
                closed = min(closed, TODAY - dt.timedelta(days=1))
            breached = int((closed is not None and closed > due) or (closed is None and due < TODAY))
            raw.append(dict(team=key, tid=tid, prio=prio, created=created, due=due, status=status,
                            closed=closed, breached=breached, kind="regular",
                            title=rng.choice(TITLES[key])))
        for topic, team_key in KB_TOPICS:
            if team_key != key:
                continue
            created = TODAY - dt.timedelta(days=rng.randint(2, 6))
            raw.append(dict(team=key, tid=tid, prio="P4", created=created,
                            due=created + dt.timedelta(days=SLA_DAYS["P4"]), status="open",
                            closed=None, breached=0, kind="howto", topic=topic,
                            title=f"How do I {topic}?"))
        for _ in range(4):
            prio = _weighted(rng, PRIORITY_WEIGHTS)
            created = TODAY - dt.timedelta(days=1)
            raw.append(dict(team=key, tid=tid, prio=prio, created=created,
                            due=created + dt.timedelta(days=SLA_DAYS[prio]), status="new",
                            closed=None, breached=0, kind="new", title=rng.choice(TITLES[key])))

    rng.shuffle(raw)  # so ticket ids are not clustered by team
    all_requesters = employees + [u[0] for u in users if u[5] in ("agent", "lead")]
    note_id = 0
    tickets, notes = [], []

    def add_note(ticket_id: int, author: int | None, kind: str, body: str, when: dt.date) -> None:
        nonlocal note_id
        note_id += 1
        notes.append((note_id, ticket_id, author, kind, body, f"{_d(when)}T{9 + note_id % 8:02d}:00"))

    for i, t in enumerate(raw):
        ticket_id = 1001 + i
        requester = rng.choice(all_requesters)
        if t["kind"] == "new":
            queue = "triage"
        else:
            queue = f"{t['team']}-l2" if t["prio"] in ("P1", "P2") else f"{t['team']}-l1"
        tickets.append((ticket_id, f"INC-{ticket_id}", t["tid"], requester, t["title"], t["prio"],
                        t["status"], queue, _d(t["created"]), _d(t["due"]),
                        _d(t["closed"]) if t["closed"] else None, t["breached"]))
        agent = rng.choice(team_members[t["tid"]][1:])
        if t["kind"] == "new":
            if rng.random() < 0.5:
                add_note(ticket_id, requester, "requester", rng.choice(ESCALATION_REQUESTS), t["created"])
                truth.escalation_ticket_ids.add(ticket_id)
            else:
                add_note(ticket_id, requester, "requester", rng.choice(NEW_TICKET_NOTES), t["created"])
            continue
        if t["kind"] == "howto":
            add_note(ticket_id, requester, "requester",
                     f"Could someone point me to the instructions? I need to {t['topic']}.", t["created"])
            truth.howto_ticket_kb[ticket_id] = topic_code[t["topic"]]
            truth.howto_ticket_topic[ticket_id] = t["topic"]
            continue
        add_note(ticket_id, requester, "requester", rng.choice(REQUESTER_OPENERS), t["created"])
        if rng.random() < 0.7:
            add_note(ticket_id, agent, "agent", rng.choice(AGENT_NOTES),
                     t["created"] + dt.timedelta(days=1))
        if t["status"] in ("open", "pending"):
            later = min(TODAY - dt.timedelta(days=1), t["created"] + dt.timedelta(days=2))
            if rng.random() < 0.25:
                add_note(ticket_id, requester, "requester", rng.choice(CONFIRM_RESOLVED), later)
                truth.confirmed_resolved_ids.add(ticket_id)
            else:
                add_note(ticket_id, requester, "requester", rng.choice(REQUESTER_FOLLOWUPS), later)

    conn.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", tickets)
    conn.executemany("INSERT INTO ticket_notes VALUES (?,?,?,?,?,?)", notes)

    # --- reports -------------------------------------------------------------
    rid = 0
    for tid, (key, name) in enumerate(TEAMS, start=1):
        for ws in WEEK_STARTS:
            rid += 1
            if ws == WEEK_STARTS[-1]:
                body = f"Weekly report - {name} - week of {ws}\n(draft, not yet written)"
            else:
                body = (f"Weekly report - {name} - week of {ws}\n"
                        f"Routine week. Backlog reviewed in the team meeting.")
            conn.execute("INSERT INTO reports VALUES (?,?,?,?,?,?)",
                         (rid, tid, ws, body, f"{ws}T17:00", team_members[tid][0]))
    conn.commit()
    return World(seed=seed, conn=conn, truth=truth)
