"""
GitHub Profile Analyzer (Flask) — Drop-in Replacement

What it does:
- Accepts a GitHub username
- Fetches public GitHub stats via GitHub APIs (REST + GraphQL when available)
- Computes an "Open Source Contribution Coefficient" (0–100)
- Generates badge metadata (Shields URLs)
- Suggests suitable tracks/roles & company types based on languages/topics/activity
- Adds:
  - Longest streak (365d) + current streak (365d)
  - Longest streak over a longer window (default 3 years) by stitching 1-year calendars (GraphQL limit-safe)
  - All-time commit contributions (since account creation) by summing 1-year chunks (GraphQL limit-safe)

Setup:
  pip install flask requests

Run:
  export GITHUB_TOKEN="github_pat_..."   # recommended (higher rate limits + GraphQL)
  python app.py
  open http://localhost:5000

Endpoints:
  GET  /                       -> renders templates/index.html (if present), else fallback page
  GET  /api/analyze?username=   -> returns JSON analysis
  POST /api/analyze             -> accepts form-data or JSON { "username": "..." }

Important GitHub GraphQL constraint:
- contributionsCollection(from,to) cannot exceed 1 year.
  This file strictly respects that by chunking requests into <= 1-year windows.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, render_template, request

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_API_BASE = "https://api.github.com"
GITHUB_GRAPHQL = "https://api.github.com/graphql"
API_VERSION = os.getenv("GITHUB_API_VERSION", "2022-11-28")

# Simple in-memory TTL cache
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "600"))

# Long-window streak (in years) for "longest streak over window"
STREAK_WINDOW_YEARS = int(os.getenv("STREAK_WINDOW_YEARS", "3"))

_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

# Username validation (GitHub allows alnum and hyphen; max length 39)
USERNAME_RE = re.compile(r"^[A-Za-z0-9-]{1,39}$")


# -----------------------------
# HTTP helpers
# -----------------------------
class GitHubAPIError(RuntimeError):
    pass


def _headers() -> Dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-profile-analyzer-flask",
        "X-GitHub-Api-Version": API_VERSION,
    }
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _request_json(method: str, url: str, *, params: Optional[dict] = None, json: Optional[dict] = None, timeout: int = 20) -> Any:
    """
    Request wrapper with small retry/backoff for transient failures and secondary rate limits.
    """
    backoff = 1.0
    last_exc: Optional[Exception] = None
    for _ in range(4):
        try:
            resp = requests.request(method, url, headers=_headers(), params=params, json=json, timeout=timeout)

            if resp.status_code == 403 and "rate limit" in (resp.text or "").lower():
                reset = resp.headers.get("X-RateLimit-Reset")
                if reset and reset.isdigit():
                    wait = max(0, int(reset) - int(time.time()))
                    wait = min(wait + 1, 20)
                else:
                    wait = min(5, backoff)
                time.sleep(wait)
                backoff *= 2
                continue

            if resp.status_code >= 400:
                raise GitHubAPIError(f"GitHub REST error {resp.status_code}: {resp.text[:600]}")

            return resp.json()
        except (requests.RequestException, GitHubAPIError) as e:
            last_exc = e
            time.sleep(min(5, backoff))
            backoff *= 2

    raise GitHubAPIError(f"GitHub REST request failed after retries: {last_exc}")


def _graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    if not GITHUB_TOKEN:
        raise GitHubAPIError("GraphQL requires GITHUB_TOKEN. Set env var GITHUB_TOKEN.")
    payload = {"query": query, "variables": variables}
    resp = requests.post(GITHUB_GRAPHQL, headers=_headers(), json=payload, timeout=25)
    if resp.status_code >= 400:
        raise GitHubAPIError(f"GitHub GraphQL error {resp.status_code}: {resp.text[:600]}")
    data = resp.json()
    if "errors" in data and data["errors"]:
        # Show only first few errors to keep responses short
        raise GitHubAPIError(f"GitHub GraphQL errors: {data['errors'][:3]}")
    return data.get("data", {})


# -----------------------------
# Utility helpers
# -----------------------------
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _days_ago(days: int) -> dt.datetime:
    return _now_utc() - dt.timedelta(days=int(days))


def _dateparse(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_log(x: float, max_x: float) -> float:
    x = max(0.0, float(x))
    max_x = max(1.0, float(max_x))
    return _clamp(math.log1p(x) / math.log1p(max_x), 0.0, 1.0)


def _entropy(counts: Dict[str, float]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    npos = 0
    for v in counts.values():
        if v <= 0:
            continue
        npos += 1
        p = v / total
        ent -= p * math.log(p)
    if npos <= 1:
        return 0.0
    return float(ent / math.log(npos))


def _add_one_year(d: dt.datetime) -> dt.datetime:
    """
    Adds 1 calendar year while avoiding invalid dates (Feb 29).
    Keeps timezone info.
    """
    try:
        return d.replace(year=d.year + 1)
    except ValueError:
        # e.g., Feb 29 -> Feb 28 next year
        return d.replace(month=2, day=28, year=d.year + 1)


def _year_windows(start: dt.datetime, end: dt.datetime) -> List[Tuple[dt.datetime, dt.datetime]]:
    """
    Produces contiguous windows where each window spans <= 1 calendar year.
    This respects GitHub GraphQL contributionsCollection constraint.
    """
    windows: List[Tuple[dt.datetime, dt.datetime]] = []
    cur = start
    while cur < end:
        nxt = _add_one_year(cur)
        win_end = end if nxt > end else nxt
        windows.append((cur, win_end))
        cur = win_end
        # Safety break
        if len(windows) > 50:
            break
    return windows


def _flatten_calendar(contribution_calendar: Dict[str, Any]) -> Dict[dt.date, int]:
    """
    Returns dict {date: contributionCount} from GraphQL contributionCalendar.
    """
    out: Dict[dt.date, int] = {}
    for w in (contribution_calendar.get("weeks") or []):
        for d in (w.get("contributionDays") or []):
            ds = d.get("date")
            if not ds:
                continue
            try:
                dd = dt.date.fromisoformat(ds)
            except Exception:
                continue
            out[dd] = int(d.get("contributionCount") or 0)
    return out


def _compute_streak_from_counts(counts: Dict[dt.date, int], start_date: dt.date, end_date: dt.date) -> Dict[str, int]:
    """
    Compute max streak and current streak for contributions > 0, inclusive of end_date.
    """
    max_streak = 0
    cur_streak = 0
    d = start_date
    one = dt.timedelta(days=1)
    while d <= end_date:
        if counts.get(d, 0) > 0:
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
        else:
            cur_streak = 0
        d += one
    return {"max_streak_days": int(max_streak), "current_streak_days": int(cur_streak)}


def _badge(label: str, message: str, color: str) -> Dict[str, str]:
    safe_label = re.sub(r"[^A-Za-z0-9 _-]", "", label).strip()
    safe_msg = re.sub(r"[^A-Za-z0-9 _\.\-+%]", "", message).strip()
    safe_color = re.sub(r"[^A-Za-z0-9_-]", "", color).strip().lower() or "blue"
    url = f"https://img.shields.io/badge/{requests.utils.quote(safe_label)}-{requests.utils.quote(safe_msg)}-{safe_color}"
    return {"label": label, "message": message, "color": safe_color, "url": url}


# -----------------------------
# GraphQL queries (1-year safe)
# -----------------------------
CONTRIB_1Y_QUERY = """
query($login:String!, $from:DateTime!, $to:DateTime!) {
  user(login:$login) {
    login
    name
    bio
    company
    location
    websiteUrl
    url
    createdAt
    followers { totalCount }
    following { totalCount }
    repositories(privacy:PUBLIC) { totalCount }
    repositoriesContributedTo(privacy:PUBLIC) { totalCount }

    contributionsCollection(from:$from, to:$to) {
      totalCommitContributions
      totalIssueContributions
      totalPullRequestContributions
      totalPullRequestReviewContributions
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
          }
        }
      }
    }
  }
}
"""

COMMITS_1Y_QUERY = """
query($login:String!, $from:DateTime!, $to:DateTime!) {
  user(login:$login) {
    contributionsCollection(from:$from, to:$to) {
      totalCommitContributions
    }
  }
}
"""

CALENDAR_1Y_QUERY = """
query($login:String!, $from:DateTime!, $to:DateTime!) {
  user(login:$login) {
    contributionsCollection(from:$from, to:$to) {
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
          }
        }
      }
    }
  }
}
"""

REPOS_QUERY = """
query($login:String!, $after:String) {
  user(login:$login) {
    repositories(
      first:100,
      after:$after,
      privacy:PUBLIC,
      ownerAffiliations:[OWNER, COLLABORATOR, ORGANIZATION_MEMBER],
      orderBy:{field:STARGAZERS, direction:DESC}
    ) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        name
        description
        url
        isFork
        isArchived
        pushedAt
        createdAt
        updatedAt
        forkCount
        stargazerCount
        watchers { totalCount }
        issues(states:OPEN) { totalCount }
        pullRequests(states:OPEN) { totalCount }
        defaultBranchRef { name }

        primaryLanguage { name }
        languages(first:10, orderBy:{field:SIZE, direction:DESC}) {
          edges { size node { name } }
        }
        repositoryTopics(first:10) {
          nodes { topic { name } }
        }
      }
    }
  }
}
"""


# -----------------------------
# GraphQL chunked fetchers
# -----------------------------
def _fetch_all_time_commit_contributions(login: str, created_at: dt.datetime, now: dt.datetime) -> int:
    """
    Sum totalCommitContributions across <=1-year windows from account creation to now.
    """
    total = 0
    for frm, to in _year_windows(created_at, now):
        data = _graphql(COMMITS_1Y_QUERY, {"login": login, "from": frm.isoformat(), "to": to.isoformat()})
        commits = (
            ((data.get("user") or {}).get("contributionsCollection") or {}).get("totalCommitContributions") or 0
        )
        total += int(commits)
    return int(total)


def _fetch_stitched_calendar(login: str, start: dt.datetime, end: dt.datetime) -> Dict[dt.date, int]:
    """
    Stitch daily contribution counts across <=1-year windows into a single date->count dict.
    """
    stitched: Dict[dt.date, int] = {}
    for frm, to in _year_windows(start, end):
        data = _graphql(CALENDAR_1Y_QUERY, {"login": login, "from": frm.isoformat(), "to": to.isoformat()})
        cal = (
            (((data.get("user") or {}).get("contributionsCollection") or {}).get("contributionCalendar")) or {}
        )
        counts = _flatten_calendar(cal)
        # Merge (later windows overwrite same dates, should be identical)
        stitched.update(counts)

    # Filter strictly to range (inclusive)
    start_d = start.date()
    end_d = end.date()
    stitched = {d: c for d, c in stitched.items() if start_d <= d <= end_d}
    return stitched


# -----------------------------
# Fetchers (GraphQL-first)
# -----------------------------
def fetch_with_graphql(username: str) -> Dict[str, Any]:
    to = _now_utc()
    from_ = to - dt.timedelta(days=365)

    data = _graphql(CONTRIB_1Y_QUERY, {"login": username, "from": from_.isoformat(), "to": to.isoformat()})
    user = (data or {}).get("user")
    if not user:
        raise GitHubAPIError("User not found (GraphQL).")

    # Repo pagination (cap)
    repos: List[Dict[str, Any]] = []
    after = None
    while True:
        chunk = _graphql(REPOS_QUERY, {"login": username, "after": after})
        u = (chunk or {}).get("user") or {}
        conn = (u.get("repositories") or {})
        repos.extend(conn.get("nodes") or [])
        page = conn.get("pageInfo") or {}
        if not page.get("hasNextPage"):
            break
        after = page.get("endCursor")
        if len(repos) >= 300:
            break

    user["_repos_nodes"] = repos
    user["_analysis_window"] = {"from": from_.date().isoformat(), "to": to.date().isoformat()}

    # All-time commit contributions (since account creation) — chunked (<=1y windows).
    created_at = _dateparse(user.get("createdAt"))
    if created_at:
        user["_all_time_commit_contributions"] = _fetch_all_time_commit_contributions(username, created_at, to)
    else:
        user["_all_time_commit_contributions"] = None

    # Long-window streak (default last N years) — stitch 1y calendars.
    long_from = to - dt.timedelta(days=365 * max(1, STREAK_WINDOW_YEARS))
    try:
        stitched = _fetch_stitched_calendar(username, long_from, to)
        user["_streak_window_years"] = int(STREAK_WINDOW_YEARS)
        user["_streak_counts"] = stitched
        user["_streak_window"] = {"from": long_from.date().isoformat(), "to": to.date().isoformat()}
    except GitHubAPIError:
        # If something goes wrong here, don't fail the whole request.
        user["_streak_window_years"] = int(STREAK_WINDOW_YEARS)
        user["_streak_counts"] = {}
        user["_streak_window"] = {"from": long_from.date().isoformat(), "to": to.date().isoformat()}

    user["_data_mode"] = "graphql"
    return user


def fetch_with_rest(username: str) -> Dict[str, Any]:
    user = _request_json("GET", f"{GITHUB_API_BASE}/users/{username}")

    # Fetch repos (public) with pagination
    repos: List[Dict[str, Any]] = []
    page = 1
    while page <= 3:  # cap to 300 repos
        page_repos = _request_json(
            "GET",
            f"{GITHUB_API_BASE}/users/{username}/repos",
            params={"per_page": 100, "page": page, "type": "owner", "sort": "updated"},
        )
        if not isinstance(page_repos, list) or not page_repos:
            break
        repos.extend(page_repos)
        if len(page_repos) < 100:
            break
        page += 1

    # Events (public) – very limited window
    events = _request_json(
        "GET",
        f"{GITHUB_API_BASE}/users/{username}/events/public",
        params={"per_page": 100, "page": 1},
    )

    user["_repos_nodes"] = repos
    user["_events"] = events if isinstance(events, list) else []
    user["_analysis_window"] = {"from": (_days_ago(90).date().isoformat()), "to": _now_utc().date().isoformat()}
    user["_data_mode"] = "rest"
    return user


# -----------------------------
# Recommendations
# -----------------------------
def _language_recommendations(lang_weights: Dict[str, float], topic_counts: Dict[str, int]) -> Dict[str, Any]:
    top_langs = sorted(lang_weights.items(), key=lambda kv: kv[1], reverse=True)[:5]
    langs = [l for l, _ in top_langs if l]
    topics = sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    top_topics = [t for t, _ in topics]

    lang_map = {
        "Python": ["Backend (FastAPI/Django)", "Data Science/ML", "Automation/Scripting"],
        "JavaScript": ["Frontend (React/Vue)", "Node.js Backend", "Full-stack Web"],
        "TypeScript": ["Frontend (React/Next)", "Node.js Backend", "Full-stack Web"],
        "Java": ["Backend (Spring)", "Android", "Enterprise Systems"],
        "Kotlin": ["Android", "Backend (Ktor/Spring)", "Mobile"],
        "Swift": ["iOS", "Mobile"],
        "Go": ["Cloud/DevOps", "Backend Services", "Networking"],
        "Rust": ["Systems", "Performance-critical Backend", "Security"],
        "C++": ["Systems", "Game Dev", "Competitive Programming"],
        "C": ["Embedded/Systems", "Low-level"],
        "C#": ["Backend (.NET)", "Windows Apps", "Game Dev (Unity)"],
        "PHP": ["Web (Laravel)", "Backend"],
        "Ruby": ["Web (Rails)", "Scripting"],
        "R": ["Data Analysis", "Statistics"],
        "Jupyter Notebook": ["Data Science/ML", "Research/Prototyping"],
        "Shell": ["DevOps/SRE", "Automation"],
    }

    topic_hints = {
        "machine-learning": "Machine Learning",
        "deep-learning": "Deep Learning",
        "ai": "AI",
        "data-science": "Data Science",
        "react": "Frontend (React)",
        "nextjs": "Frontend (Next.js)",
        "vue": "Frontend (Vue)",
        "angular": "Frontend (Angular)",
        "nodejs": "Node.js",
        "express": "Node.js (Express)",
        "fastapi": "Backend (FastAPI)",
        "django": "Backend (Django)",
        "flask": "Backend (Flask)",
        "spring": "Backend (Spring)",
        "android": "Android",
        "ios": "iOS",
        "flutter": "Flutter",
        "kubernetes": "Kubernetes",
        "docker": "Docker",
        "devops": "DevOps",
        "sre": "SRE",
        "blockchain": "Blockchain",
        "security": "Security",
        "computer-vision": "Computer Vision",
        "nlp": "NLP",
        "typescript": "TypeScript",
    }

    tracks: List[Dict[str, Any]] = []
    for lang in langs[:3]:
        for track in lang_map.get(lang, []):
            tracks.append({"track": track, "reason": f"Strong activity in {lang}."})

    for t in top_topics[:10]:
        key = t.lower().strip()
        if key in topic_hints:
            tracks.append({"track": topic_hints[key], "reason": f"Repo topics include '{t}'."})

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for tr in tracks:
        k = tr["track"]
        if k in seen:
            continue
        seen.add(k)
        deduped.append(tr)

    company_types: List[str] = []
    if any(l in langs for l in ("Python", "R", "Jupyter Notebook")) or any(k in top_topics for k in ("machine-learning", "data-science", "ai", "nlp")):
        company_types += ["AI/ML product teams", "Analytics & data platforms", "Research-oriented engineering teams"]
    if any(l in langs for l in ("Go", "Rust", "C++", "C")) or any(k in top_topics for k in ("kubernetes", "docker", "devops", "sre")):
        company_types += ["Cloud infrastructure", "Developer tools", "Systems & performance engineering"]
    if any(l in langs for l in ("JavaScript", "TypeScript")) or any(k in top_topics for k in ("react", "nextjs", "vue", "nodejs")):
        company_types += ["Consumer/product SaaS", "Web platforms", "Full-stack product teams"]
    if any(l in langs for l in ("Java", "Kotlin", "Swift")) or any(k in top_topics for k in ("android", "ios", "flutter")):
        company_types += ["Mobile-first product teams", "Fintech/mobile payments", "B2C apps"]

    if not company_types:
        company_types = ["General software product teams", "Web/backend engineering teams"]

    company_types = list(dict.fromkeys(company_types))

    return {
        "top_languages": top_langs,
        "top_topics": topics,
        "recommended_tracks": deduped[:10],
        "suggested_company_types": company_types[:10],
    }


# -----------------------------
# Analysis
# -----------------------------
def analyze_user(raw_user: Dict[str, Any]) -> Dict[str, Any]:
    mode = raw_user.get("_data_mode", "unknown")
    window = raw_user.get("_analysis_window", {})
    repos_nodes = raw_user.get("_repos_nodes") or []
    events = raw_user.get("_events") or []

    profile = {
        "login": raw_user.get("login"),
        "name": raw_user.get("name"),
        "bio": raw_user.get("bio"),
        "company": raw_user.get("company"),
        "location": raw_user.get("location"),
        "website": raw_user.get("websiteUrl") if mode == "graphql" else raw_user.get("blog"),
        "github_url": raw_user.get("url") if mode == "graphql" else raw_user.get("html_url"),
        "created_at": raw_user.get("createdAt") if mode == "graphql" else raw_user.get("created_at"),
        "followers": (raw_user.get("followers") or {}).get("totalCount") if mode == "graphql" else raw_user.get("followers", 0),
        "following": (raw_user.get("following") or {}).get("totalCount") if mode == "graphql" else raw_user.get("following", 0),
        "public_repos": (raw_user.get("repositories") or {}).get("totalCount") if mode == "graphql" else raw_user.get("public_repos", 0),
        "data_mode": mode,
        "analysis_window": window,
    }

    contributions: Dict[str, Any] = {}
    streak_365: Dict[str, int] = {"max_streak_days": 0, "current_streak_days": 0}

    if mode == "graphql":
        cc = (raw_user.get("contributionsCollection") or {})
        cal = (cc.get("contributionCalendar") or {})
        total_contrib = int(cal.get("totalContributions") or 0)

        # Flatten counts for 365d streak + activity
        counts_365 = _flatten_calendar(cal)
        start_365 = dt.date.fromisoformat(window["from"])
        end_365 = dt.date.fromisoformat(window["to"])
        streak_365 = _compute_streak_from_counts(counts_365, start_365, end_365)
        total_days = (end_365 - start_365).days + 1
        active_days = sum(1 for d in (start_365 + dt.timedelta(days=i) for i in range(total_days)) if counts_365.get(d, 0) > 0)

        # Active weeks
        active_weeks = 0
        weeks = cal.get("weeks") or []
        for w in weeks:
            week_sum = sum(int(d.get("contributionCount") or 0) for d in (w.get("contributionDays") or []))
            if week_sum > 0:
                active_weeks += 1

        # Long-window streak (stitched)
        long_counts: Dict[dt.date, int] = raw_user.get("_streak_counts") or {}
        long_window = raw_user.get("_streak_window") or {}
        long_streak = None
        if long_counts and long_window.get("from") and long_window.get("to"):
            ls = dt.date.fromisoformat(long_window["from"])
            le = dt.date.fromisoformat(long_window["to"])
            long_streak = _compute_streak_from_counts(long_counts, ls, le)

        contributions = {
            "total_contributions": total_contrib,
            "commits_365d": int(cc.get("totalCommitContributions") or 0),
            "pull_requests": int(cc.get("totalPullRequestContributions") or 0),
            "issues": int(cc.get("totalIssueContributions") or 0),
            "reviews": int(cc.get("totalPullRequestReviewContributions") or 0),
            "active_days": int(active_days),
            "active_days_ratio": float(active_days / total_days) if total_days else 0.0,
            "active_weeks": int(active_weeks),
            "active_weeks_ratio": float(active_weeks / max(1, len(weeks))),
            "streak": streak_365,
            "all_time_commit_contributions": raw_user.get("_all_time_commit_contributions"),
            "streak_window_years": raw_user.get("_streak_window_years"),
            "streak_over_window": long_streak,
            "streak_window": long_window,
        }
    else:
        evt_types: Dict[str, int] = {}
        for e in events:
            t = e.get("type", "Unknown")
            evt_types[t] = evt_types.get(t, 0) + 1

        contributions = {
            "total_contributions": None,
            "note": "REST-only mode: 365-day totals/streak require a token (GraphQL). Showing recent public event counts instead.",
            "recent_public_events": len(events),
            "event_type_counts": evt_types,
            "all_time_commit_contributions": None,
            "streak_window_years": None,
            "streak_over_window": None,
        }

    # Repo stats + language/topic aggregation
    total_stars = 0
    total_forks = 0
    open_issues = 0
    open_prs = 0
    watchers = 0
    archived = 0
    forks_count = 0
    recently_active = 0

    lang_weights: Dict[str, float] = {}
    topic_counts: Dict[str, int] = {}
    top_repos: List[Dict[str, Any]] = []

    cutoff_recent = _days_ago(90)

    for r in repos_nodes:
        if mode == "graphql":
            stars = int(r.get("stargazerCount") or 0)
            forks = int(r.get("forkCount") or 0)
            wcount = int((r.get("watchers") or {}).get("totalCount") or 0)
            issues_open = int((r.get("issues") or {}).get("totalCount") or 0)
            prs_open = int((r.get("pullRequests") or {}).get("totalCount") or 0)
            is_archived = bool(r.get("isArchived"))
            is_fork = bool(r.get("isFork"))
            pushed = _dateparse(r.get("pushedAt"))
            repo_url = r.get("url")
            name = r.get("name")
            desc = r.get("description")
            primary_lang = (r.get("primaryLanguage") or {}).get("name")

            for edge in (r.get("languages") or {}).get("edges") or []:
                ln = (edge.get("node") or {}).get("name")
                sz = float(edge.get("size") or 0)
                if ln:
                    lang_weights[ln] = lang_weights.get(ln, 0.0) + sz

            for tn in (r.get("repositoryTopics") or {}).get("nodes") or []:
                t = ((tn.get("topic") or {}).get("name") or "").strip()
                if t:
                    topic_counts[t.lower()] = topic_counts.get(t.lower(), 0) + 1
        else:
            stars = int(r.get("stargazers_count") or 0)
            forks = int(r.get("forks_count") or 0)
            wcount = int(r.get("watchers_count") or 0)
            issues_open = int(r.get("open_issues_count") or 0)
            prs_open = 0
            is_archived = bool(r.get("archived"))
            is_fork = bool(r.get("fork"))
            pushed = _dateparse(r.get("pushed_at"))
            repo_url = r.get("html_url")
            name = r.get("name")
            desc = r.get("description")
            primary_lang = r.get("language")
            if primary_lang:
                lang_weights[primary_lang] = lang_weights.get(primary_lang, 0.0) + 1.0

        total_stars += stars
        total_forks += forks
        watchers += wcount
        open_issues += issues_open
        open_prs += prs_open
        if is_archived:
            archived += 1
        if is_fork:
            forks_count += 1
        if pushed and pushed >= cutoff_recent:
            recently_active += 1

        top_repos.append(
            {
                "name": name,
                "url": repo_url,
                "description": desc,
                "stars": stars,
                "forks": forks,
                "watchers": wcount,
                "open_issues": issues_open,
                "primary_language": primary_lang,
                "pushed_at": pushed.isoformat() if pushed else None,
            }
        )

    top_repos_sorted = sorted(top_repos, key=lambda x: (x["stars"], x["forks"]), reverse=True)[:8]

    # Languages summary
    lang_total = sum(lang_weights.values())
    lang_percent = {k: (v / lang_total) if lang_total else 0.0 for k, v in sorted(lang_weights.items(), key=lambda kv: kv[1], reverse=True)}

    diversity = _entropy(lang_weights) if lang_weights else 0.0

    # Coefficient scoring
    impact_norm = _norm_log(total_stars, 5000)

    if mode == "graphql":
        contrib_total = float(contributions.get("total_contributions") or 0)
        collab_total = float((contributions.get("pull_requests") or 0) + (contributions.get("issues") or 0) + (contributions.get("reviews") or 0))
        activity_norm = _norm_log(contrib_total, 1500)
        collab_norm = _norm_log(collab_total, 600)

        active_weeks_ratio = float(contributions.get("active_weeks_ratio") or 0.0)
        streak_norm = _clamp(float(streak_365.get("max_streak_days", 0)) / 60.0, 0.0, 1.0)
        consistency = 0.7 * active_weeks_ratio + 0.3 * streak_norm
    else:
        evt_count = float(contributions.get("recent_public_events") or 0)
        activity_norm = _norm_log(evt_count, 100)
        evt_types = contributions.get("event_type_counts") or {}
        collab_proxy = float(evt_types.get("PullRequestEvent", 0) + evt_types.get("IssuesEvent", 0) + evt_types.get("PullRequestReviewEvent", 0))
        collab_norm = _norm_log(collab_proxy, 50)
        consistency = 0.35

    score = 100.0 * (0.40 * activity_norm + 0.20 * collab_norm + 0.20 * impact_norm + 0.10 * consistency + 0.10 * diversity)
    score = float(_clamp(score, 0.0, 100.0))

    if score >= 85:
        level = "Elite"
    elif score >= 70:
        level = "Prolific"
    elif score >= 55:
        level = "Active"
    elif score >= 40:
        level = "Emerging"
    else:
        level = "Starter"

    coefficient = {
        "open_source_contribution_coefficient": round(score, 2),
        "level": level,
        "breakdown": {
            "activity_norm": round(activity_norm, 4),
            "collaboration_norm": round(collab_norm, 4),
            "impact_norm": round(impact_norm, 4),
            "consistency": round(consistency, 4),
            "diversity": round(diversity, 4),
        },
        "interpretation": (
            "A composite score using recent contribution activity, collaboration signals (PRs/issues/reviews), "
            "repository impact (stars), consistency, and language diversity."
        ),
    }

    # Badges
    badges: List[Dict[str, str]] = []
    badges.append(_badge("Open Source", level, "blue"))
    if mode == "graphql":
        badges.append(_badge("Contrib (365d)", str(contributions.get("total_contributions", 0)), "brightgreen" if score >= 55 else "yellow"))
        badges.append(_badge("Streak (365d)", f"{streak_365.get('max_streak_days', 0)}d", "purple" if streak_365.get("max_streak_days", 0) >= 14 else "lightgrey"))
        badges.append(_badge("Active Weeks", f"{int((contributions.get('active_weeks_ratio', 0) or 0)*100)}%", "orange"))
        badges.append(_badge("PRs (365d)", str(contributions.get("pull_requests", 0)), "informational"))
        if contributions.get("all_time_commit_contributions") is not None:
            badges.append(_badge("Commits (all-time)", str(contributions.get("all_time_commit_contributions", 0)), "success"))
        if contributions.get("streak_over_window") and contributions.get("streak_window_years"):
            badges.append(_badge(f"Longest Streak ({contributions.get('streak_window_years')}y)", f"{contributions['streak_over_window'].get('max_streak_days',0)}d", "blueviolet"))
    else:
        badges.append(_badge("Mode", "REST-Limited", "lightgrey"))
        badges.append(_badge("Events (recent)", str(contributions.get("recent_public_events", 0)), "informational"))

    badges.append(_badge("Stars", str(total_stars), "success" if total_stars >= 50 else "yellow"))
    distinct_langs = len([k for k, v in lang_weights.items() if v > 0])
    badges.append(_badge("Languages", str(distinct_langs), "blueviolet" if distinct_langs >= 5 else "blue"))

    # Recommendations
    recs = _language_recommendations(lang_weights, topic_counts)

    summary = {
        "repos_analyzed": len(repos_nodes),
        "fork_repos": forks_count,
        "archived_repos": archived,
        "recently_active_repos_90d": recently_active,
        "total_stars_received": total_stars,
        "total_forks_received": total_forks,
        "watchers_total": watchers,
        "open_issues_total": open_issues,
        "open_prs_total": open_prs if mode == "graphql" else None,
    }

    return {
        "profile": profile,
        "summary": summary,
        "contributions": contributions,
        "top_repositories": top_repos_sorted,
        "languages": {
            "weights": lang_weights,
            "percentages": lang_percent,
            "diversity_entropy": round(diversity, 4),
        },
        "topics": {
            "counts": topic_counts,
            "top": sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:15],
        },
        "coefficient": coefficient,
        "badges": badges,
        "recommendations": recs,
        "meta": {
            "generated_at": _now_utc().isoformat(),
            "data_mode": mode,
            "token_configured": bool(GITHUB_TOKEN),
            "cache_ttl_seconds": CACHE_TTL_SECONDS,
        },
    }


# -----------------------------
# Flask routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    try:
        return render_template("index.html")
    except Exception:
        return (
            """
            <!doctype html>
            <html>
            <head><meta charset="utf-8"><title>GitHub Analyzer</title></head>
            <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 24px;">
              <h2>GitHub Analyzer API is running</h2>
              <p>Try: <code>/api/analyze?username=octocat</code></p>
              <p>Add a template at <code>templates/index.html</code> to build the UI.</p>
            </body>
            </html>
            """,
            200,
            {"Content-Type": "text/html; charset=utf-8"},
        )


def _get_username_from_request() -> Optional[str]:
    if request.method == "GET":
        return (request.args.get("username") or "").strip()
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return (payload.get("username") or "").strip()
    return (request.form.get("username") or "").strip()


@app.route("/api/analyze", methods=["GET", "POST"])
def api_analyze():
    username = _get_username_from_request()

    if not username:
        return jsonify({"error": "Missing 'username'."}), 400

    if not USERNAME_RE.match(username):
        return jsonify({"error": "Invalid GitHub username format."}), 400

    key = username.lower()
    cached = _CACHE.get(key)
    if cached:
        ts, data = cached
        if (time.time() - ts) <= CACHE_TTL_SECONDS:
            return jsonify({"cached": True, **data})

    try:
        raw = fetch_with_graphql(username) if GITHUB_TOKEN else fetch_with_rest(username)
        analysis = analyze_user(raw)
        _CACHE[key] = (time.time(), analysis)
        return jsonify({"cached": False, **analysis})
    except GitHubAPIError as e:
        msg = str(e)
        status = 404 if "not found" in msg.lower() else 502
        return jsonify({"error": msg, "hint": "Set GITHUB_TOKEN to improve reliability & stats."}), status
    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {e}"}), 500


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True, "token_configured": bool(GITHUB_TOKEN), "cache_ttl_seconds": CACHE_TTL_SECONDS})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
