# GitHub Profile Analyzer (Open Source Interest Analyzer)

A Flask web app that analyzes a GitHub username using GitHub REST + GraphQL APIs to produce:
- Contribution stats (365-day totals, streaks)
- **All-time commit contributions** (GraphQL, computed in 1-year chunks)
- An **Open Source Contribution Coefficient** (0–100)
- Language/topic insights, top repos, badges, and track recommendations

This is useful for **mentors / professors** to understand student interests and track progress, and for **hiring managers** to quickly get structured insights into a candidate’s public GitHub signals.

## Quick start

```bash
pip install -r requirements.txt
export GITHUB_TOKEN="github_pat_..."
python app.py
```

Open: `http://localhost:5000`

## Token notes

- Without `GITHUB_TOKEN`, the app runs in **REST-limited mode** and cannot compute accurate 365-day contribution totals/streaks.
- With `GITHUB_TOKEN`, the app uses **GraphQL** and unlocks contribution totals, streaks, and all-time commit aggregation.
- GitHub GraphQL has a strict rule: `contributionsCollection(from,to)` cannot exceed **1 year**. This app handles it safely by chunking requests.

## API

- `GET /api/analyze?username=<github_username>`
- `POST /api/analyze` with JSON: `{ "username": "octocat" }`
- `GET /healthz` for status

## Deploy

Works on any Flask-friendly platform (Render, Railway, Fly.io, VPS).
Set environment variable:

- `GITHUB_TOKEN`
- optional: `CACHE_TTL_SECONDS` (default 600), `STREAK_WINDOW_YEARS` (default 3)

## Landing page

A static landing page is included as `landing.html`. You can host it with GitHub Pages / Netlify / Vercel, or serve it from Flask.

---

**Privacy:** The analyzer is designed to use **public GitHub data**. Use responsibly and with user consent.
