# AI Apply — Automated Job Application Pipeline

Fully automated job application system: scrapes listings from 10 job boards, scores them against your profile using semantic embeddings, and for top matches automatically generates a tailored CV, cover letter, form answers, and digest email — all driven by a local LLM (Qwen via Ollama).

---

## Features

- **Multi-source scraping** — pulls from 10 job boards simultaneously
- **Semantic matching** — sentence-transformer embeddings rank jobs against your full life story + profile
- **AI relevance filter** — automatically filters for ML/AI/CV-related roles
- **SQLite storage** — deduplicates and persists all scraped jobs and application state
- **Automated CV customization** — LLM rewrites `employment.tex`, `skills.tex`, `projects.tex` for each job and compiles to PDF
- **Automated cover letter generation** — LLM writes a tailored LaTeX cover letter, compiled to PDF
- **Form answer generation** — LLM pre-answers common application questions (motivation, salary, visa, etc.)
- **Form-fill guide** — maps pre-generated answers to field names for browser-based auto-fill
- **Digest email notifier** — sends an HTML email every 2–3 days with new high-match jobs
- **Background daemon** — runs the full pipeline on a configurable interval (default: every 48 h)
- **Web dashboard** — Flask UI with filtering, sorting, and apply/hide actions
- **CLI tools** — scrape, match, export, customize, run pipeline, and view top jobs from the terminal

---

## Supported Job Sources

| Source | Type | API Key Required |
|---|---|---|
| **Remotive** | REST API | No |
| **Arbeitnow** | REST API | No |
| **Himalayas** | REST API | No |
| **The Muse** | REST API | No |
| **Adzuna** | REST API | Yes ([developer.adzuna.com](https://developer.adzuna.com)) |
| **JSearch** (Google Jobs) | RapidAPI | Yes ([rapidapi.com](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)) |
| **LinkedIn** | Guest scraper | No |
| **Indeed** | Web scraper | No |
| **Glassdoor** | Web scraper | No |
| **StepStone** | Web scraper | No |

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url> && cd job_finder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama (local LLM)

```bash
bash setup_ollama.sh
# or manually:
# brew install ollama && ollama pull qwen3.5:9b
```

### 3. Configure Your Profile

Edit `profile.yaml` with your skills, desired titles, keywords, locations, and scoring weights.

```yaml
pipeline:
  ollama_model: qwen3.5:9b   # local model used for CV/cover letter generation
  min_score: 0.50            # minimum match score to trigger automation
  max_applications_per_run: 10
```

For CV customization, place your `life-story.md` and LaTeX CV files in `~/CV/`:

```
~/CV/
├── life-story.md         # free-form description of your background — fed to the LLM
├── cv-llt.tex            # main CV LaTeX file
├── employment.tex        # overwritten per application
├── skills.tex            # overwritten per application
├── projects.tex          # overwritten per application
└── applications/         # output folder — one subfolder per application
```

### 4. Set API Keys (optional but recommended)

```bash
export ADZUNA_APP_ID="your_app_id"
export ADZUNA_APP_KEY="your_app_key"
export RAPIDAPI_KEY="your_rapidapi_key"

# For digest emails:
export GMAIL_USER="you@gmail.com"
export GMAIL_APP_PASSWORD="your_app_password"
export NOTIFY_EMAIL="you@gmail.com"
```

> **Tip:** Add these to a `.env` file in the project root.

---

## Usage

### Scrape Jobs

```bash
# Scrape all configured boards
python main.py scrape

# Scrape specific boards only
python main.py scrape --boards remotive arbeitnow himalayas

# Limit results per board per query
python main.py scrape --max 30

# Also fetch full job descriptions (slower but better matching)
python main.py scrape --fetch-details
```

### Re-Score Jobs

After editing `profile.yaml`, re-score all stored jobs without re-scraping:

```bash
python main.py match

# Only keep jobs above a minimum score
python main.py match --min-score 0.3
```

### View Top Matches

```bash
python main.py top

# Show top 50 with minimum score
python main.py top --limit 50 --min-score 0.2
```

### Generate Customized Application for a Single Job

```bash
# Generates tailored CV (PDF), cover letter (PDF), and form answers
python main.py customize --url "https://example.com/job/123"
```

Output is saved to `~/CV/applications/<company-role-slug>/`.

### Show Pre-Generated Form Answers

```bash
python main.py answers --url "https://example.com/job/123"
```

Prints a fill guide mapping form field names to your pre-generated answers.

### Run the Full Automation Pipeline (one shot)

```bash
# Scrape → match → customize → cover letter → form answers → email
python main.py pipeline

# Preview without generating any files
python main.py pipeline --dry-run

# Process up to 5 jobs above a 0.6 threshold
python main.py pipeline --max 5 --threshold 0.6
```

### Run as Background Daemon

```bash
# Repeats every 48 hours (default)
python main.py daemon

# Custom interval
python main.py daemon --interval 24
```

Send `SIGTERM` or `Ctrl+C` for a graceful shutdown after the current cycle.

### Export to JSON

```bash
python main.py export -o top_jobs.json
python main.py export --limit 100 --min-score 0.3 -o filtered.json
```

### Launch Web Dashboard

```bash
python main.py ui

# Custom port / debug mode
python main.py ui --port 8080 --debug
```

Open **http://localhost:5000** in your browser.

---

## How the Automation Pipeline Works

```
scrape (10 boards)
    │
    ▼
semantic match (sentence-transformers + profile embeddings)
    │
    ▼  jobs above threshold
customize CV  ──►  cover letter  ──►  form answers
    │
    ▼
save application to DB  ──►  digest email (every 2-3 days)
```

1. **Scrape** — pulls fresh listings from all configured boards in parallel
2. **Match** — encodes each job description + your profile into embeddings, ranks by cosine similarity
3. **Customize CV** — LLM reads `life-story.md` and rewrites `employment.tex`, `skills.tex`, `projects.tex` to emphasize relevant experience; compiles to PDF via `pdflatex`
4. **Cover letter** — LLM writes tailored body paragraphs; a fixed LaTeX wrapper is applied and compiled to PDF
5. **Form answers** — LLM pre-answers common screening questions (motivation, relocation, salary, visa)
6. **Notify** — sends an HTML digest email with new matches grouped by domain (3D Vision, Robotics, etc.)

---

## Project Structure

```
job_finder/
├── main.py              # CLI entry point
├── app.py               # Flask web dashboard
├── pipeline.py          # Full automation orchestrator + daemon loop
├── matcher.py           # Semantic scoring engine (sentence-transformers)
├── cv_customizer.py     # LLM-driven CV tailoring + LaTeX compilation
├── cover_letter.py      # LLM-driven cover letter generation + LaTeX compilation
├── form_answers.py      # LLM-driven screening question answers
├── form_filler.py       # Field-mapping fill guide for browser auto-fill
├── notifier.py          # Digest email sender (Gmail SMTP)
├── llm.py               # Ollama/Qwen integration (generate_latex, generate_structured)
├── models.py            # Data models (Job, JobBoard, SearchQuery)
├── storage.py           # SQLite persistence (jobs, applications, pipeline runs)
├── profile.yaml         # Your profile config (skills, titles, search params, weights)
├── setup_ollama.sh      # One-shot Ollama + model installer
├── requirements.txt     # Python dependencies
├── jobs.db              # SQLite database (created on first run)
├── scrapers/
│   ├── base.py          # Abstract scraper interface
│   ├── adzuna.py        # Adzuna API scraper
│   ├── jsearch.py       # JSearch (RapidAPI) scraper
│   ├── remotive.py      # Remotive API scraper
│   ├── arbeitnow.py     # Arbeitnow API scraper (EU + remote, no key)
│   ├── himalayas.py     # Himalayas API scraper (remote tech, no key)
│   ├── themuse.py       # The Muse API scraper (400k+ jobs, no key)
│   ├── linkedin.py      # LinkedIn scraper (authenticated)
│   ├── linkedin_guest.py# LinkedIn guest scraper (no login)
│   ├── indeed.py        # Indeed scraper
│   ├── glassdoor.py     # Glassdoor scraper
│   └── stepstone.py     # StepStone scraper
└── templates/           # Jinja2 templates for web UI
    ├── base.html
    ├── dashboard.html
    ├── jobs.html
    └── job_detail.html
```

---

## Configuration Reference

### `profile.yaml`

| Section | Description |
|---|---|
| `skills` | Your technical skills (matched against job descriptions) |
| `titles` | Desired job titles |
| `keywords` | Domain keywords that boost a job's score |
| `search.queries` | Search terms sent to each job board |
| `search.locations` | Locations to search |
| `search.boards` | Which boards to scrape |
| `search.remote` | Include remote positions |
| `search.max_age_days` | Skip jobs older than N days |
| `preferred_locations` | Locations that boost score |
| `weights.skills` | Weight for skill keyword overlap |
| `weights.title` | Weight for title match |
| `weights.semantic` | Weight for embedding similarity (recommended: 0.55+) |
| `weights.location` | Weight for location preference |
| `pipeline.ollama_model` | Ollama model to use (default: `qwen3.5:9b`) |
| `pipeline.min_score` | Minimum score to trigger automation |
| `pipeline.max_applications_per_run` | Cap on applications per pipeline run |

---

## Environment Variables

| Variable | Required For |
|---|---|
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | Adzuna scraper |
| `RAPIDAPI_KEY` | JSearch scraper |
| `GMAIL_USER` | Digest email sender |
| `GMAIL_APP_PASSWORD` | Digest email (Gmail App Password) |
| `NOTIFY_EMAIL` | Digest email recipient |

---

## Adding a New Scraper

1. Create `scrapers/my_board.py` implementing `scrape()` and `get_job_details()` methods
2. Add the board to the `JobBoard` enum in `models.py`
3. Register it in `scrapers/__init__.py`:
   ```python
   from .my_board import MyBoardScraper
   SCRAPERS["my_board"] = MyBoardScraper
   ```
4. Add `"my_board"` to `search.boards` in `profile.yaml`

---

## License

MIT
