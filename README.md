# job-search-agent

A small agent that checks companies' career pages for jobs matching your keywords, reads each posting, scores it against a profile you write in plain text, and sends the best matches to Telegram.

It runs on your own machine once or twice a day. The scoring can use a local model (Ollama), Claude or OpenAI.

## How it works

1. **Discover.** For each source in `sources.txt` and each keyword in `config.yaml`, it downloads the company's search results (or queries its JSON API) and collects the job links.
2. **Process.** For each job it has not seen before, it downloads the posting and applies cheap rules first (blocked countries, blocked keywords). The jobs that remain are scored 0-10 by the LLM against your profile.
3. **Notify.** The best unsent matches go to a Telegram chat. Problems (a source that returned nothing, a model that is down) arrive as a separate warning message.

Every job is stored in a SQLite database, so each posting is downloaded, scored and sent once. If a run is interrupted, the next one continues where it stopped.

## Requirements

- Python 3.10 or newer
- One LLM: [Ollama](https://ollama.com) with a model installed (for example `qwen3:14b`), or a Claude or OpenAI API key
- A Telegram bot and a chat to send to (setup below)

## Install

```
git clone <your repository address> job-search-agent
cd job-search-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The `-e` install lets Python find the package, so `python -m job_search_agent` works from any folder. Use a virtual environment: on many Linux systems a system-wide `pip install` is refused.

To use Claude or OpenAI, install its library too: `python -m pip install -e ".[claude]"` or `".[openai]"`.

## Set up

Copy the three templates and edit the copies. The real files are listed in `.gitignore`, so they are never shared by accident.

```
cp .env.example .env
cp config.example.yaml config.yaml
cp sources.example.txt sources.txt
```

### 1. `.env` (secrets)

```
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
ANTHROPIC_API_KEY=       # only if llm.provider is claude
ANTHROPIC_WORKSPACE_ID=  # only if Anthropic says the key needs a workspace (see Troubleshooting)
OPENAI_API_KEY=          # only if llm.provider is openai
OLLAMA_HOST=             # optional, overrides llm.ollama.host
```

No quotes and no spaces around `=`. Real environment variables take precedence over this file, so an old `TELEGRAM_BOT_TOKEN` exported in your shell will silently override it.

### 2. Telegram

1. In Telegram, open **@BotFather**, send `/newbot` and follow the steps. It gives you the bot token.
2. Open a chat with your new bot and press **Start**. A bot cannot message you until you have messaged it.
3. Find your chat ID. With the token already in `.env`:

   ```
   python - <<'EOF'
   import httpx
   from job_search_agent.config import load_config
   cfg = load_config(require_telegram=False)
   data = httpx.get(f"https://api.telegram.org/bot{cfg.telegram.bot_token}/getUpdates", timeout=30).json()
   for u in data.get('result', []):
       chat = (u.get('message') or {}).get('chat', {})
       if chat: print(chat['id'], chat.get('type'), chat.get('title') or chat.get('first_name'))
   EOF
   ```

   If it prints nothing, send the bot another message and run it again.
4. To send to a **group**, add the bot to the group, type `/start@your_bot_username` in it, and run the script again. Group IDs are negative numbers (often starting with `-100`). Keep the minus sign in `.env`.

### 3. `config.yaml`

- **keywords**: words searched on every source.
- **profile**: who you are and what you want, in plain text. The model judges every job only against this text, so write what you mean and nothing more. Say what you rule out (countries, seniority, internships).
- **blocked_countries / blocked_keywords**: jobs matching these are dropped before the LLM sees them. Use `blocked_keywords: []` for none.
- **llm**: choose `ollama`, `claude` or `openai` and set its model name. A small, cheap model is enough for scoring.
- **limits**: delay between downloads, how many jobs to score per run, the score threshold for notifications, and how many matches to send per run.

### 4. `sources.txt`

One company per line. Two kinds of source:

**HTML sources**, where the company's search page lists jobs in the page itself:

```
name | search_url_with_{query} | link_pattern
```

- Search for one word on the company's site, copy the address, and replace the word with `{query}`.
- `link_pattern` is a regular expression that the job posting addresses match. Take it from a real posting address. For `https://careers.nordicsemi.com/jobs/8587502-senior-soc-design-lead` the pattern is `/jobs/\d+-`. Without a pattern the source is skipped.
- Only the first page of results is read.

**API sources**, for sites whose job list is loaded by JavaScript and so is empty in the downloaded page:

```
name | adapter:<adapter_name> | posting_url_with_{id}
```

Adapters included: `amd`, `mediatek`, `realtek`, `tencent`. `{id}` is replaced with the job's ID to build the link sent to you. See "Adding a company" below for how to recognise and add one.

**Note:** Some API sources (like Realtek) may return job data in a different language. Check the results and consider blocking unwanted languages in your profile if needed.

Lines starting with `#` are ignored, so park broken sources with a `#`.

## Run

```
python -m job_search_agent --check-config     # validate config.yaml and sources.txt
python -m job_search_agent --check-sources    # test every source and its link pattern
python -m job_search_agent --test-telegram    # send a test message
python -m job_search_agent --dry-run          # full run, but print matches instead of sending
python -m job_search_agent                    # full run
```

Suggested first use, in that order. The first real run may find more new jobs than `max_jobs_scored_per_run`; the rest are handled on the next run, and nothing is lost. `-v` gives more detailed logging.

A dry run still downloads, scores and stores jobs, so a following real run only has to notify.

### Schedule it

Cron does not activate your virtual environment, so call its Python directly. For example, daily at 08:00:

```
0 8 * * * cd /path/to/job-search-agent && .venv/bin/python -m job_search_agent >> data/run.log 2>&1
```

Once or twice a day is plenty: career pages do not change faster, and it keeps the load on each site small.

## Choosing the LLM

| Provider | Notes |
|---|---|
| `ollama` | Free and private: nothing leaves your machine. A 14B model is enough. Slower per job. |
| `claude` | Needs `ANTHROPIC_API_KEY` and API credit (separate from a chat subscription). |
| `openai` | Needs `OPENAI_API_KEY` and API credit. |

With Claude or OpenAI, your profile and every job posting are sent to that provider. Each run scores at most `max_jobs_scored_per_run` jobs, roughly 3,000 to 4,000 tokens each, and only new jobs after the first run.

## Writing a good profile

The scoring prompt (`src/job_search_agent/prompts/score_job.txt`) tells the model to rate each job only against what the profile says. So:

- State seniority and role type. Without it, junior roles can score high.
- State hard exclusions plainly (countries, internships, graduate roles).
- Do not add wishes you do not mean. If the profile says "willing to relocate", that is all the model will assume.
- If scores look wrong, read the one-line reason in the message, then fix the profile or the prompt.

## Adding a company

1. In your browser, search a keyword on the company's careers site and copy the address.
2. Run `python -m job_search_agent --check-sources` after adding the line to `sources.txt`. It shows how many job links were found and the first few.
3. If the job count is zero and the page is large, the list is probably loaded by JavaScript. Open the browser's developer tools (F12), go to **Network**, filter by **Fetch/XHR**, search again, and look for a request that returns the job list as JSON. If you find one, an adapter can be written for it: a small module in `src/job_search_agent/adapters/` with a `search(fetcher, keyword)` function that returns jobs, registered in `adapters/__init__.py`. See `adapters/amd.py` as a model.

## Responsible use

- The agent does **not** read `robots.txt`. Some career sites ask automated tools to stay away from search pages. Check a site's `robots.txt` and terms before adding it, and decide for yourself.
- It waits `delay_seconds` between requests to the same site and retries politely. Keep that delay, run it a couple of times a day at most, and raise the delay if you see HTTP 403 or 429 errors.
- Do not use it on LinkedIn or Glassdoor with your account: both prohibit automated access and may restrict accounts that do it.
- Your database (`data/jobs.db`) holds the full text of every posting it processed. Do not publish it.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `externally-managed-environment` when installing | You are not inside the virtual environment. Activate it (`source .venv/bin/activate`) and check `which python`. |
| `No module named job_search_agent` | Run `python -m pip install -e .` inside the virtual environment. |
| `Telegram rejected the bot token` | The token is wrong, revoked, or an environment variable is overriding `.env` (`env \| grep TELEGRAM`). In BotFather, `/mybots` shows the current token. |
| `chat not found` | You have not pressed Start in the bot's chat, or `TELEGRAM_CHAT_ID` is wrong. Groups can get a new ID if converted to supergroups. |
| `Cannot reach Ollama` | Ollama is not running (`ollama serve`) or `llm.ollama.host` is wrong. |
| `Ollama has no model named ...` | Check `ollama list` and set the exact name in `config.yaml`. |
| A source returns `0 job links` | The company has no matches for that keyword, the link pattern is wrong, or the list is loaded by JavaScript. See "Adding a company". |
| Scoring stops after 3 failures | The model is probably down. The failed jobs are retried on the next run (up to 3 attempts each). |
| Claude error: key "is not scoped to a workspace" | Create the API key inside a workspace in the developer console, or set `ANTHROPIC_WORKSPACE_ID` in `.env` to that workspace's ID. |
| A config error mentions a list | Empty `-` lines under a list are invalid YAML for this project. Use `[]` for an empty list. |

## Project layout

```
config.example.yaml   settings template (copy to config.yaml)
sources.example.txt   sources template (copy to sources.txt)
.env.example          secrets template (copy to .env)
data/                 created at runtime: jobs.db and logs
src/job_search_agent/
  cli.py              command line
  pipeline.py         one full run: discover, process, notify
  config.py           loads and validates config.yaml and .env
  sources.py          parses sources.txt, builds search URLs
  fetcher.py          polite downloads: delays, retries, cookies
  links.py            extracts links from a results page
  pagetext.py         reduces a posting to readable text
  filters.py          blocked countries and keywords
  matcher.py          builds the scoring prompt, validates the answer
  db.py               SQLite storage
  adapters/           JSON API sources (amd, mediatek, realtek, tencent)
  llm/                Ollama, Claude and OpenAI backends
  notify/telegram.py  Telegram messages
  prompts/            the scoring prompt, a plain text file you can edit
```

## Limitations

- Pages built with JavaScript are only supported through an API adapter.
- Only the first page of an HTML search is read.
- Scoring quality depends on how much text the company's data provides. Some APIs give only a short summary.
- The country filter is conservative on purpose: ambiguous locations pass through and the model makes the call.
