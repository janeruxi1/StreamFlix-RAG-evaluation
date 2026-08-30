# Setup

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure credentials

```bash
cp .env.example .env
```

Open `.env` and paste your OpenAI key into `OPENAI_API_KEY`.

**`.env` is gitignored — it will never be committed.**

### Recommended: use a project-scoped key with a spend cap

At [platform.openai.com/api-keys](https://platform.openai.com/api-keys):

1. Create a **project-scoped key** for this project specifically, not your
   account-wide key. If it ever leaks, you revoke one project — not everything.
2. Set a **monthly usage cap**. `$30` is generous for this project
   (realistic full-run cost is `$10–25`, and the response cache means
   re-runs are free).

## 3. Verify the key loaded — without exposing it

```bash
python -m src.llm.provider --check
```

Expected output:

```
==============================================================
  Credential check — values are MASKED and safe to share
==============================================================
  LLM_PROVIDER      : openai
  GENERATION_MODEL  : gpt-4o-mini
  JUDGE_MODEL       : gpt-4o
  EMBEDDING_MODEL   : BAAI/bge-small-en-v1.5

  OPENAI_API_KEY: sk-proj...9f2a  (length 164)
  ANTHROPIC_API_KEY: (not set)

  ✓ Provider constructed: <OpenAIProvider model='gpt-4o-mini'>
  ✓ Ready. Credentials resolved without being exposed.
==============================================================
```

The key is shown masked — enough to confirm the right one loaded, useless
to anyone who sees your screen or a screenshot.

**Never** print the raw key. If you need to check it, use this command.

## 4. Install the commit-time safety net

```bash
pre-commit install
detect-secrets scan > .secrets.baseline
```

This installs git hooks that run on **every commit** and block it if:

- Anything resembling an API key is staged
- A notebook has un-stripped outputs (outputs can bake in a leaked key permanently)
- A file over 5MB is staged
- Unresolved merge-conflict markers are present

Verify the hooks are live:

```bash
pre-commit run --all-files
```

## 5. Confirm nothing sensitive is tracked

```bash
git status --ignored | grep -A5 "Ignored files"
```

`.env` should appear under **ignored**, never under "changes to be committed."

---

## Security model

| Layer | What it protects against |
|---|---|
| `.gitignore` (created before `.env` existed) | The `.env` file entering git |
| `.env.example` with placeholders | Anyone confusing the template for the real thing |
| `src/llm/provider.py` reads only from env | Hardcoded keys in source |
| `masked_credential()` helper | Keys appearing in terminal output or screenshots |
| `nbstripout` pre-commit hook | Keys baked into `.ipynb` output JSON |
| `detect-secrets` pre-commit hook | Anything key-shaped reaching a commit |
| `CallBudget` ceiling | Runaway loops burning your spend cap |
| Project-scoped key + monthly cap | Blast radius if a key does leak |

### If a key ever leaks

**Revoke first, clean up second.**

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Delete the exposed key (takes ~15 seconds)
3. Generate a replacement, update `.env`

A revoked key is harmless no matter where it ended up. Do **not** spend time
scrubbing git history before revoking — that's backwards.

---

## Cost control

Two mechanisms keep spend predictable:

**Response cache** (`ENABLE_LLM_CACHE=true`) — identical calls are served from
disk. Re-running a notebook after a kernel restart costs nothing.

**Call ceiling** (`MAX_LLM_CALLS_PER_RUN=2000`) — a hard stop per process run.
If a loop misbehaves, it raises instead of draining your budget.

Estimated full-project cost with both enabled: **$10–25**, most of it in the
evaluation harness (LLM-as-judge is the expensive part).
