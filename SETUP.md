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
detect-secrets scan --baseline .secrets.baseline
```

> **Windows / PowerShell:** use `--baseline` as shown above, **not**
> `detect-secrets scan > .secrets.baseline`. PowerShell's `>` redirect
> writes **UTF-16**, and detect-secrets can only read UTF-8 — the hook
> then fails with `error: Unable to read baseline` on every commit. The
> failure mode is confusing because the file looks fine in an editor.
> `--baseline` has the tool write the file itself, in the right encoding.
>
> If you already hit this, re-encode rather than regenerate (regenerating
> loses any audited entries):
>
> ```bash
> python -c "import json,pathlib; p=pathlib.Path('.secrets.baseline'); d=json.loads(p.read_bytes().decode('utf-16')); p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')"
> ```

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

Estimated cost of the full credentialed run with both enabled: **about $3**
with the default two judged arms, most of it in the evaluation harness
(LLM-as-judge is the expensive part). The run script prints the plan and
asks before it spends anything:

```bash
python scripts/run_llm_eval.py --dry-run     # the plan; spends nothing
python scripts/run_llm_eval.py               # confirm, then run notebooks 01-07
```

It writes each notebook's output to `reports/llm_run/` with credentials
scrubbed, and the measured numbers to `reports/metrics/`. Set
`JUDGE_ARMS=all` in `.env` to judge every prompt variant (about $6, and
raise `MAX_LLM_CALLS_PER_RUN` to 3000 — the script refuses to start a run
that would hit the ceiling halfway through).
