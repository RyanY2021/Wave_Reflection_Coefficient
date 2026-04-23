---
name: raise-issue
description: Draft a research issue in lab-notebook style, review with the user, then push to this repo's GitHub Issues. Use when the user wants to record a bug, open question, unexplained observation, or research thread for this project.
---

# raise-issue

Workflow: **discuss → draft → approve → push**. The user describes a problem
or observation; you draft a GitHub issue in the notebook style of
`docs/lab_notebook.md`; the user reviews; on approval, you push via `gh`.

## When to invoke

- User explicitly types `/raise-issue`.
- User says something like "let's file this as an issue", "open an issue for
  this", "track this", or otherwise signals they want a finding recorded on
  GitHub — invoke proactively in that case.
- Do **not** invoke for throwaway TODOs or tasks already covered by an open
  issue — check existing issues first (`gh issue list`) before drafting a
  duplicate.

## Repo

`RyanY2021/Wave_Reflection_Coefficient` — pass explicitly via
`--repo RyanY2021/Wave_Reflection_Coefficient` on every `gh` call; do not
rely on the working directory being inside a cloned remote.

## Step 1 — Gather context

Before drafting, make sure you have:
- **What the user was doing** when the observation surfaced (test id, script,
  pipeline stage).
- **Concrete observation** — numbers, file paths, figures if any.
- **What they already suspect** (hypothesis) or whether it's a pure
  "don't know yet" observation.

If any of these are thin, ask one short clarifying question before drafting.
Don't draft from a one-line prompt — thin issues decay fast.

## Step 2 — Draft in notebook style

Use this structure verbatim for the issue body:

```markdown
**Category:** coding | pipeline | physics | research-target | confusing-phenomenon
**Status:** open
**Related:** <files, PDFs, test ids, prior issue numbers — use full paths or #NN>

### Context
<what the user was doing; which test / script / pipeline stage>

### Observation / problem
<concrete facts; numbers; file paths. Bullets are fine.>

### Hypothesis
<what you think is going on, with cited section of the pipeline doc / PDF
if relevant. If genuinely unknown, say so — don't invent a hypothesis.>

### Possible approach (next actionable steps)
<numbered list of concrete things the user can *do next*, not vague
aspirations. Each item should be small enough to attempt in one sitting.
Reference file paths and specific commands where possible.>

### Disproof criteria
<what result would rule out the main hypothesis? This is what turns a
notebook entry into a useful research log — always include it unless the
issue is a pure coding bug with no hypothesis.>

### Resolution
_(fill in with commit hash + one-line explanation when resolved)_
```

**Title style:** short, specific, avoids generic words. Prefer
"Goda & MF disagree on RW009 at widest spacing" over "reflection bug".

**Category cheat-sheet:**
- `coding` — Python bug, refactor, tooling, CLI UX
- `pipeline` — processing-stage convention (clipping, detrend, FFT, averaging)
- `physics` — dispersion, spectral conventions, sign/scaling, tank acoustics
- `research-target` — open question or goal driving the work
- `confusing-phenomenon` — observation you can't yet explain

Pick **one** primary category label; add a second only if the issue clearly
straddles two domains.

## Step 3 — Present for review

Show the user:
1. Suggested **title**.
2. Suggested **labels** (category + any status other than `open`).
3. The full body as a fenced markdown block.

Then stop and wait. Do **not** call `gh issue create` until the user says
"push it" / "go ahead" / equivalent explicit approval. If they say "tweak
X", revise inline and re-present — don't push a stale draft.

If the user says they'll edit later in the GitHub UI and just wants it up,
that counts as approval — push the current draft.

## Step 4 — Push

Write the body to a temp file first (avoids heredoc-quoting hazards with
backticks and `$` in the content), then create the issue, then delete the
temp file.

```bash
GH="$LOCALAPPDATA/Microsoft/WinGet/Packages/GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/gh.exe"
REPO="RyanY2021/Wave_Reflection_Coefficient"

# Body written via the Write tool to .issue_body.md (gitignored implicitly
# by the leading dot + not committed). Prefer Write over heredoc.

"$GH" issue create \
  --repo "$REPO" \
  --title "<title>" \
  --label "<category>" [--label "<secondary>"] \
  --body-file ".issue_body.md"

rm ".issue_body.md"
```

If `gh` is on PATH in the current shell, use it directly; the full-path
fallback above is for the common case where `winget` installed it but PATH
hasn't refreshed in the active shell.

After push, report the issue URL returned by `gh` back to the user in one
line. Don't summarise the body — they just reviewed it.

## Labels available in this repo

**Categories:** `coding`, `pipeline`, `physics`, `research-target`,
`confusing-phenomenon`
**Non-`open` statuses:** `investigating`, `blocked`, `wontfix`
**GitHub defaults still present:** `bug`, `enhancement`, `documentation`,
`question`, `duplicate`, `good first issue`, `help wanted`, `invalid`

If a new label is genuinely needed, create it first with
`gh label create <name> --color <hex> --description "..." --repo "$REPO"`,
then apply it. Don't silently spawn ad-hoc labels.

## What this skill does NOT do

- It does not update `docs/lab_notebook.md` or `docs/issues_index.md`. Those
  pre-date the move to GitHub Issues; treat GitHub as the live tracker and
  leave the markdown log as historical record unless the user says otherwise.
- It does not comment on existing issues, close issues, or triage labels on
  pre-existing issues. Those are separate operations — handle them directly
  with `gh` rather than invoking this skill.
- It does not push commits. Issue creation and code commits are independent
  flows.
