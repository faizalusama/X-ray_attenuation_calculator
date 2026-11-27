# AGENTS.md — shared instructions for every coding agent

This file is the **single source of truth** for AI agents working in this
repository. Codex reads it directly; Claude Code reads it through `CLAUDE.md`,
which imports it. Edit the rules here, never in a copy, so both agents always
follow the same ones.

## What this project is

The X-ray Attenuation Workbench computes narrow-beam X-ray attenuation in
multicomponent glasses, ceramics and planar multilayers. It is **scientific
software with a published DOI**: concept DOI 10.5281/zenodo.17602370, v1.0.0
DOI 10.5281/zenodo.22878495. A wrong number that looks plausible is worse than
a crash. The owner is Mohamed Faizal Ussama Jalaludeen (GitHub `faizalusama`).

Layout: `xray_workbench/physics.py` (engine), `xray_workbench/backends/`
(attenuation data sources), `xray_workbench/materials/` (materials library with
provenance), `xray_workbench/server.py` (local FastAPI app), `static/` (browser
UI), `tools/` (maintainer scripts), `tests/`, `docs/`, `legacy/` (archived
original — never edit or calculate with it).

## Hard rules

1. **Never push to `main`, merge pull requests, create tags or publish
   releases.** A release on `main` mints a Zenodo DOI. Work on a branch and open
   a pull request; the owner merges.
2. **Physics numbers must not move silently.** If a change alters any computed
   output, say so in the PR title, explain the physics, and cite the source.
   `docs/VALIDATION.md` records reference values (e.g. spectral transmission
   0.7802626916899833 for `examples/borosilicate-multilayer.json`); they must
   stay bit-identical unless the change is deliberate and documented.
3. **Never widen a tolerance to make a test pass.** A failing reference
   comparison is evidence. Investigate it.
4. **Never invent data or citations.** Materials enter the library only through
   the provenance tiers in `docs/MATERIALS.md`. A literature entry needs a
   Crossref-verified DOI (`python tools/verify_doi.py`), a table or page
   locator, and a named person who checked the values. A density without a
   checked source stays `unverified`.
5. **Never average or blend data sources.** Report disagreement. xraylib is the
   same evaluated data as Elam (identical within floating-point rounding), so it
   is never evidence of cross-database validation.
6. **Units are explicit:** keV, mm normal thickness, g/cm³, degrees from the
   surface normal. Convert once, at a named boundary, with a comment.
7. **Assert the reason:** `pytest.raises(ValueError)` without `match=` is not
   accepted; the engine raises ValueError for every invalid input.
8. **The version lives only in `xray_workbench/__init__.py`.**
9. **Do not reformat existing code.** `ruff format` is intentionally not used;
   match the surrounding compact style.

## Before you commit

```
python -m pytest
python -m ruff check .
python -m mypy
python tools/stamp_assets.py        # after editing anything in static/
python tools/manifest.py write      # after any intentional file change
python tools/manifest.py verify
```

`python tools/ui_audit.py` audits the running interface (needs Playwright). CI
runs everything above on Windows, Linux and macOS, plus the interface audit and
the frontend Node tests. **Do not open a PR with red CI** unless the PR exists
to fix CI.

Static assets are cached by browsers for a year as immutable, keyed on their
`?v=` stamp. The stamp is the file's content hash, set by
`tools/stamp_assets.py`; a test fails if any stamp is stale. Never hand-edit
stamps.

## Interface standards

WCAG 2.2 AA is the floor, and the interface audit enforces it in CI: text
contrast at least 4.5:1 (3:1 for large text), no text below 12 px, targets at
least 24×24 px, every control named, no horizontal overflow at 375 px. Status
must never be carried by colour alone. `static/readability.css` is the
accessibility layer: prefer fixing the base stylesheets and deleting the
matching override there, rather than adding new overrides.

## Working alongside another agent

Claude Code and Codex both work on this repository, often on the same days.

- **Start every session with `git fetch --all`** and read what changed since
  your last work. Never assume the tree is as you left it.
- **One branch per task.** Prefix it with the agent: `claude/<topic>` or
  `codex/<topic>`. Never commit to another agent's branch; open a PR into it or
  ask the owner.
- **Claim work before starting it** by adding a line under *In progress* in
  `docs/IMPROVEMENT_BACKLOG.md` on your branch (agent, branch, date). Pick only
  unclaimed items.
- **Shared hot spots:** `static/app.js`, `static/dashboard.js`,
  `static/index.html` and `xray_workbench/physics.py` are touched by both
  agents. Keep changes there small and say in the PR what you changed so the
  other agent can rebase.
- **Hand-offs:** if you stop mid-task, leave a short note at the top of the PR
  description: done, not done, and what the next agent should check first.
- Git identity for commits made on the owner's behalf: `faizalusama`,
  `70561810+faizalusama@users.noreply.github.com`. Add your own agent's
  co-author trailer.

## Where to look

- `docs/SCIENTIFIC_METHODS.md` — the physics and its limits
- `docs/VALIDATION.md` — what has been verified, how, and what has not
- `docs/MATERIALS.md` — materials library tiers and how to add entries
- `docs/ROADMAP.md` — planned research work
- `docs/IMPROVEMENT_BACKLOG.md` — the queue of engineering and interface work
- `CONTRIBUTING.md` — the same rules, written for human contributors
