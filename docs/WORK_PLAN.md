# PR7 coordinated work plan

Updated: 13 November 2026. Integration branch: claude/web-version. PR: #7 (draft).
Coordinator: Codex. Worker limit: three active at a time. No main merge, tag,
release, Pages enablement, public deployment or repository setting changes.

## State inspected before planning

Fetched all branches. PR7 starts at add823c. Open PRs #2 (accessibility),
#3 (citation), #4 (paper), #7 (candidate) are separate scopes. Existing Claude
routine branches include library-origin and XrayDB-4.6 compatibility work.
Do not duplicate or silently integrate these. The original temporary integration
checkout has missing source files; it is preserved. A durable coordinator checkout
is at outputs/xray-pr7-coordinator. The preview copy is outputs/preview-1.1.0.

Read AGENTS.md, HANDOFF_PR7.md and the backlog. Handoff items 1–15 are ALL
implemented already, verified against the branch source and completion record.
They are SKIPPED_DONE, not queued again. No unbuilt handoff feature was found.
This round changes presentation and research evidence, not scientific numbers.

## Ownership and review protocol

Worker source-file allowlists below are disjoint. Coordinator alone owns this plan,
the shared backlog, MANIFEST.sha256, generated asset stamps/dark.css, commits,
integration merges and pushes. Workers do not edit generated artifacts or commit
before coordinator review; they leave edits in their own branch/worktree and
report readiness. Coordinator reviews scope/diff, generates artifacts, runs all
checks, commits the reviewed branch, then merges it into claude/web-version.
Generated artifacts are resealed after each merge, resolving generated-only
conflicts by regeneration rather than changing source behavior.

Every reviewed branch must pass full pytest, ruff, strict mypy, Node suites,
manifest verification, and the native digest:
4d50d8cd05b54d9c0251e633a5d803b6ced194e1ec3eb4c5134ee02fdb5372e4.
New scientific numbers require independent reference tests and verified DOI
evidence; no scientific values are planned here. Standards without a DOI use
their official versioned URL, with no invented identifier.
Never widen tolerance, alter data, or erase user edits to make a branch pass.
Reject or return a failing branch for repair and record the concrete reason.
Integrate passing branches one at a time, rerun checks, push and check PR7 CI.

## L1 — Workspace layout and progressive disclosure

Status: MERGED
Branch: codex/layout-workspace
Worktree: outputs/xray-task-layout
Goal: visibly group tabs into Attenuation, Scattering and Reference, put the
configuration panel on the left with accessible collapse/reopen controls, introduce
Simple and Advanced modes without changing any input values, and provide a
prominent materials-library entry point.
May touch: static/index.html, static/app.js, static/workspace.js,
static/workspace.css, tools/build_dark_css.py, tests/test_workspace.cjs.
Constraints: preserve existing tab IDs and keyboard behavior; modes must not
silently reset or disable active scientific settings. Hidden invalid/advanced
settings must remain discoverable. Use theme-safe styles and phone layouts.
Checks: focused Node tests, existing frontend tests, full review gate, final UI audit.
Dependencies: none. Owner: Codex worker L1 (Anscombe; 01a1d5e3-fd0d-7b33-9fba-1a417c92b839).

Reviewed commit: 19af369. Source allowlist respected; 826 Python and 70 Node tests,
ruff, mypy, manifest and exact digest passed. Unchanged interface audit passed
in installed headless Edge; desktop/phone screenshots reviewed. Chromium restored
for the final combined audit. Initial missing Chromium was an environment gap,
not a waived assertion or accepted source defect.

## L2 — Four-plot initial dashboard

Status: MERGED
Branch: codex/compact-dashboard
Worktree: outputs/xray-task-dashboard
Goal: show four useful plots on a fresh workspace and make Show all restore all
available plots. Preserve explicitly saved layouts and scientific/plot settings.
May touch: static/dashboard.js, tests/test_dashboard.cjs, tools/ui_audit.py.
Constraints: no new physics or changed plotted data. Adapt the audit to exercise
both four-plot startup and Show all rather than merely lowering an assertion.
Checks: default/saved/custom/Show-all Node regressions, full review gate, audit.
Dependencies: none. Owner: Codex worker L2 (Ptolemy; 01a1d5e3-fea6-7191-9093-4f274a3c98eb).

Reviewed commit: d5846f2. Full gate: 826 Python tests, 62 Node tests, ruff,
mypy, manifest and exact digest passed. Existing live audit passed in Chromium
with four-plot startup plus all-eleven expansion, light/dark and phone coverage.

## R1 — Source and benchmark evidence only

Status: IN_PROGRESS
Branch: codex/research-validation
Worktree: outputs/xray-task-research
Goal: gather and verify primary references relevant to the layout and existing
scientific capabilities, including DOI metadata and independent benchmark
provenance. Flag unsupported claims or candidate data for human checking.
May touch: docs/research/PR7_SOURCE_EVIDENCE.md,
docs/research/pr7-source-evidence.json.
Constraints: research only; no changes to UI, physics, materials, reference values
or manuscript. Never label literature data human-verified without the required
named reviewer. Do not duplicate the JOSS/citation PRs or active routine work.
Checks: resolve DOIs against Crossref/DataCite/primary records, link exact locators,
validate JSON, full review gate. Evidence is not experimental validation.
Dependencies: none. Owner: Codex worker R1 (Wegener; 01a1d5e3-ff04-7692-a15e-b5fe583196f5).

## A1 — Audit the combined result

Status: PREPARING_AUDIT (final execution waits for L2/R1 integration)
Branch: codex/final-layout-audit
Worktree: outputs/xray-task-audit
Goal: audit the actual integrated layout, modes, library entry, collapse/reopen,
four plots/Show all, exports and tab keyboard behavior in light/dark themes at
desktop, tablet and 375px phone width, including populated results.
May touch: tools/workspace_ui_audit.py, tests/test_workspace_audit.py,
.github/workflows/ci.yml (wire the additional live audit into CI).
Constraints: do not edit implementation to mask a failure. Report reproducible
defects to coordinator/L1/L2 for repair. Preserve all existing audit coverage.
Checks: full review gate, existing interface audit, added interaction audit,
manual screenshot review, actual browser/native parity and PR7 CI.
Dependencies: L1 merged; L2/R1 must merge before final execution.
Owner: Carver, 01a1d60f-3c0a-7193-af85-aeffab93363f. Independent audit code
preparation is permitted now; coordinator runs it only on the combined result.

## C1 — Review existing XrayDB 4.6 test compatibility patch

Status: MERGED
Branch: codex/xraydb46-test-compat
Worktree: outputs/xray-task-compat
Owner: coordinator reviewing an already completed Claude contribution; no fourth worker.
Goal: repair the newly reproduced CI failure reading compressed XrayDB 4.6 table
columns. Reuse only the completed test-only change from Claude commit 16ecfcb;
do not duplicate its implementation or merge its unrelated branch history.
May touch: tests/test_physics.py only (coordinator reseals generated metadata).
Checks: existing edge-knot assertions unchanged under XrayDB 4.5.8 and 4.6.1,
full review gate, unchanged complete 16-case digest under both, PR7 CI.
Reviewed commit: 9e3cac7. Full reference gate passed (826 tests, 58 Node);
109 physics tests and the exact full digest also passed under XrayDB 4.6.1.
Reason added: current unpinned CI installs XrayDB 4.6.1 and fails decoding plain
JSON in the test; production engine and browser parity already pass.

## Daily coordinator protocol

Existing scheduled task improve-x-ray-workbench-daily updated and ACTIVE at
18:00 Europe/Copenhagen in this chat (replaces its earlier 07:00 time).

Daily at 18:00 Europe/Copenhagen, read this plan and current repo instructions.
Fetch and inspect PR7, branches, open PRs and active claims. Review waiting
branches first. If three or more branches are awaiting review, start no workers:
report the queue and stop. Otherwise skip DONE, MERGED, SKIPPED_DONE, IN_PROGRESS
and BLOCKED items. Start at most three eligible tasks per run, never more than
three concurrent workers, each in a separate worktree and codex/<task> branch
based only on claude/web-version. Apply the full branch review and serial
integration procedure above, then update this plan and report results.
If all items are complete, inspect for regressions and record justified follow-up
candidates; do not invent changes or start unrelated work.

## Integration and rejection record

- Baseline gate passed: 826 Python tests, 58 Node tests, ruff, strict mypy, manifest and exact native digest. Plan committed as 2de66a6; worker round 1 started.
- C1 accepted: reviewed test-only patch from Claude 16ecfcb, port 9e3cac7; decoder compatibility only, no assertion/tolerance/production change.
- L1 accepted: 19af369; grouped tabs, left collapse/reopen, modes preserve settings, library entry visible. Full gate and live audit passed; source and screenshots reviewed.

## Preview and publishing

Update outputs/preview-1.1.0 to the tested integration revision only if clean;
preserve local edits otherwise. Preview locally using its START_WINDOWS.cmd or
python launch.py; public hosting remains unapproved. A preview update is not a
release. Keep #7 in draft. Paper/JOSS review and publishing remain owner actions.

Coordination note: L2/R1 tool approvals stalled with no edits; workers redirected
to supply exact patches/content for coordinator application and checks. No source
branch has been rejected. L1 PR7 CI is green at 77d1977. Browser parity compares
247,930 values within the unchanged tolerance (maximum relative error 5.46e-16).

- L2 accepted: d5846f2. Four useful startup plots; explicit saved layouts, ranges,
  typography and visibility preserved. Generated-only index stamp/manifest merge
  conflicts resolved by regeneration; L1 source layout retained.
