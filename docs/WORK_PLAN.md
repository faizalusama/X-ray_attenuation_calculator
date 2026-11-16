# PR7 coordinated work plan

Updated: 16 November 2026. Integration branch: claude/web-version. PR: #7 (draft).
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

Status: MERGED
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

Reviewed commit: 86f5288. Saved JSON and five reference metadata records match
Crossref exactly. Full gate passed: 826 Python tests, 58 Node, ruff, mypy,
manifest and exact digest. Primary-source limitations and the Chantler page-range
discrepancy remain explicitly recorded; no production scientific data added.

## A1 — Audit the combined result

Status: MERGED
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
Dependencies: L1/L2/R1 merged; final execution must use the combined result.
Owner: Carver, 01a1d60f-3c0a-7193-af85-aeffab93363f. Independent audit code
preparation is permitted now; coordinator runs it only on the combined result.

Reviewed commit: d7f30bf. Combined-result gate passed: 831 Python tests,
14 subtests, 76 Node tests, ruff, strict mypy, manifest and exact native digest.
Both live audits passed. The added audit covers eight theme/viewport scenarios
at 1440, 768, 375 and 320 px, with no failed checks or browser errors. Existing
audit coverage is retained. Screenshots were reviewed; automated contrast checks
still report their explicit visual-review limits, not full WCAG certification.
Audit findings F1/F2/F3 were repaired in their own reviewed branches below.

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

- R1 accepted: 86f5288. All five Crossref records verified by coordinator;
  primary locators and test mappings documented without new numerical data.
  Unresolved original-table/human review limits are retained, not waived.

## F1 — Audit-discovered mode button state correction

Status: MERGED
Branch: codex/workspace-mode-button-state
Worktree: outputs/xray-task-mode-state
Owner: coordinator; no additional worker.
Goal: mode pressed-state and click binding apply only to mode buttons, never
to the page root which also carries data-workspace-mode for styling.
May touch: static/workspace.js, tests/test_workspace.cjs.
Checks: real DOM root has no aria-pressed, mode buttons retain correct state;
Node regression, full gate, unchanged digest, final interaction audit.
Reason: live audit found ambiguous mode selectors and invalid aria-pressed on
html. The audit locator is corrected separately; production state leak needs F1.

- F1 accepted: d0d60f5, full gate826Python+75Node passed with exactdigest.
  The final audit initially rejected selector ambiguity, lint/import typing and
  an assumed spectrum table; coordinator is repairing those audit defects against
  the real DOM, retaining the intended data/focus checks. No branch rejected.

## F2 — Audit-discovered dashboard button state correction

Status: MERGED
Branch: codex/dashboard-button-state
Worktree: outputs/xray-task-dashboard-state
Owner: coordinator.
Goal: layout/gesture pressed-state must apply only to their actual buttons, not
to the plot-grid container sharing data-layout.
May touch: static/dashboard.js. Checks: full gate and final real-browser assertion
that non-button styling containers carry no aria-pressed; digest unchanged.

- F2 accepted: 917bc16, full gate826Python+75Node passed; only button
  selectors changed. Native scientific digest remains unchanged.

## F3 — Repair phone library legend clipping

Status: MERGED
Branch: codex/library-legend-reflow
Worktree: outputs/xray-task-library-reflow
Owner: coordinator.
Goal: wrap long library chart legend names at narrow widths without clipping,
truncation, smaller fonts or changed plotted data.
May touch: static/materials.js, tests/test_materials.cjs.
Checks: safe label wrapping regressions, full gate, exact digest; unchanged final
320/375px library-detail bounds assertion and screenshot review.
Reason: 320px audit failed in both themes; focused screenshot confirmed total
legend text extends beyond its SVG and is cut off. Not a waived audit finding.

- F3 accepted: b52f3cd, full gate826Python+76Node passed. Complete labels
  are safely escaped and wrapped using available chart width. Plot arrays,
  fonts, densities and reference outputs remain unchanged. Final A1 rerun passed.

## Completion and remaining review

L1/L2/R1/C1/F1/F2/F3/A1 are complete. No branch was rejected or left waiting
for review; reproduced defects were fixed before acceptance. All 15 handoff
features were already implemented and were not duplicated. No new scientific
numbers, materials or numerical tolerances were introduced in this round.
The final merge repeats the full gate; PR7 CI must also pass on that commit.

Owner review of the unpublished candidate remains open. Source evidence records
the original-table/human-review limits and Chantler bibliography discrepancy;
these are research follow-ups, not verified production data. Daily runs should
inspect for justified regressions or evidence-backed follow-ups before claiming
new tasks, using the queue limits above. Publishing remains unapproved.

## Daily inspection — 15 November 2026

Fetched all branches and confirmed the expected GitHub remote, clean coordinator
checkout and local/remote integration revision 1e3e570. PR7 is still open and draft.
CI run 43594965929 passed all 11 jobs on that revision, including native/browser
parity and both live interface audits. No new source changes or review branches
were found. All coordinator task branches are already integrated; the waiting
worker queue is zero. PRs #2/#3/#4 and Claude routine branches remain separate.

No eligible tasks: zero workers started, zero feature branches accepted or
rejected. The original 15 handoff features and this plan's implementation tasks
remain complete. This run records status only; scientific outputs and UI behavior
are unchanged. The preview checkout was clean at the same tested revision.

Next follow-ups remain the documented primary-table/named-human source review
and cross-browser/physical-device validation gaps. No unverified literature data
is eligible for import. Owner review of the draft candidate remains in progress;
publishing is not approved. The documentation-only update uses the required
local commit gate and PR7 CI; its live audit results are retained with that run.

## Daily inspection — 16 November 2026

Fetched current branches and checked open PRs again: integration remains 1e3e570
with green CI run 43594965929. The only local changes were the coordinator's
unfinished 15 November inspection note and its manifest entry; no user or source
edits were overwritten. This run finishes that documentation update using the
required verification gate. No worker branches await review and no tasks are
eligible; zero workers, merges or rejections. Separate PRs and Claude routine
branches remain untouched. The next eligible work still depends on the source
review or additional browser/device evidence described above. The candidate
remains unpublished and in owner review.

## Synchrotron development

The owner requested continued development across total scattering, SAXS, WAXS,
tomography and related techniques. Daily runs may implement the READY items
below in order of dependency once an independent benchmark has been identified.

- S1, IN_PROGRESS, codex/detector-q-coverage. Plan SAXS/WAXS Q coverage from
  measured detector radii, distance and energy. Own detector_coverage.py in the
  package and its namesake test. Check against an independent geometry reference.
- S2, IN_PROGRESS, codex/tomography-projection. Convert one flat/dark/sample
  count triplet to transmission and a projection. Own tomo_projection.py in
  the package and its namesake test. Compare with reference normalization,
  including invalid and above-flat counts.
- S3, IN_PROGRESS, codex/synchrotron-interface. Connect both tools to the
  shared API and existing exports. Own scattering_tools.py, static/index.html,
  static/scattering-tools.js, static/scattering.css, tools/ui_audit.py,
  tools/web_parity.py and docs/SCATTERING_TOOLS.md. Exercise populated tools,
  both themes, phone widths and native/browser entry points.
- S4, READY, codex/pdf-q-window-study. Compare two selected Qmax cutoffs on
  the same normalized S(Q) with the existing window choice, reporting both
  effective endpoints. Own a new pdf_window_study module and tests. Validate
  finite-range transforms against analytic integrals before UI integration.
- S5, READY, codex/saxs-guinier. Fit a user-selected low-q interval of
  background-corrected I(q), report Rg, I(0), selected range and qRg bounds.
  Own a new guinier module and tests. Verify a published benchmark and the
  assumptions on shape, aggregation and uncertainty before enabling the fit.
- S6, READY, codex/tomo-sampling. Estimate angular sampling for parallel-beam
  tomography from object support and requested resolution. Own a new
  tomo_sampling module and tests. Verify the original sampling convention
  and an independent example, distinguishing detector pitch and resolution.

S4–S6 integrations must include user controls, export, parity and accessibility
checks. Start shared interface edits serially after the independent modules.
