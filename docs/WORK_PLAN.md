# PR7 coordinated work plan

Updated: 24 November 2026. Integration branch: claude/web-version. PR: #7 (draft).
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

- S1, MERGED, codex/detector-q-coverage. Plan SAXS/WAXS Q coverage from
  measured detector radii, distance and energy. Own detector_coverage.py in the
  package and its namesake test. Check against an independent geometry reference.
- S2, MERGED, codex/tomography-projection. Convert one flat/dark/sample
  count triplet to transmission and a projection. Own tomo_projection.py in
  the package and its namesake test. Compare with reference normalization,
  including invalid and above-flat counts.
- S3, MERGED, codex/synchrotron-interface. Connect both tools to the
  shared API and existing exports. Own scattering_tools.py, static/index.html,
  static/scattering-tools.js, static/scattering.css, tools/ui_audit.py,
  tools/web_parity.py and docs/SCATTERING_TOOLS.md. Exercise populated tools,
  both themes, phone widths and native/browser entry points.
- S4, MERGED, codex/pdf-q-window-study. Compare two selected Qmax cutoffs on
  the same normalized S(Q) with the existing window choice, reporting both
  effective endpoints. Own a new pdf_window_study module and tests. Validate
  finite-range transforms against analytic integrals before UI integration.
- S5, MERGED, codex/saxs-guinier. Fit a user-selected low-q interval of
  background-corrected I(q), report Rg, I(0), selected range and qRg bounds.
  Own a new guinier module and tests. Verify a published benchmark and the
  assumptions on shape, aggregation and uncertainty before enabling the fit.
- S6, MERGED, codex/tomo-sampling. Estimate angular sampling for parallel-beam
  tomography from object support and requested resolution. Own a new
  tomo_sampling module and tests. Verify the original sampling convention
  and an independent example, distinguishing detector pitch and resolution.

S4–S6 integrations must include user controls, export, parity and accessibility
checks. Start shared interface edits serially after the independent modules.

Reviewed S1 (3dc9c35), S2 (9e63f75) and S3 (51b5bd7). Reference tests use
pyFAI geometry and TomoPy normalization/log conventions. Crossref confirmed
all three scientific DOIs in the new modules and tests. In the integrated
interface 859 Python and 76 Node tests passed. Native/browser parity compared
248,547 values with a maximum comparison metric of 5.46e-16. Native digest
matches the reference. Populated detector, projection and keyboard-operable
technique guide passed the desktop/tablet/phone audit in light and dark.
Next eligible tasks are S4–S6. No branches rejected.

## D1 — Live technique schematics

Status: MERGED. Branch: codex/live-technique-schematics.
Worktree: outputs/xray-task-schematics. Owner: Codex, 17 November 2026.
Goal: parameter-linked detector geometry and tomography count diagrams with
keyboard sliders, validated-result probes, exact readouts and SVG export.
Files: static/technique-schematics.js, scattering-tools.js, scattering.css,
index.html; server.py download allowlist; tests/test_scattering.cjs;
tools/ui_audit.py, tools/workspace_ui_audit.py and CI syntax checks.
Coordinator maintains generated stamps, manifest and these status files.
Checks: full review gate, populated light/dark interaction audit at
1440/768/375/320px, invalid-input recovery, safe SVG export and unchanged
native digest. Drawings consume the existing science responses.

Reviewed fe843f4. The native digest matches the reference; 859 Python and
78 Node tests pass with ruff, strict mypy, stamps and manifest. Both live
audits pass, including 1440/768/375/320px light/dark interaction checks,
keyboard sliders, stale-result clearing, above-flat and invalid-count
recovery, SVG downloads and normal tool switching. The local development
preview is on port 8767. Initial asset-stamp failure during editing was
corrected by regeneration and the complete gate was rerun successfully.

Follow-up D2, MERGED, codex/optics-schematic: grazing-angle geometry and
penetration-depth probe using the existing optical response, with numeric
equivalents for keyboard users. Follow-up D3, MERGED, codex/pdf-schematic:
link the selected transform window and Q range to the existing PDF curves.
Each follow-up owns its new diagram module and tests, integrating shared
controls only after prior branches have been reviewed.

## Daily run — 17 November 2026

Starting at clean 7a8e640, CI run 44068305310 passed all 11 jobs.
Fetched branches and inspected PRs #2/#3/#4/#7; no waiting coordinator
branches. D2 and D3 started in separate worktrees, each restricted to its
new diagram module and Node test. Coordinator owns shared controller, index,
styles, audit and CI integration on codex/schematic-extension (I2), the third
task. Native calculations and returned result arrays are reused.
Acceptance: full branch gates, safe exports, keyboard probes, actual result
correspondence and existing populated light/dark desktop/phone audit coverage.
S4–S6 remain eligible after these tasks. No source data is being added.

D2 accepted as e17d612, D3 as 4cb21c1; integrated serially in 0e2d76a and
df28739 after each full gate and unchanged native digest. I2 accepted as
bef4bff, with both new modules wired into the browser, exports and audits.
859 Python and 82 Node tests pass, plus ruff, strict mypy and the manifest.
Both live audits pass. The combined interaction audit matched both new probe
values against downloaded scientific results in all eight theme/viewport
scenarios, with zero failed checks and browser errors. No branches rejected;
no waiting coordinator queue. The final pushed revision is checked through
PR7 CI before this daily run completes.

The two workers produced no files or handoff after follow-up, so they were
closed before implementation and the coordinator completed their disjoint
modules. All three task limits and worktree isolation were retained.
The regular preview remains the tested detached copy in outputs/preview-1.1.0.
S4–S6 are the next eligible tasks. Publishing remains unapproved.

## D4 — Workbench design and 3D detector

Branch: codex/workbench-design. Source scope: static styles, shell, plots
and schematics, plus browser checks and scientific interface documentation.
The detector uses an orthographic projection of ideal detector coordinates.
Camera movement preserves the selected physical ray, calculated quantities
and original side view. The PDF preview draws existing transform data.
Checks cover the full test gate, native digest, browser interaction and
accessible contrast on desktop, tablet and phones in both themes.

Reviewed dee9640. Passed 859 Python and 82 JavaScript tests, lint, types,
manifest and the reference native digest. Both interface audits pass, with
camera and keyboard checks down to 320 px. Corrected the first audit's
contrast findings in target values, camera labels and the materials list.
The 3D view and PDF previews read the existing calculations.

## Daily run - 19 November 2026

Clean starting revision ef66328. CI run 44161471923 passed. The open PRs
remain #2, #3, #4 and draft #7. There are no worker branches awaiting review.
The three tasks for this run are S4 on codex/pdf-q-window-study, S5 on
codex/saxs-guinier and S6 on codex/tomo-sampling. Each owns only its new
Python module and namesake test; S5 may also own its benchmark fixture.
The coordinator owns dispatch, web controls, documentation and audits, and
will integrate them serially after independent validation.

- S4 ec36476 accepted. Tests independently compare both finite sine integrals
  with and without Lorch to Wang et al. (DOI verified). 883 Python and 82
  Node tests, Ruff, strict mypy, manifest and native digest passed. Shared UI
  and browser parity checks cover the comparison and both exports.
- S5 9283d8a accepted. Weighted and unweighted fits match independent
  regressions on the licensed, attributed SASDUE4 consensus curve. Crossref
  confirmed the paper DOI and the report distinguishes the fixed test window
  from the published AUTORG result. 905 Python and 82 Node tests, Ruff, strict
  mypy, manifest and native digest passed.
- S6 1e21025 accepted. The Crowther estimate follows the verified Du and
  Crowther references. The independent ForMAX example checks the exact
  ceiling at the reported pixel count. Pixel support and angular shortfall
  are separate. All Python and Node suites, Ruff, strict mypy, manifest
  and native digest passed.
- All three worker branches passed. No rejected or waiting branches.
  Fixed a lint error in the coordinator's export audit assertions before
  sealing the branches. On a 375 px page, edits recalculated all three
  tools and keyboard-triggered downloads carried the new settings.
- Combined tests passed: 924 Python tests, 82 Node tests, Ruff, strict mypy,
  asset stamps and file manifest. Native attenuation digest is unchanged.
  The Pyodide check compared 249,206 values across native and browser builds
  at a maximum comparison metric of 5.46e-16. The HTML accessibility
  audit and all eight populated workspace scenarios passed in light and dark,
  including keyboard use, modes, panel collapse/reopen, four plots/Show all,
  and the materials library. The accessibility report measures automated
  coverage and lists remaining HTML/SVG elements for visual review.
- CI: the first push (29b3803, run 44586249744) was superseded after ten
  successful jobs. See [draft PR7 checks](https://github.com/faizalusama/X-ray_attenuation_calculator/pull/7/checks)
  for the live CI of the combined revision. The latest local preview uses
  the tested candidate revision and serves only on the loopback address.

- The three-tool integration at 2d7a3fc passed all 11 CI jobs in
  [run 44587632754](https://github.com/faizalusama/X-ray_attenuation_calculator/actions/runs/44587632754).
  Visual review found that the Guinier fit line covered the measured trace.
  Accepted S5 follow-up 7ff007a shows observations as open markers, keeps
  the fit as a line, and displays the actual q and qRg bounds. The numeric
  fits are unchanged. All 924 Python and 82 Node tests, Ruff, strict mypy,
  the manifest, the native digest and the updated interface audit passed.

### Next eligible research

- S7, READY, codex/live-synchrotron-evidence. Define reference cases for
  interactive explanations of the Guinier fit interval and tomography view
  spacing. Touch only new docs/RESEARCH_LIVE_SCHEMATICS.md. Verify primary
  paper metadata and reusable data, record exact inputs and expected plot
  or geometry values, and distinguish model output from display geometry.
  Propose keyboard and narrow-screen checks before implementation.

## 21 November 2026 - three branches waiting for integration review

The daily run reached its three-branch stop threshold before dispatch. The
integration checkout, fetched draft PR7 and clean preview started at bac0d3b.
All 11 CI jobs passed on that application revision, including the interface
audit and browser/native comparison. [CI run](https://github.com/faizalusama/X-ray_attenuation_calculator/actions/runs/44588575993)

- `codex/guide-explorer` has a local commit, 0417d46. The explorer awaits
  integration review with the guide shell, topic catalogs and keyboard flows.
- `codex/guide-scattering` has 24 topic entries in its worktree's uncommitted
  `static/guide-scattering.js`. Scientific sources, links and tool targets
  still need coordinator review.
- `codex/guide-imaging` has 25 topic entries in its worktree's uncommitted
  `static/guide-imaging.js`. The same source, link and target checks are pending.

No worker started and no feature was merged or rejected in this run.
The coordinator's previously parked guide shell and diagrams are in stash
34f6a73. The live-results, guide-quality and S7 evidence worktrees
have no changes against the starting revision and are outside the review
count. PRs #2, #3 and #4 are separate. A review session can clear the queue
by checking the explorer, then each catalog, before starting further tasks.
The tested application preview is in `outputs/preview-1.1.0`.

## 21 November 2026 - technique guide review and integration

The owner requested continued work after the daily queue stop. All three
waiting branches passed review. The queue is clear.

- Explorer (0417d46) adds keyboard search and filters by technique and
  workflow stage. The shared entry point is **Technique guide** in the toolbar.
- Imaging (a22cb1b) covers tomography, grazing-incidence optics and
  attenuation in 25 topics. Review verified 11 distinct DOIs, checked linked
  controls, and revised guidance on sample motion and spectral weighting.
- Scattering (4099cff) covers total scattering/PDF, SAXS and WAXS in 24
  topics. Review verified 13 distinct DOIs and corrected PDF cutoff selection,
  Guinier input requirements and the treatment of powder peak areas.
- Tomography geometry (c5257cf) adds three camera views and three specimen
  orientations. Camera changes move the observer. Sample rotation preserves
  the positions of the beam and detector. The figure is illustrative.
- All four integrations passed pytest, Ruff, strict mypy, frontend tests,
  manifest verification and the pinned native digest. No branch was rejected.
  Shared UI checks exercise topic discovery, keyboard navigation, diagram
  changes, camera and specimen controls, and narrow viewports in both themes.
- S7 reference-case research remains available. The earlier queue record above
  describes the scheduled stop before this review. The live review status
  for the integrated guide is on [PR7](https://github.com/faizalusama/X-ray_attenuation_calculator/pull/7/checks).

## 24 November 2026 - guide accessibility and schematic reference cases

- Finished the guide integration at 261bebf. The browser audit found three
  filter labels at 10 px with insufficient contrast in both themes. They now
  render at 14 px in the theme text colour. The automated interface audit
  passed and all eight populated workspace scenarios passed, including 375 px
  and 320 px keyboard checks in light and dark. The manual Run control also
  cancels a pending automatic calculation before it can invalidate an export.
- Accepted S7 from `codex/live-synchrotron-evidence` (e87ae64).
  [The reference cases](RESEARCH_LIVE_SCHEMATICS.md) give reproducible
  inputs and expected values for the existing Guinier fit and tomography
  sampling tools. Four paper DOIs resolved, the public SAXS data matched
  the fixture, and the focused scientific tests passed. The ForMAX geometry
  example requires no projection images. Their archive reuse terms are unconfirmed.
- The research branch passed the full Python, lint, strict type, Node,
  manifest and native-digest checks. The review queue is empty. No branch
  was rejected. The separate accessibility, paper and citation PRs remain
  outside this work. [Draft PR7 CI](https://github.com/faizalusama/X-ray_attenuation_calculator/pull/7/checks)

### Interactive tools and guide expansion

- S8, ACCEPTED, `codex/guinier-sample-probe`: show the selected measured point,
  fitted intensity in log space, and signed residual. Worker owns only new
  `static/guinier-probe.js` and `tests/test_guinier_probe.cjs`. Check the
  weighted and unweighted SASDUE4 probe values, keyboard endpoints and
  invalid interval recovery against S7.
- S9, ACCEPTED, `codex/tomography-view-probe`: let users step through the minimum
  or planned half-turn schedule while reading the selected view number and
  angle. Worker owns only new `static/tomo-view-probe.js` and
  `tests/test_tomo_view_probe.cjs`. Check the ForMAX first/last angles,
  omitted plan and separate pixel-pitch status against S7.

The coordinator owns common UI wiring, styling and the combined keyboard
and phone audits for both probes. The tested preview is in `outputs/preview-1.1.0`.

- G1, ACCEPTED, `codex/technique-field-checks`: add sourced field
  checks across six techniques. Worker owns only `static/guide-field-checks.js`.
  Review source claims, linked controls, duplicate topics and search results.

Coordinator review accepted S8 (7c0d267), S9 (b5c57d6) and G1
(5b3d712). All three branches passed the test, lint, type, manifest and
native digest gates. The probes use the existing fit and sampling responses.
When the formatted angle would round to the excluded 180° endpoint, the
tomography readout shows the distance to that endpoint. The guide gained 12
field checks with 11 verified DOIs plus the NIST attenuation reference.
Topics are grouped under Plan the experiment, Collect data and Interpret
results. The combined browser audit tests keyboard endpoints, input recovery,
schedule changes, 3D geometry, and narrow layouts in both themes.
[Draft PR7 checks](https://github.com/faizalusama/X-ray_attenuation_calculator/pull/7/checks)
