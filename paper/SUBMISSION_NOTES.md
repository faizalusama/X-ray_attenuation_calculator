# JOSS submission notes

Prepared 27 October 2026. This is an author preparation record; it is not part
of the paper and does not certify submission eligibility or acceptance.

**Current status: editorial refinement and bounded numerical checks are
complete; actual research use and the author's personal review
have not yet been established. Do not submit until the confirmations below
are resolved.**

## Manuscript format

The current [JOSS paper format](https://joss.readthedocs.io/en/latest/paper.html)
specifies 750-1750 words and six required content sections: Summary, Statement
of need, State of the field, Software design, Research impact statement and AI
usage disclosure. Retain Acknowledgements and References as applicable. Keep
installation instructions and API reference material in the software
documentation rather than expanding the paper into a manual.

## Author confirmations still needed

- Confirm the author list, author consent, ORCID and exact institutional
  affiliation. The draft currently lists Mohamed Faizal Ussama Jalaludeen,
  ORCID `0000-0003-0077-691X`, Aalborg University, Denmark.
- Supply a specific example of actual research use: the study or ongoing
  analysis, the software's role, and the exact version or commit used. Include
  an appropriate publication or preprint citation if available. Private work
  may be described to the editors without exposing unpublished results. Do
  not imply external adoption or a publication that has not been established.
- Confirm funding, acknowledgements and any conflicts of interest; absence of
  supplied details is not evidence that there are none.
- Complete the AI disclosure with the actual tools/models and versions used
  in code, tests, documentation and manuscript preparation. Confirm which
  outputs the human author reviewed, edited and validated and which core
  design decisions were human decisions. Do not sign a review assertion
  before that review has occurred.
- Check for any remaining explicit draft markers before submission.

The [current submission policy](https://joss.readthedocs.io/en/latest/submitting.html)
also screens public development history, demonstrated research use and open
development practices. Verify the public timeline and evidence of iteration;
old commit timestamps alone do not establish when a repository became public.
The editors decide whether the evidence meets JOSS's requirements.

JOSS permits disclosed AI assistance in software and paper preparation. Its
current policy prohibits AI-authored conversational interactions with editors
or reviewers except for translation. The human author should handle those
interactions and remains accountable for the submission.

## Reproduce and inspect the PDF

The **JOSS paper preview** workflow compiles `paper/paper.md` using the official
Open Journals action. It runs for changes under `paper/`, changes to the
workflow, pull requests touching those paths, or a manual workflow dispatch.
Download the `joss-paper-preview` artifact from a successful Actions run and
inspect every page, especially citations, mathematical notation, headings and
author metadata. Compilation success does not establish scientific correctness
or JOSS eligibility.

The action revision is pinned, but its inspected implementation uses the
official `openjournals/inara:latest` container. Keep the build log when an exact
compiler environment matters. The workflow uploads only a temporary build
artifact; it does not commit a PDF, create a release or submit to JOSS.

For a local build with Docker, run this from the repository root in PowerShell:

```powershell
docker run --rm --volume "${PWD}/paper:/data" --env JOURNAL=joss openjournals/inara
```

The resulting file is `paper/paper.pdf`. See the
[official compilation instructions](https://joss.readthedocs.io/en/latest/paper.html#checking-that-your-paper-compiles)
for supported alternatives. Do not commit generated drafts accidentally.

## Software archive and reviewed revision

The existing version-specific DOI `10.5281/zenodo.22878495` identifies the
archived **1.0.0 release**. The concept DOI `10.5281/zenodo.17602370` identifies
the software across versions. Neither DOI, by itself, identifies subsequent
unreleased commits on a review branch. State the reviewed revision accurately
and cite the exact archived release used for a research result. A new archive
should be made only when the owner authorizes the appropriate release; do not
retag or overwrite the existing archive.

The sources include `paper.md`, `paper.bib` and the software repository.
Publication of a software archive, a successful preview build and preparation
of this manuscript are separate from submitting a paper to JOSS. No journal
submission is performed by this workflow.
## Status after author clarification (27 October 2026)

The author confirmed that the current software revision has **not yet been
used in the research manuscript**, and that the AI-assisted code/calculations
have **not yet been personally reviewed**. Accordingly, the revised paper is
a submission-preparation draft, not an assertion of submission readiness.
No research publication, external adoption or completed human validation is
invented. The supplied DOI `10.5281/zenodo.22878495` resolves to the software
archive in DataCite metadata, not a research manuscript.

Before submission, complete a real research application and describe its
material system, scientific task, exact software revision, inputs, and output
checks. Read `CALCULATION_REVIEW.md` for bounded numerical evidence. AI review
and automated checks cannot supply the author's required human-review
declaration. Personal details and funding remain for the author to add.

