# CLAUDE.md

All project rules live in `AGENTS.md`, shared with Codex. Edit them there, not here.

@AGENTS.md

## Claude Code specifics

- On this owner's Windows machine, `git` and `gh` may not be on the shell PATH:
  use `C:\Program Files\Git\cmd\git.exe` and `C:\Program Files\GitHub CLI\gh.exe`.
  Node.js is not installed locally, so the Node frontend tests run only in CI;
  say so rather than implying they passed.
- Prefer the built-in browser preview to check interface changes, and re-run
  `tools/ui_audit.js` in the page after visual changes.
- Commit trailer: `Co-Authored-By: Claude <noreply@anthropic.com>`, or the
  exact trailer your session instructions specify.
