# Repository Instructions

- When asked to check a GitHub issue, read the whole issue before acting: the description, all comments, linked comments, and recent owner/reporter follow-ups. Treat issue comments as part of the requirements, not optional context.

## Read first, if you have it

Fresh agents should use `docs/maintenance-handoff.md` for the current completed
work, evidence and bounded next actions, then claim a task in `PLAN.md`.
`PLAN-ARCHIVE.md` and dated handoffs are historical snapshots, not active queues.

Local operational notes, when present, are in gitignored `CLAUDE.md`. Read them
for storage/worktree/job recovery; keep actual machine paths and private account
details there. Public plans and receipts use portable aliases and public run IDs.

Maintainer checkouts keep a development guide **beside** this repository, at
`../crispasr-crispembed-dev.md` — a sibling of the repo root, not a file inside
it, and deliberately not tracked here (it spans CrispASR *and* CrispEmbed).

**If that file exists, read it in full before writing any code.** It is the
method, and it is not summarised anywhere: the hard rules, the
convert → quantize → dump-reference → diff → parity port pipeline, the ggml and
GPU-portability gotchas, the A/B discipline, and the storage layout. Where it
and this file disagree, it wins.

If it is not there you are on an ordinary clone — nothing is missing from the
build or the tests. Use the map below, `README.md` and `docs/`.

## Where things live

Read this before opening any of them: together they are >2 MB, and only the
development guide above is meant to be read end to end.

| Need | Read |
|---|---|
| A specific past lesson ("has this bitten us?") | `docs/LEARNINGS-INDEX.md` first — 272 lessons by topic and by model — then grep the heading in `LEARNINGS.md` |
| Is this backend fast / which quant ships | `PERFORMANCE.md` |
| What is in flight right now | `PLAN.md` — and claim your task there before starting |
| What already shipped | `HISTORY.md` — archive; consult to confirm a claim, never as a plan |
| Adding a backend | `docs/contributing.md` (12-point checklist) |
| Build, test, lint commands | `README.md`, and the development guide above |

Two habits that repeatedly cost time here:

- **`PLAN.md` and `HISTORY.md` prose goes stale.** Items marked OPEN are often
  already shipped. Audit against the CODE, never the note.
- **Auto-detection working in the CLI proves nothing about the bindings.** The
  CLI has a filename pass that short-circuits the GGUF-architecture table the C
  ABI depends on; test through the C ABI. (Issue #335.)

## Regenerated files — do not hand-edit

- `docs/LEARNINGS-INDEX.md` → `python tools/gen-learnings-index.py` (CI gates it
  with `--check`; adding a `##` section to `LEARNINGS.md` shifts every line
  number it cites).
- `bindings/go/whisper.go` cgo `LDFLAGS` → `python tools/sync_go_cgo_ldflags.py`.

- Ruby Session/Stream/Chat context guard declarations and registrations →
  `python tools/sync_ruby_context_guards.py` (`--check` runs in the Ruby ABI probe).

---

## Context-Assist fork (`kuzmi4/CrispASR`) — read before fork work

This checkout is the fork that ships the STT engine for Recheo. Upstream is
`CrispStrobe/CrispASR`; the fork adds the Context-Assist patches (F1–F12) and publishes
static arm64 + Metal archives that the app pins. This section wins over the generic maintainer routine above
when the two disagree about this line's branches, tags or scope.

**Where the work happens.** Fork work — merging a new upstream release, keeping the
patches, cutting `ca-*` tags — runs in an agent session whose working directory is *this
checkout*. The Recheo session only consumes a release tag: it bumps the pin and releases
the app, and the procedure for that half lives in Recheo
(`docs/operations.md#движок-crispasr-форк-релиз-пин`). Do not edit Recheo's pin files
(`scripts/vendor-crispasr.sh`, `CRISPASR_PIN`, `crispasr.rs` constants, bench pins) from
here.

**Branches.**

| Branch | Role |
| --- | --- |
| `main` | mirror of `upstream/main`, no own commits |
| `context-assist` (default) | production line — the patches, and the only source of `ca-*` tags |
| `context-assist-<version>` | frozen lines behind old pins: never commit to them |

**Tags.** `ca-<upstream version>.<n>` (`ca-0.8.49.1`); `-local` marks the local fallback
(`context-assist/build-static.sh`, `scripts/ca-local-release.sh` in Recheo). A `ca-*` tag
may be deleted and re-cut on the corrected commit only while no app pin points at it —
after that a mistake is fixed by a new `<n>`. CI `context-assist-static.yml` runs `test`
first and deletes the tag on red, then `build` publishes the archive, its `.sha256` and
`BUILD-INFO`.

**Bumping to a new upstream release.**

1. `git fetch upstream main --tags`; pick the release (`gh release list -R CrispStrobe/CrispASR`).
2. `git checkout -b sync/<release> origin/context-assist` and **merge** (never rebase)
   `upstream/main`, so the fork's history stays linear for the app's pins.
3. Resolve conflicts keeping our patches **and** upstream validation: the model gate in
   `session_open_explicit_impl` (`reject_external_file`, category `MODEL`), upstream's new
   rows in `backend_caps_table.h` plus our `unbounded-input`/`internal-chunking` verbs.
4. Verify the footprint before committing:
   - `git diff --stat upstream/main -- src/ include/ models/ tests/` is exactly our patches;
     an empty or foreign file in that diff means a patch was lost;
   - `git grep -n "Context-Assist" HEAD -- src/ include/` — the F1–F12 anchors are present;
   - fixes from earlier pins still exist **by content, not by tag** (`git grep <marker> HEAD`):
     a fix can live on a side line and silently drop out of a rebased release — that is how
     `ca-0.8.44.3`–`.7` shipped without the TDT blank gate of `ca-0.8.44.2`.
5. Build and test as CI does: submodules (`ggml`, `third_party/c2pa-audio`), the ggml Metal
   patch, `cmake -DCRISPASR_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug`,
   `cmake --build build-test --target context-assist-tests`,
   `ctest --test-dir build-test -L unit -L context-assist --no-tests=error`.
6. `git push origin sync/<release>:context-assist`, tag, push the tag, watch the workflow.
7. One `HISTORY.md` entry per merge; upstream's entries are never shortened.

**Keeping merges cheap.** Upstream owns most of this repo, so every fork-only line is a
future conflict:

- fork-only tests live in their own files, and the `tests/CMakeLists.txt` hunk stays as
  small as possible;
- the CI gate stays model-free (`live` label excluded): models and corpora belong to
  Recheo's benchmark, not to this repository's tests;
- the `ggml` submodule pointer is upstream's; Metal behaviour changes arrive as a patch in
  `context-assist/patches/`;
- a test that needs a GGUF, a dataset or the app's pipeline belongs to Recheo, where the
  behaviour it protects is owned.
