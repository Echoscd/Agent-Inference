# Worklog

One file per ISO week: `YYYY-Www.md`. Newest at the top of the table below.

The point of this folder is that `result/` records *what was run* but not *what
we were trying to find out*, and not what we decided afterwards. A week that
produced no experiment still gets an entry if it produced a decision.

| week | headline | entries |
|---|---|---|
| [2026-W34](2026-W34.md) | third size-vs-hazard replicate; metric layer unified; repo prepared for release | 28, 33 |
| [2026-W30](2026-W30.md) | hazard_grade introduced, first hazard A/Bs on both models | 24, 25, 26, 27 |
| [2026-W27](2026-W27.md) | A/B harness built; density / dual-descent / fidelity; config-wiring bug | 13, 15–23 |
| [2026-W26](2026-W26.md) | characterisation: prefix cache, long context, 32- and 80-way saturation, CPU offload | 01–12, 14 |

## Format

Each week's file has four sections, in this order:

```markdown
# 2026-Www  (Mon DD – Sun DD)

## TODO
- [ ] what we intend to do this week, one line each
- [x] done
- [~] dropped or deferred, with a reason on the same line

## Experiments
| id | question | outcome |
|----|----------|---------|
one row per run that landed in result/, including the ones that failed

## Findings
What we now believe that we did not believe on Monday. Include the negative
results: "X made no measurable difference" is a finding.

## Decisions
Choices that constrain later work — a metric definition, a config change, an
experiment we agreed not to run. These are the entries that are expensive to
reconstruct later.
```

Two rules that keep this useful:

- **Write the outcome, not the intent, in the Experiments table.** "aborted at
  6/80" is more useful than "size vs density".
- **Record invalid runs.** Experiments 17 and 18 sat in `result/` for weeks
  looking like data before anyone noticed both arms had run the same policy.

## Current TODO

Lives in the newest week's file, not here, so it is versioned with the week it
belonged to. Start a new week by copying the previous file's structure and
carrying over unfinished items.
