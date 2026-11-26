# GPT vs. RILE: automated left–right placement of party manifestos

Code sample from my bachelor's thesis *"How well does GPT reconstruct parties' left–right
positions compared to MARPOR's RILE?"* (B.A. Political Science, University of Tübingen,
first examiner Jakub Sowula, submitted 30 Sept 2025, grade 1.0).

---

## Research question

The RILE scale of the Manifesto Project (MARPOR) is one of the most widely used indicators of
party positions, and it has been criticised for years for coding errors and inconsistencies.
The thesis tests whether an LLM (GPT) can serve as a reproducible cross-check: I have GPT
classify the same quasi-sentences that human MARPOR coders coded, aggregate the labels into a
RILE-analogous index, and compare both against an external expert benchmark.

**Data:** 49 elections in seven countries (DE, ES, UK, DK, NO, NL, GR), late 1990s to 2020s.
Texts from the Manifesto Project Database (Lehmann et al. 2024), benchmark from
Gemenis & Bruinsma (2020), based on CHES expert data.

**Result in one sentence:** A tie on average, but a conditional one: where MARPOR's human
coding is strong, RILE wins; where it is weak, GPT is more accurate. Across repeated runs, GPT
reaches higher test–retest consistency than the published intercoder reliability of human
coders. LLMs are therefore a complement, not a replacement: useful for flagging suspicious
data points for targeted audits.

## What's in here

| File | Contents |
| --- | --- |
| `classify_manifestos.py` | Classification pipeline: CSV of quasi-sentences in, labels (left/right/neutral) out. Batching, JSON output, error handling, merge via row IDs. |
| `rile_score.py` | Aggregates the labels into a RILE-analogous score per manifesto. |
| `requirements.txt` | `pandas`, `openai`. |

```bash
export OPENAI_API_KEY=sk-...
python classify_manifestos.py --input data/SPDGER2013.csv --output results/
python rile_score.py results/*.csv
```

File names follow the pattern `[party abbreviation][ISO country code][election year]`, e.g.
`SPDGER2013` for the SPD manifesto for the 2013 German federal election.

## Design decisions visible in the code

**Three labels instead of 56 categories.** The first attempt to map MARPOR's 56 categories
directly onto left/right/neutral produced overly long definitions, and "neutral" became a
residual category where everything ambiguous ended up. Three fixes: definitions condensed to
the core distinctions; the task framed as a three-way choice from the start, so that neutral
is conceptually on equal footing; contested categories such as 305 *Political Authority*
excluded, which MARPOR counts as "right" and which the literature identifies as a main source
of error in RILE (Mikhaylov et al. 2012).

**JSON instead of a table or free text.** All three output formats were tested. JSON parses
reliably and goes straight into a DataFrame. Every input row carries its ID, so answers are
matched correctly even if the model changes the order or skips rows.

**Missing labels stay empty.** A batch with broken JSON costs at most `batch_size` rows; those
remain unclassified and are counted at the end instead of being silently lost. For a
measurement instrument, a visible gap is better than an invisible one.

**Parameters:** `temperature=0.2` (Wang et al. 2024). Batch size was varied and tested without
any substantial effect, so a large batch size was chosen for efficiency. Main model GPT-4o;
GPT-5 only exploratively as a sensitivity check.

## Transparency about this version

The classification code is the main script from the thesis appendix (pp. 29–32). Compared with
the submitted version it has been **restructured, not changed in substance**: paths and
parameters are now passed on the command line instead of hard-coded variables, the API key
comes from an environment variable instead of a file, prompt construction and the batch loop
live in named functions, and docstrings have been added. Prompt, label definitions,
aggregation logic and output format are unchanged.

`rile_score.py` is newly written and reconstructs the aggregation described in the methods
section (section 4.7), which was not printed in the appendix.

## References

Gemenis, K. & Bruinsma, B. (2020): *Replication Data for: Challenging the Manifesto Project
Data Monopoly.* Harvard Dataverse. | Lehmann, P. et al. (2024): *Manifesto Project Dataset.*
| Mikhaylov, S., Laver, M. & Benoit, K. (2012): Coder Reliability and Misclassification in the
Human Coding of Party Manifestos. *Political Analysis* 20(1), 78-91.
