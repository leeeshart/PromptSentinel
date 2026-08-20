# RQ3 — Qwen3.8-as-Judge Replication (corrected)

This folder extends **PromptSentinel** RQ3 (the LLM-judge correction experiment,
paper §III.C / notebook5.ipynb) by re-running the exact judge protocol with
**Qwen3.8** as the zero-shot judge, and — critically — correcting two methodological
errors found in an earlier attempt:

1. RQ3 uses the **MiniLM embedding classifier** (`sentence-transformers/all-MiniLM-L6-v2`
   + `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)`), **not** a
   TF-IDF classifier. The FP/TP/FN sample must be derived from this classifier.
2. Table X is computed on **FP+TP only (700)** with `classifier-alone = "unsafe"` for
   every row and `pipeline = "unsafe" iff judge says UNSAFE` — not on all 900 with the
   judge replacing the classifier.

## Files

| File | Purpose |
|---|---|
| `rq3_qwen_judge.py` | Reproducible pipeline: trains the MiniLM classifier, draws the 500/200/200 sample (`--train`), and scores a judge's one-word verdicts into Table IX/X (`--score`). Contains the **verbatim** judge prompt. |
| `sample900_mini.csv` | The 900-prompt sample (500 FP / 200 TP / 200 FN), `random_state=42`. |
| `judge_verdicts.csv` | Qwen3.8 one-word verdicts (`prompt_id,verdict`) for the 900-sample. |
| `audit_predictions_mini.csv` | Full audit trail: `prompt_id, true_label, classifier_pred, judge_pred, judge_raw_output, group`. |
| `metrics_mini.json` | Computed Table IX / Table X. |
| `RQ3_QWEN_AUDIT.md` | Full audit of the prior (wrong) run, corrected methodology, results, qualitative analysis, and flagged deviations. |

## Headline results (Qwen3.8 judge)

**Table IX (judge accuracy per group)** — FP 92.6% · TP 98.5% · FN 85.5%
(paper's Llama: 94.6% / 24.0% / 20.5%).

**Table X (pipeline, FP+TP = 700)** — classifier-alone 0.286 / 1.000 / 0.444 →
**+Qwen3.8 0.842 / 0.985 / 0.908** (paper's Llama: 0.640 / 0.240 / 0.349).

Only **3 / 200 (1.5%)** true-positives were flipped to SAFE (vs Llama 152/200 = 76%),
and **none** were persona-override jailbreaks — supporting a **model-capability-driven**
account of the judge's failure mode rather than a task-inherent one.

## Reproduce

```bash
# 1) train classifier + draw sample (needs combined_dataset_final.csv + sentence-transformers)
python rq3_qwen_judge.py --data combined_dataset_final.csv --train --out sample900_mini.csv

# 2) judge the 900 prompts with Qwen3.8 using JUDGE_SYSTEM_PROMPT (first 2000 chars,
#    max_tokens=5, temperature=0.0; verdict = UNSAFE if "UNSAFE" in response else SAFE)
#    -> produce judge_verdicts.csv (prompt_id,verdict)

# 3) score
python rq3_qwen_judge.py --score --out sample900_mini.csv --judge judge_verdicts.csv
```

> Note: in the original run the judge was executed via Qwen3.8 subagents embodying the
> verbatim prompt (temperature/length params are an API-call detail of the notebook's
> direct call). The sample, verdicts, and metrics here are the actual produced artifacts.
