# Metrics and Result Presentation Plan

This file summarizes how we should present results for the MultiHop KGQA paper, with a focus on a space-constrained NeurIPS submission.

## 1. Recommended Evaluation Protocol

Use the following setup as the default paper protocol:

- Main benchmark: `Train on original questions -> Test on original questions`
- Robustness evaluation: `Train on original questions -> Test on paraphrase-expanded questions`
- Optional augmentation ablation: `Train on paraphrase-sampled questions -> Test on original and paraphrase-expanded questions`

### Why this protocol

- It keeps the main benchmark aligned with the standard task definition.
- It treats paraphrases as a robustness test instead of redefining the task.
- It lets us separate two questions:
  `Can the model solve the task?`
  `Does the model remain correct under wording variation?`

### Important paraphrase note

In the inspected `mquake_st` CSVs, `Question-Paraphrased` appears to include the original wording as one of the entries. Combined with the current loader:

- `question_format='paraphrased'` means training-time paraphrase sampling, not dataset expansion.
- `evaluate_paraphrases=True` means evaluation-time row expansion, where each paraphrase becomes its own evaluation example.

So `Test on paraphrase-expanded questions` is not strictly paraphrase-only unless we filter out entries identical to the original question first.

## 2. Main Paper Results

Because the dataset provides evidence paths and the adapted models produce explicit reasoning trails, the paper should not present answer quality alone. The main results should jointly report:

- answer quality
- reasoning faithfulness to the evidence path

### Table 1: Overall Answer + Faithfulness Results

This should be the primary quantitative table in the paper.

Report:

- `Mix-Hop Hits@1`
- `Mix-Hop MRR`
- `GT-Edge Overlap F1` only for single-answer settings
- `Relation Edit Distance`
- `Answer-Set F1` only for multi-answer settings

Optional extra column if space allows:

- `Path Edit Distance`

### Why these metrics belong in the main table

- `Hits@1` is the clearest headline answer-quality metric.
- `MRR` adds ranking quality without introducing too many columns.
- `GT-Edge Overlap F1` directly measures whether the predicted reasoning trail aligns with the ground-truth evidence edges, which is one of the distinctive strengths of this dataset. Basically, ground truth triplets set vs predicted triplet sets of the top rollout.
- `Relation Edit Distance` captures ordered reasoning quality at the relation-chain level and complements edge overlap by being sequence-sensitive.
  Because it is normalized per example, the overall mix-hop value should be treated as a summary rather than a standalone fidelity result.
- `Answer-Set F1` is useful only when multiple answers are valid, because it measures coverage across rollouts rather than only top-rollout correctness.
- `Path Edit Distance` is a stricter exact-path metric and is worth adding only if the table can absorb one more column.

### Recommended compact column layout

If space is tight, use:

- `Hits@1`
- `MRR`
- `Edge F1`
- `Rel. Edit Dist.`
- `Answer-Set F1` if applicable

If there is a little more room, add:

- `Path Edit Dist.`

### Table 2: Per-Hop Performance

Keep hop-difficulty analysis in a separate table so the main table stays readable.

Report:

- `2-hop Hits@1`
- `3-hop Hits@1`
- `4-hop Hits@1`
- `2-hop MRR`
- `3-hop MRR`
- `4-hop MRR`
- `2-hop Relation Edit Distance`
- `3-hop Relation Edit Distance`
- `4-hop Relation Edit Distance`

If space is extremely tight, keep only the per-hop `Hits@1` columns in the main paper and move per-hop `MRR` to the appendix.

### Why per-hop edit distance should be shown

The edit-distance metrics in this repo are normalized per example and then averaged across questions.
Since the dataset mixes different hop counts, a single overall normalized edit-distance value can be misleading when shown alone.

For the paper:

- show the overall mix-hop edit distance as a summary
- but always pair it with per-hop edit distance

This is especially important for `Relation Edit Distance`, which is one of the main faithfulness metrics.

## 3. Robustness Results

### Table 3: Paraphrase Robustness

Keep paraphrase experiments in a separate robustness table, not in the main benchmark table.

Recommended rows:

- `Train: Original -> Test: Original`
- `Train: Original -> Test: Paraphrase-expanded`
- `Train: Paraphrase-sampled -> Test: Original`
- `Train: Paraphrase-sampled -> Test: Paraphrase-expanded`

Recommended columns:

- `Hits@1`
- `MRR`
- optional `Relation Edit Distance`

### If only one robustness experiment fits

Prioritize:

- `Train: Original -> Test: Paraphrase-expanded`

This is the cleanest robustness result because it asks whether the standard model survives wording changes without paraphrase-specific training.

### What "paraphrase-expanded" means

In this repo, `paraphrase-expanded` evaluation means:

- take the `Question-Paraphrased` list for each example
- create one evaluation row per entry in that list
- replace `Question` with that entry

In the current `mquake_st` dataset, the paraphrase list appears to include the original question itself.
So under the current code and current data, `paraphrase-expanded` includes:

- the original wording
- the paraphrased variants

It is therefore not a strict paraphrase-only evaluation unless entries identical to the original question are filtered out before expansion.

## 4. Faithfulness and Interpretability Results

Since evidence-path fidelity is part of the paper's core claim, not all faithfulness metrics should be hidden in the appendix. The main table should already include:

- `GT-Edge Overlap F1`
- `Relation Edit Distance`

The remaining faithfulness metrics should be reported as supporting evidence:

- `Relation F1`
- `Path Edit Distance`
- `Node Overlap F1`
- `Answer-Set Precision`
- `Answer-Set Recall`

### Suggested interpretation

- `GT-Edge Overlap F1`:
  best evidence-grounding metric because it directly compares predicted edges with the gold evidence path
- `Relation Edit Distance`:
  best sequence-sensitive faithfulness metric for the main paper, but it should be reported per hop in addition to the overall value
- `Relation F1`:
  useful companion because it gives order-insensitive relation overlap
- `Path Edit Distance`:
  stricter than relation edit distance because it requires exact edge-level path agreement
- `Node Overlap F1`:
  weakest of the faithfulness metrics; useful, but lower priority if space is tight

## 5. Analysis and Ablations Only

These metrics should be used only when they directly support a method claim, such as STOP/RESTART signals or trajectory quality analysis.

- `Stop Rate`
- `Correct Stop Rate`
- `Incorrect Stop Rate`
- `Termination Steps`
- `Restart Any Rate`
- `Post-Restart Success Rate`
- `Restart-and-Hit Rate`
- `Special Step Rate`
- `Cycle Rate`
- `Backtrack Rate`
- `No-Op Rate`
- `Unique Edges`
- `Redundancy`
- `Avg Segment Hops`

These are useful for error analysis and behavioral diagnostics, but not for headline benchmark reporting.

## 6. Metrics To Keep Out of the Main Paper

Do not spend main-paper space on these unless they are central to a specific claim:

- `Hits@3`
- `Hits@5`
- `Hits@10`
- `Hits@20`
- `Question Entropy`
- `Path Entropy`
- `Valid Action Count`

They are useful internally, but they are not the most efficient use of limited paper space.

## 7. Reporting Notes and Caveats

- In this repo, `Hits@K` are rollout-ranked answer metrics, not full-entity-ranking KG completion metrics.
  Define this explicitly in the paper.
- `Relation Edit Distance`, `Path Edit Distance`, and overlap metrics are computed from the top-scoring rollout/path per question.
- `Answer-Set F1` is computed over the union of answer endpoints across rollouts for a question.
  It measures coverage/diversity, not only top-rollout quality.
- The normalized edit distances are averaged across examples.
  Because path lengths differ, overall edit-distance numbers should be interpreted together with per-hop breakdowns rather than shown alone.
- If using paraphrase-expanded evaluation, state clearly whether original-identical paraphrases were kept or filtered out.
  In the current dataset format, they appear to be kept unless explicitly removed.

## 8. Minimal Fallback Set

If space becomes extremely tight, keep only:

1. `Mix-Hop Hits@1`
2. `Mix-Hop MRR`
3. `GT-Edge Overlap F1`
4. `Relation Edit Distance`
5. `Per-Hop Hits@1`
6. `Answer-Set F1` only when the task is multi-answer

## 9. Final Presentation Summary

If we follow the recommended structure, the paper should present results in this order:

1. Main benchmark table:
   answer quality plus evidence-path faithfulness on original questions
2. Per-hop table:
   difficulty breakdown by reasoning depth
3. Small robustness table:
   performance under paraphrased wording
4. Appendix diagnostics:
   stop/restart behavior and trajectory analysis

## 10. Reference Implementations

The following Python functions are lightweight reference implementations of the main faithfulness metrics used in this repo.
They are written so collaborators can reproduce the metric definitions without needing to read the full training code.

```python
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def compute_precision_recall_f1(
    pred: Set,
    gt: Set,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def edit_distance(
    seq1: Sequence,
    seq2: Sequence,
) -> Tuple[int, int, int]:
    m = len(seq1)
    n = len(seq2)

    if m == 0 and n == 0:
        return 0, m, n
    if m == 0 or n == 0:
        return max(m, n), m, n

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1,  # substitution
                )

    return dp[m][n], m, n


def canon_edge(
    h: int,
    r: int,
    t: int,
    inverse_mapping: Dict[int, int],
) -> Tuple[int, int, int]:
    """
    Convert an inverse edge token back to its canonical forward edge.
    If r is an inverse relation token, map it back to its base relation
    and swap head/tail.
    """
    if r in inverse_mapping:
        return (t, inverse_mapping[r], h)
    return (h, r, t)


def canon_rel(
    r: int,
    inverse_mapping: Dict[int, int],
) -> int:
    """
    Convert an inverse relation token back to its base relation token.
    """
    return inverse_mapping.get(r, r)


def gt_edge_overlap_f1(
    pred_path: Sequence[Tuple[int, int, int]],
    gt_path: Sequence[Tuple[int, int, int]],
    special_tokens: Set[int],
    inverse_mapping: Dict[int, int],
) -> Tuple[float, float, float]:
    """
    Permutation-invariant edge overlap between predicted and gold paths.

    Mirrors the repo behavior:
    - remove special tokens such as NO_OP / STOP / RESTART
    - canonicalize inverse edges back into forward edges
    - compare edge sets
    """
    pred_edges = {
        canon_edge(h, r, t, inverse_mapping)
        for h, r, t in pred_path
        if r not in special_tokens
    }
    gt_edges = {(h, r, t) for h, r, t in gt_path}
    return compute_precision_recall_f1(pred_edges, gt_edges)


def relation_edit_distance_norm(
    pred_relations: Sequence[int],
    gt_relations: Sequence[int],
    special_tokens: Set[int],
    inverse_mapping: Dict[int, int],
    eps: float = 1e-8,
) -> float:
    """
    Normalized relation-sequence edit distance.

    Mirrors the repo behavior:
    - remove special relation tokens such as NO_OP / STOP / RESTART
    - canonicalize inverse relation tokens
    - compute Levenshtein distance
    - normalize by max(pred_len, gt_len)
    """
    pred_rels = [
        canon_rel(r, inverse_mapping)
        for r in pred_relations
        if r not in special_tokens
    ]
    gt_rels = list(gt_relations)

    dist, m, n = edit_distance(pred_rels, gt_rels)
    return dist / (max(m, n) + eps)


def answer_set_f1(
    predicted_endpoints: Iterable[int],
    gold_answers: Iterable[int],
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """
    Answer-set precision / recall / F1 over rollout endpoints.

    Use this when multiple answers are valid:
    - predicted_endpoints: all final entities reached across rollouts
    - gold_answers: all correct answer entities

    The metric compares sets, so duplicate rollout endpoints are ignored.
    """
    pred_set = set(predicted_endpoints)
    gold_set = set(gold_answers)

    tp = len(pred_set & gold_set)
    precision = tp / (len(pred_set) + eps)
    recall = tp / (len(gold_set) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1
```

### Usage Notes

- `GT-Edge Overlap F1` should be computed on the cleaned predicted path used for evaluation, not on the raw rollout trace if that raw trace still contains `NO_OP`, `STOP`, or `RESTART`.
- `Relation Edit Distance` is normalized, so for mixed-hop datasets it should be reported per hop in addition to the overall value.
- `Answer-Set F1` is mainly useful in multi-answer settings. For single-answer tasks, use a singleton gold set if needed, but it should not replace `Hits@1` or `MRR`.
