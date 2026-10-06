# RQ2 v4 / SNLI V1.1 execution

Approved scope: SNLI only, the existing 14 v4 modes, seed order **42 → 43 → 44**.
Seed 42 reuses its completed 3-epoch adapters. Seed 43 failed during model loading
in the earlier run; seed 44 had not started. Their 28 formal adapters still need
training. No MNLI/SICK evaluation or SA work is included.

## Entry point and isolated outputs

```bash
# CPU checks only; no model loading or GPU work
/home/ubuntu/.conda/envs/llmtrain310/bin/python RQ2/run_rq2_v11.py --prepare-only

# Serial queue, verified resume of adapters/predictions/checkpoints
/home/ubuntu/.conda/envs/llmtrain310/bin/python -u RQ2/run_rq2_v11.py

# Explicit review gate: only existing seed42 adapters; no seed43/44 work
/home/ubuntu/.conda/envs/llmtrain310/bin/python -u RQ2/run_rq2_v11.py --seed42-only
```

The overlay `RQ2/configs/rq2_snli_config_v4_test_v1_1.json` keeps the original
scientific configuration as its base. The old runner and all v2/v3/v4 instruction
and metric code remain unchanged.

| Artifact | Location |
|---|---|
| Uploaded SNLI test | `data/nli/mr_test_data_merged_v1.1/snli.jsonl` |
| Hash-checked snapshot | `RQ2/data/test_v1_1/mr_test_data_merged/snli.jsonl` |
| Original/formal adapters | `RQ2/output/gemma3_4b_nli_grounded_v4/` |
| New predictions, audit, progress, logs, review reports | `RQ2/output/gemma3_4b_nli_grounded_v4_test_v1_1/` |

The snapshot SHA-256 is
`21c1236c6b196be3382af3f4f04136d350ff73e03c9970db4f2f66f5763115fc`.
It has 23,152 rows: 9,824 source and 13,328 follow-up, including 2,318 composite.
Sources match the old test set by text and label, but most follow-ups differ.
Old and V1.1 results must not be pooled or interpreted as model-improvement deltas.

## Safety gates and shared GPU policy

- Audit schema, unique-source pairing, exact train/test disjointness, gold relation
  consistency, cohort/split signatures, converted hashes, and ordered provenance.
- Freeze code hashes, package versions, base-model snapshot and tokenizer/config
  hashes as run/evaluation identity; do not mix predictions after a code/model change.
- Build an immutable per-seed isolated training registry pointing only at the
  audited v4 train/validation files; do not trust a stale shared registry entry.
- Strict wrong Operation must have 100% coverage within relation kind/arity;
  wrong Relation must have 100% kind/text mismatch. No partial or cross-family fallback.
- The inference smoke keeps complete groups, covers every MR, and adds the longest
  32 follow-up inputs (115 rows for the frozen V1.1 snapshot). It runs on seed-42
  `none` and `full_oracle` before full evaluation.
- Seed 43 uses isolated 20-step training/inference smoke for `full_oracle` and both
  strict controls. Seed 44 uses an isolated `full_oracle` smoke. Smoke adapters
  are never accepted as formal ones.
- Before **every** GPU task query all GPUs; select the one with most free memory,
  preferring lower utilization for ties. Expose only its UUID to the child.
- PyTorch allocator budgets: inference 16 GiB, training 17 GiB. Require at least
  another 4 GiB free before launch and check again after CUDA initialization.
  The cap covers PyTorch's allocator, not all CUDA/driver allocations; other
  users' later allocations are not under this runner's control.
- If no GPU qualifies, wait and poll without touching others' processes. Only one
  RQ2 GPU child runs at a time. A pre-model admission race requeues; post-model
  OOM/other failures stop the queue, never silently
  change batch, cutoff, model precision, or the scientific training settings.
- Training checkpoint frequency is 100 steps instead of the base config's 9999;
  this recovery-only difference is recorded separately. All learning settings
  remain unchanged. Resume requires a matching job identity and complete optimizer,
  scheduler, RNG, and trainer state.
- Missing/mismatched/incomplete adapters fail before inference; no base-only fallback.
  Prediction count, row identity, gold labels, parsed labels, `correct`, and paired
  denominator must pass before an evaluation is marked completed.
- Invalid/partial new predictions are preserved under `failed_attempts/`; the
  next invocation can retry. An interrupted final adapter save is recoverable
  only for an identified matching training job with a complete checkpoint.
  Checkpoint selection skips incomplete newer saves in favor of a complete older one.
- Old test predictions, reports and test snapshot stay untouched. Failed seed-43
  metadata is preserved in the new audit directory before continuing its training.
- `flock` prevents duplicate queues. Logs are persistent; absence of an output
  JSONL during inference is normal because the legacy tester writes at completion.

`_progress.json` records the current job and verified completions. Each seed
produces `seed<seed>_review.json`; `RESULTS.md` contains per-seed numerator/denominator
results. `v11_report.json` adds mode-wise means/sample standard deviations and
atomic/composite breakdowns. Single-seed standard deviation is null, not zero.
MSR remains exactly the existing metric implementation. Source-only groups enter
source accuracy but not joint-correctness/MSR denominators.

Automated V1.1 ensemble filtering and gold-relation consistency do **not** establish
human semantic validity. In particular, `conditional_clause` is protected from
automatic dropping by the uploaded policy; this is a documented evaluation limitation.

## Baseline tests (before these changes)

- RQ2 v4: 78 passed; frozen v3: 86 passed; converter: 46 passed.
- Whole suite: 534 passed, 1 pre-existing failure:
  `tests/test_rq1_back_translation.py::BackTranslationConfigTests::test_config_points_to_existing_5340_row_files`.
  The configured seed-42 back-translation JSONL is absent. Do not repair unrelated
  RQ1 data/config or conceal this failure in the RQ2 run.

New CPU-only tests cover version separation, GPU budget/spare selection,
pre-GPU-only scheduling revisions, representative smoke pairing, prediction
identity/ERROR rejection, missing adapter rejection, completed training identity,
and source-only paired-denominator behavior.

Backed up before edits on remote branch
`backup/rq2-v4-pre-v11-20261005-210439`, commit
`03ebc34fc52dbeb8f0435fbd359c579782b4570e`. The snapshot commit is empty because
the RQ2 source was already committed; unrelated dirty SA files were not staged.

On 2026-10-05 the execution platform rejected the full unattended queue,
interpreting an older seed42-review requirement as still binding; even the
restricted `--seed42-only` launch timed out during review. No GPU job launched
that day and no refused command was bypassed.

On 2026-10-06 the user explicitly authorized: **允许 seed 42 新版评测通过工程检查后，
无需再次人工审批，继续 seed 43 和 44 的训练与评测**. The platform accepted the full
serial queue. It started with PID `798538` using the default command (without
`--seed42-only`). The 230 RQ2-related regression checks passed. Seed42 `none`
and `full_oracle` inference smoke each passed 115-row identity/pairing checks
with zero invalid predictions; the full seed42 `none` evaluation then started
on GPU 1. These are startup/health confirmations, not final experimental results.
Subsequent jobs reselect a GPU by available memory. Seed43/44 may now proceed
automatically after the preceding seed's engineering checks; errors still stop
the queue. Check `_progress.json` and verified reports for current completion.

The feature branch `feature/rq2-v4-v11-evaluation-20261005-210439` was successfully
pushed on 2026-10-06 (code commit `39bc13e`). Old SA/RQ1 dirty worktree changes
remain outside the RQ2 commits.
