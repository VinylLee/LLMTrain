# RQ2 — MR information representation (SNLI)

Controlled SNLI experiments asking: under a fixed standard SFT objective, the
same metamorphic examples and a matched training budget, how do different forms
of MR information affect persistent MR-compliant behaviour?

**Start here → [`RQ2_V4_DESIGN_SUMMARY.md`](RQ2_V4_DESIGN_SUMMARY.md)** — the
single-entry description of the current state: dataset, template assembly,
Pair/Operation/Relation/Label design, every MR's description, and the known
leakage / control caveats.

## Current version

| | |
|---|---|
| `instruction_template_version` | **4** (current). 2 and 3 are frozen and reproducible |
| Modes | 8 core (grounding-aware 2×2×2) + 3 controls + 1 diagnostic = **12** |
| Training data | `data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json` |
| Model | `google/gemma-3-4b-it`, LoRA rank 8, 3 epochs, lr 3e-4 |
| Seeds | 42 / 43 / 44, all sharing one cohort + split manifest per seed |
| Config | [`configs/rq2_snli_config_v4.json`](configs/rq2_snli_config_v4.json) |

> The v4 run uses a **different cohort file** than v3: it is the only artifact
> whose composite rows carry ordered `component_mrs` provenance. Source rows are
> identical across the two, but the metamorphic examples are not, so **v3 and v4
> results are not comparable example-for-example**.

## Documents

| File | Contents |
|---|---|
| [`RQ2_V4_DESIGN_SUMMARY.md`](RQ2_V4_DESIGN_SUMMARY.md) | **Entry point.** Dataset, template, P/O/R/L design, MR description tables, risks |
| [`RQ2_INSTRUCTION_DESIGN_V4.md`](RQ2_INSTRUCTION_DESIGN_V4.md) | What v4 changes vs v3, control quality, limitations |
| [`MR_RELATION_AUDIT_V4.md`](MR_RELATION_AUDIT_V4.md) | Per-MR relation evidence; why `flip` narrowed to `E→C` |
| [`RQ2_INSTRUCTION_DESIGN.md`](RQ2_INSTRUCTION_DESIGN.md) | Shared 2×2×2 matrix + v3 wording (v3 frozen) |
| [`MR_RELATION_AUDIT.md`](MR_RELATION_AUDIT.md) | Relation evidence for the v3 text tables |

## Run

```bash
conda activate llmtrain310

# 8 core conditions x seeds 42/43/44
python RQ2/run_rq2_snli.py --config RQ2/configs/rq2_snli_config_v4.json \
  --seeds 42 43 44 \
  --modes none operation_only relation_only operation_relation \
          pair_only pair_operation pair_relation full_specification

# all 12 conditions (33 fine-tunes with 3 seeds)
python RQ2/run_rq2_snli.py --config RQ2/configs/rq2_snli_config_v4.json --seeds 42 43 44
```

Stages are `audit → sample → convert → finetune → test → summarize`, selectable
with `--steps`. `--smoke` caps training at 20 steps for pipeline health checks
only — not an effect result.

## Inspect before spending GPU time

```bash
# render every mode for one real pair (no model, no GPU)
python scripts/inspect_rq2_instructions.py \
  --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json \
  --pair-id 0 --mr-id composite_flip

# composite provenance / trace audit
python scripts/audit_rq2_v4_provenance.py --input <same file>
```

## Data layout

```
RQ2/data/cohorts/      # frozen v3 cohorts (do not overwrite)
RQ2/data/cohorts_v4/   # v4 cohorts
RQ2/data/converted/    # frozen v2 conversions
RQ2/data/converted_v3/ # frozen v3 conversions
RQ2/data/converted_v4/ # v4 conversions
RQ2/output/            # adapters, tests, summaries (gitignored)
```
