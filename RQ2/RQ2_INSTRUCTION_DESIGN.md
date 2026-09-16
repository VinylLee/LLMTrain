# RQ2 — Grounding-aware 2×2×2 MR-information design

**RQ2.** Under a fixed standard supervised fine-tuning objective, the same
metamorphic examples, and a matched training budget, how do different forms of
MR information affect *persistent MR-compliant behavior*?

This document specifies the experimental design implemented in
`scripts/mr_instruction_design.py` (the single source of truth), consumed by
`scripts/convert_nli_to_ft.py`, `RQ2/scripts/convert_snli_rq2.py` and
`RQ2/run_rq2_snli.py`.

> **Version note (2026-09-16).** This document describes the shared experimental
> matrix and the **v3** wording (`instruction_template_version = 3`), which is now
> **frozen**. v4 (`= 4`) is the current version for new formal runs and changes only
> the Operation/Relation *content model* — see `RQ2_INSTRUCTION_DESIGN_V4.md`. The
> matrix, block order, controls and `full_oracle` semantics below apply verbatim to
> both.

Related: `RQ2/MR_RELATION_AUDIT.md` (per-MR relation evidence, v3),
`RQ2/RQ2_INSTRUCTION_DESIGN_V4.md` + `RQ2/MR_RELATION_AUDIT_V4.md` (v4).

---

## 1. The three information dimensions

| | Dimension | Meaning |
|---|---|---|
| **P** | Pair / source–follow-up grounding | The paired source sample (`x_source`) is rendered. |
| **O** | Operation / input-transformation specification | Describes the input-side transformation `T_r`. |
| **R** | Relation / output-relation specification | Describes the source→follow-up label mapping. |
| **L** | Source-label anchoring *(diagnostic only)* | The source sample's gold label is rendered. |

**P is a grounding factor, not a symmetric peer of O and R.**

* `P = 0` → O and R are rendered as **ungrounded** metadata / provenance:
  *the model is told what kind of transformation was applied and what the
  relation between a source and its follow-up is, but not which source.*
* `P = 1` → O and R become **source-grounded** specifications: *they are bound
  to a concrete source–follow-up pair the model can inspect.*

This distinction — not O and R alone — is the object of study. The question is
not only *whether* operation/relation information helps, but *whether its
effect depends on being grounded to a specific pair*.

`L` is **not** one of the three factors and is **never** part of the core
2×2×2. Only the diagnostic condition `full_oracle` sets `L = 1`.

---

## 2. Core factorial cells

| Canonical mode | P | O | R | L | Grounding | Interpretation |
|---|:-:|:-:|:-:|:-:|---|---|
| `none` | 0 | 0 | 0 | 0 | ungrounded | implicit metamorphic examples only |
| `operation_only` | 0 | 1 | 0 | 0 | ungrounded | ungrounded operation metadata |
| `relation_only` | 0 | 0 | 1 | 0 | ungrounded | ungrounded relation metadata |
| `operation_relation` | 0 | 1 | 1 | 0 | ungrounded | ungrounded O+R metadata |
| `pair_only` | 1 | 0 | 0 | 0 | source_grounded | explicit source–follow-up grounding |
| `pair_operation` | 1 | 1 | 0 | 0 | source_grounded | grounded operation specification |
| `pair_relation` | 1 | 0 | 1 | 0 | source_grounded | grounded relation specification |
| `full_specification` | 1 | 1 | 1 | 0 | source_grounded | grounded operation + relation |

`none` is the reference cell. `full_specification` is the **P1O1R1 L0** cell —
it must never show the source label.

### Semantic caveat on the P=0 cells

In `relation_only` / `operation_relation` the source pair is *not* shown, so the
Relation block can only be read as an **abstract metadata / specification**, not
as an instance-level, executable relational constraint. Analysis must not treat
`relation_only` and `pair_relation` as the same manipulation at two intensities;
they differ in kind.

---

## 3. Controls and diagnostic

| Mode | P | O | R | L | Role |
|---|:-:|:-:|:-:|:-:|---|
| `pair_shuffled_operation` | 1 | 1 (wrong, matched) | 0 | 0 | control |
| `pair_shuffled_relation` | 1 | 0 | 1 (wrong, matched) | 0 | control |
| `mismatched_pair` | 1 (wrong source) | 0 | 0 | 0 | control |
| `full_oracle` | 1 | 1 | 1 | **1** | diagnostic |

* `pair_shuffled_operation` — pair correct, current sample correct, target label
  unchanged; only the Operation text comes from a different MR. It isolates
  *correct grounded Operation semantics* from *matched wrong semantics*.
* `pair_shuffled_relation` — same idea for the Relation text. The Relation comes
  from a **relation type** different from the true one, so the rendered text is
  guaranteed to differ (verified in the conversion report:
  `relation_text_unchanged_count == 0`).
* `mismatched_pair` — the double-input format and token budget are preserved, but
  the source belongs to a different group. It isolates *correct correspondence*
  from *the mere presence of a second input*.
* `full_oracle` = **P1 O1 R1 L1**. It is a diagnostic upper-bound condition used
  to test whether **source-label anchoring** makes the Relation easier to
  execute. It is **not** the "full" cell of the factorial.

**Do not** interpret `full_oracle − pair_operation` as a pure Relation effect:
it adds both the Relation and the source label. The label-anchoring increment is
`full_oracle − full_specification`, and it is only valid because
`full_specification` already has P1 O1 R1.

---

## 4. Instruction structure

Every instruction is the concatenation of fixed blocks in a fixed order:

```
[PAIR_BLOCK]           if P = 1
[LABEL_ANCHOR_BLOCK]   only if L = 1
[OPERATION_BLOCK]      if O = 1
[RELATION_BLOCK]       if R = 1
[NLI_TASK_BLOCK]       always
```

Only block **presence** varies across modes. Wording, block order, separators
and the NLI task text are identical everywhere. `INSTRUCTION_TEMPLATES` is
*derived* from `MODE_SPECS`; there is no per-mode hand-written template to drift.

Canonical block wording:

```text
Paired source sample:
<source_premise>
{original_premise}
</source_premise>
<source_hypothesis>
{original_hypothesis}
</source_hypothesis>
```
```text
Input-transformation specification:
{operation_description}
```
```text
Output-relation specification:
{relation_description}
```
```text
Source label:
{original_label}
```
```text
Determine the natural language inference relation between the premise and hypothesis.
Answer with exactly one label: entailment, neutral, or contradiction.
```

The pair block is deliberately *not* worded like a few-shot demonstration, and
carries no label, no operation and no relation. Its function is only to say:
**this sample and that source belong to the same source–follow-up pair.**

### Operation vs. Relation

* **Operation** is *declarative provenance*, never an imperative. It describes the
  input-side edit and never the expected output. Forbidden in the Operation text:
  `label`, `prediction`, `entailment`, `neutral`, `contradiction`, `output`,
  `relation is preserved`. Enforced by
  `tests/test_rq2_instruction_design.py::test_operation_text_has_no_output_relation_hint`.
* **Relation** states the abstract source→follow-up mapping, never the current
  sample's label. Forbidden: "the correct label for this sample is …".

### Fixed invariants

* The training target is always the **current** sample's label; the source label
  never enters a core 2×2×2 cell (hard-checked in `convert_to_alpaca`).
* Source rows (`mr_id == none`) use the plain NLI instruction in **every** mode.
* The training cohort, the source/follow-up counts and the labels never change
  with the mode.
* No extra loss, no KL/ranking/relation loss; standard SFT objective only.
* No MR specification is provided at inference/test time.

---

## 5. Planned contrasts

Do **not** lead with traditional factorial main effects. The primary analysis is
a set of planned conditional contrasts.

### Operation

| Contrast | Interpretation |
|---|---|
| `operation_only − none` | ungrounded operation metadata effect |
| `pair_operation − pair_only` | grounded operation effect |
| `(pair_operation − pair_only) − (operation_only − none)` | **grounding moderation of Operation** |

### Relation

| Contrast | Interpretation |
|---|---|
| `relation_only − none` | ungrounded relation metadata effect |
| `pair_relation − pair_only` | grounded relation effect |
| `(pair_relation − pair_only) − (relation_only − none)` | **grounding moderation of Relation** |

### Grounded decomposition

| Contrast | Interpretation |
|---|---|
| `full_specification − pair_operation` | Relation effect, conditional on Pair+Operation |
| `full_specification − pair_relation` | Operation effect, conditional on Pair+Relation |

### Semantic controls

| Contrast | Interpretation |
|---|---|
| `pair_operation − pair_shuffled_operation` | correct grounded Operation semantics vs. matched wrong semantics |
| `pair_relation − pair_shuffled_relation` | correct grounded Relation semantics vs. matched wrong semantics |
| `pair_only − mismatched_pair` | correct correspondence vs. double-input format alone |

### Label anchor

| Contrast | Interpretation |
|---|---|
| `full_oracle − full_specification` | source-label anchoring increment |

---

## 6. Shortcut risk in the ungrounded Relation cells

Relation texts for `flip` and `neutral` MRs state, in the abstract, that the
follow-up label is `contradiction` / `neutral`. Without the source pair, that
text is close to a direct statement of the target label: for those MRs
`relation_only` degenerates toward `MR identity → target label` rather than
relation learning.

In the SNLI seed-42 cohort this affects **1809 / 3307 augmented rows (54.7%)**
(`flip` 1102 + `neutral` 707). The `inv` MRs (1498 rows, 45.3%) are not exposed
in this way, because their text says only "same as the source label" and the
source label is hidden when P=0.

Consequences:

* `relation_only` / `operation_relation` are retained as **metadata
  diagnostics**, not as evidence of relation learning.
* Primary Relation conclusions must come from `pair_relation`,
  `full_specification` and `pair_shuffled_relation`.
* This limitation is inherent to the relation definition; it is not fixable by
  rewording without destroying the specification's meaning.

---

## 7. Known limitations

1. **Composite Operation provenance.** The SNLI cohort in use
   (`data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json`)
   records **no** `component_mrs` field. All three composites therefore fall back
   to one shared generic Operation text
   (`composite_operation_fallback_count` in the conversion report; 1857/3155
   train rows for seed 42). A newer artifact
   (`...-v3_3/augmented_data_all_label_mrs_v3_3_full.json`) *does* carry
   `component_mrs`, and the resolver will use it automatically — but switching
   cohorts would change the matched metamorphic examples, which is out of scope
   here.

   Note the fallback is deliberately **identical for all three composites**, so
   that the Operation wording cannot reveal the relation class.

2. **Inert rows in `pair_shuffled_operation`.** Because the three composites
   share one Operation text, a composite row reassigned to another composite MR
   keeps the same rendered text. Measured on the seed-42 train split: **833 /
   3155 augmented rows (26.4%)**; the theoretical floor under matched marginals
   is 559. This is reported as
   `operation_text_unchanged_count` but does not fail strict mode. It affects
   exactly those rows whose Operation text carries no MR-specific semantics, so
   the contrast `pair_operation − pair_shuffled_operation` should be read as
   driven by the non-composite rows.

3. **`pair_shuffled_relation` has no inert rows** (`relation_text_unchanged_count
   == 0`): the shuffle is over relation *types*, and every reassignment changes
   the rendered text.

4. **Token-length mismatch.** The P=1 cells carry the source premise/hypothesis
   and are longer than the P=0 cells. This is a property of the manipulation, not
   a bug, but it means P contrasts are not length-matched. Token statistics per
   mode are emitted by the converter (`token_length`) for reporting.

5. **Unsupported MR ids.** `race_sensitive_transformation`,
   `same_type_named_entity_substitution` and `tense_shift` appear in no dataset
   and have no generator or oracle definition; they are retained only as a
   defensive whitelist. See `RQ2/MR_RELATION_AUDIT.md`.

---

## 8. Versioning and backward compatibility

* `INSTRUCTION_TEMPLATE_VERSION = 3` (`scripts/convert_nli_to_ft.py`).
* Version **2** is a *frozen legacy* template set kept so historical configs
  (`"instruction_template_version": 2`, i.e. all
  `experiments_config_mrinstr_*.json`) remain loadable and reproducible. It is
  reachable via `--instruction-template-version 2` and must not be extended.
* Deprecated mode names, still accepted:

  | Deprecated | Canonical | Note |
  |---|---|---|
  | `relation_aware` | `full_specification` | was in fact P1 O1 R1 L0 |
  | `shuffled_operation` | `pair_shuffled_operation` | already included the pair |

  Both print a deprecation warning at run time. They are **exact** aliases: the
  generated training files are byte-identical (tested).
* Because the v3 block wording and order differ from v2, previously converted
  RQ2 data (`RQ2/data/converted/*`) and previously trained adapters are **not
  comparable** to v3 runs. v3 conversions and the v3 output root are versioned
  separately (`RQ2/data/converted_v3/`, `RQ2/output/gemma3_4b_nli_grounded_v3/`).

---

## 9. Reproducibility contract

Every conversion records, in `conversion_report.json`:

* `mr_design` — `pair` / `operation` / `relation` / `label_anchor`, `grounding`,
  `pair_source` / `operation_source` / `relation_source`, `role`;
* `design_integrity` — `pair_fallback_count`, `missing_operation_count`,
  `missing_relation_count`, `missing_source_label_count`,
  `composite_operation_fallback_count`,
  `shuffled_operation_identity_collision_count`,
  `shuffled_relation_identity_collision_count`, `relation_type_mismatch_count`,
  `operation_provenance_distribution`, `relation_type_distribution`;
* `instruction_template_version`, `instruction_template_hash`,
  `operation_description_hash`, `relation_effect_hash`, `data_signature`,
  `manifest_hash`, `ordered_sample_signature`, converted-file SHA-256.

The same `mr_design` block is written to `experiment_meta.json` by both
`RQ2/run_rq2_snli.py` and `scripts/run_batch_experiments.py`, so downstream
analysis never has to re-derive the design from the mode name.

**Strict mode** (`--strict-pairing`, and always on inside the RQ2 converter)
fails on: a missing Operation where O=1, a missing Relation where R=1, a missing
source where P=1, a missing source label where L=1, and any identity collision in
a shuffled control.

All 11 conditions × seeds 42/43/44 share one `cohort_id`, one split manifest, one
cohort sample file and therefore one set of training examples.

---

## 10. Inspecting instructions

```bash
# list available (pair_id, mr_id) combinations in a data file
python scripts/inspect_rq2_instructions.py \
    --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json --list

# render all core + control + diagnostic modes for one real follow-up sample
python scripts/inspect_rq2_instructions.py \
    --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json \
    --pair-id 358 --mr-id adding_contradiction

# machine-readable
... --json
```

No model, tokenizer or GPU is required.
