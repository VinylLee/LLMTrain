# RQ2 v4 — ordered provenance-explicit Operation + constraint-form Relation

`instruction_template_version = 4`. Supersedes v3 for new formal runs.

**Entry point: [`RQ2_V4_DESIGN_SUMMARY.md`](RQ2_V4_DESIGN_SUMMARY.md)** — read that
first for the dataset, the full template assembly, the P/O/R/L design and the
per-MR description tables. This document covers only what v4 *changes*.
Companion docs: `RQ2_INSTRUCTION_DESIGN.md` (shared matrix, v3 wording),
`MR_RELATION_AUDIT_V4.md` (relation evidence).

v4 keeps **everything** about the experimental design fixed and changes only the
content model of two information channels. The three channels become cleanly
separable:

| Channel | Question it answers | Type |
|---|---|---|
| **Pair** | Which source input is this follow-up grounded to? | grounding |
| **Operation** | What ordered input transformations produced this follow-up? | factual provenance |
| **Relation** | What must a valid source–follow-up pair's outputs satisfy? | normative constraint |

---

## 1. What changed from v3

| | v3 | v4 |
|---|---|---|
| Pair header | `Paired source sample:` | `Paired source input:` |
| Operation header | `Input-transformation specification:` | `Input transformation:` |
| Operation content | one generic sentence, named by the **composite's** mr_id | **ordered trace** `T1 → … → Tk`, expanded from `component_mrs` |
| Composite wording | 3 composites shared 1 generic fallback | each row gets its **real** component sequence |
| Relation header | `Output-relation specification:` | `Output relation:` |
| Relation content | totalised `E↔C, N→N` map, "likely/tends to" left over in early drafts | `must`-form constraint on the **applicability-restricted** branch |
| Relation kinds | `inv` / `flip` / `neutral` | `invariance` / `entailment_to_contradiction` / `entailment_to_neutral` |
| Composite provenance | optional, silent fallback | **required** for a formal run (`--require-composite-provenance`) |

**Unchanged:** the 2×2×2 matrix, the existing controls, `full_oracle`, block order,
`LABEL_ANCHOR` semantics, the NLI task block, source-row behaviour, the training
cohort *mechanism*, and the standard SFT objective.

Block order is still, for every mode:

```
[PAIR_BLOCK, if P=1]
[LABEL_ANCHOR_BLOCK, only if L=1]
[OPERATION_BLOCK, if O=1]
[RELATION_BLOCK, if R=1]
[NLI_TASK_BLOCK]
```

---

## 2. Operation = an ordered transformation trace

Canonical rendering — **atomic and composite rows use the same renderer**; an
atomic row is simply an arity-1 trace:

```text
Input transformation:
The follow-up input was derived from its source input through the following ordered transformation sequence:
1. {step_1}
2. {step_2}
```

Atomic example (`synonym_replacement`):

```text
Input transformation:
The follow-up input was derived from its source input through the following ordered transformation sequence:
1. One or more words were replaced with context-appropriate synonyms.
```

Composite example (`composite_flip`, `component_mrs = ["pronoun_substitution", "negation_flip"]`):

```text
Input transformation:
The follow-up input was derived from its source input through the following ordered transformation sequence:
1. A noun phrase or pronoun was replaced with a coreferential expression.
2. A negation marker was added, removed, or reversed.
```

### `component_mrs` is an ordered sequence

The array order **is** the execution order. The implementation never sorts,
never casts to a set, never deduplicates, and never reorders by relation class.
A repeated entry means the transformation ran twice and is rendered twice
(`tests/test_rq2_v4_instruction_design.py::test_repeated_component_renders_two_steps`).

### Atomic step table

| mr_id | canonical step sentence |
|---|---|
| `synonym_replacement` | One or more words were replaced with context-appropriate synonyms. |
| `pronoun_substitution` | A noun phrase or pronoun was replaced with a coreferential expression. |
| `voice_switch` | A sentence was rewritten between active and passive voice. |
| `conditional_clause` | A conditional clause was added to one component of the input. |
| `negation_flip` | A negation marker was added, removed, or reversed. |
| `antonym_substitution` | One or more content words were replaced with context-appropriate antonyms. |
| `uninformative` | Additional content that is semantically irrelevant to the affected component was inserted. |
| `adding_contradiction` | Additional content that conflicts with a proposition in the affected component was inserted. |

None of these names an output label, a relation, or a prediction. Wording
semantics were checked against the generator contract (MTrain
`OPERATION_SPECS`) rather than selected for fluency.

### Composite Operation is decided by `component_mrs` alone

If `composite_inv` and `composite_flip` happen to carry the same
`component_mrs`, their Operation blocks are **byte-identical**. The relation
class may only appear in the Relation block. This is enforced by
`test_same_component_sequence_gives_identical_operation_across_composite_types`.

### Provenance requirement

| situation | provenance tag | formal v4 |
|---|---|---|
| atomic row with a defined step | `exact` | ok |
| composite row with a usable `component_mrs` | `composite_components` | ok |
| composite row with **no** `component_mrs` | `composite_fallback` | **fails** |
| component / mr_id with no step definition | `unsupported` | **fails** |
| source row | `none` | n/a |

`--require-composite-provenance` (set in `rq2_snli_config_v4.json`) turns
`composite_fallback` into a hard error, so a generic
"multiple transformations were applied" can never silently enter a formal run.
Non-strict callers (inspection, smoke) still get the fallback text and a count.

---

## 3. Relation = an output constraint

```text
Output relation:
For a valid source-follow-up pair… , the output labels must satisfy the following constraint:
…
```

Three kinds, one shared text each (shared text is deliberate: the wording must
not identify the specific MR):

| kind | text |
|---|---|
| `invariance` | For a valid source-follow-up pair, the output labels must satisfy the following constraint: the follow-up label must be the same as the source label. |
| `entailment_to_contradiction` | For a valid source-follow-up pair **to which this relation applies**, the output labels must satisfy the following constraint: if the source label is entailment, the follow-up label must be contradiction. |
| `entailment_to_neutral` | For a valid source-follow-up pair **to which this relation applies**, the output labels must satisfy the following constraint: if the source label is entailment, the follow-up label must be neutral. |

`must`, never `likely` / `generally` / `may` / `tends to` — for a pair that has
already passed the MR's applicability filter, this is a behavioural oracle, not
a tendency. These texts describe only the abstract `source output → follow-up
output` mapping; they never state the current sample's answer.

See `MR_RELATION_AUDIT_V4.md` for why `flip` was narrowed from the totalised
3-class map to the applicability-restricted `E → C` branch.

---

## 4. Shuffled controls

### `pair_shuffled_operation` — matched derangement

The control is built from `operation_trace_id`, with preferences applied in
order:

1. **same arity** (1-step ↔ 1-step, 2-step ↔ 2-step);
2. **same relation kind** where possible;
3. the assigned trace must **differ** from the row's own trace;
4. seed-dependent ordering, so the control is not one fixed permutation.

Hard requirements, enforced in the RQ2 wrapper:
`operation_trace_identity_collision_count == 0` and
`operation_text_unchanged_count == 0`.

Rows that cannot be deranged inside their arity stratum (Hall's condition) are
repaired by an explicit cross-stratum donor swap and counted in
`shuffled_operation_cross_arity_fallback_count`. Nothing falls back silently.

Measured on the v4 cohort, seed 42 (`RQ2/data/converted_v4/rq2_snli_pair_shuffled_operation_seed42/`):

| metric | value |
|---|---|
| `operation_trace_identity_collision_count` | **0** |
| `operation_text_unchanged_count` | **0** |
| `shuffled_operation_same_arity_rate` | 0.994 (3150 / 3170) |
| `shuffled_operation_cross_arity_fallback_count` | 20 |
| `shuffled_operation_no_valid_candidate_count` | 0 |
| `shuffled_operation_same_relation_kind_rate` | 0.285 |

This is the metric v3 could not deliver: in the v3 cohort 833/3155 augmented
rows had `correct operation text == shuffled operation text`, because the three
composites shared one generic fallback. In v4 that count is **0**.

### `pair_wrong_operation_relation_matched` — strict relation-matched negative

This control keeps the correct Pair and shows an Operation payload from a donor
with the same `relation_kind` and `operation_arity`, while requiring both the
ordered `operation_trace_id` and rendered Operation text to differ. Donors may
be reused; this matches the Relation-family prior and arity, not the Operation
marginal distribution. Selection is deterministic by seed and uses
token-length and trace-complexity proximity as soft tie-breakers.

Rows without a strict donor are explicitly unavailable. They are reported by
the feasibility audit and are never filled with a cross-family or cross-arity
donor. Default formal conversion fails on such rows;
`--allow-partial-control-coverage` may be enabled only after reviewing the
audit, in which case unavailable rows are omitted rather than substituted.

### `pair_shuffled_relation`

Reassigns the relation **kind**, preserving the kind marginals, so a row never
renders its own relation text when that is attainable.

It is **not always attainable** on this cohort: `entailment_to_neutral` covers
1837 of 3170 augmented train rows (58%), and with only three kinds Hall's
condition fails. The residual equals the theoretical minimum
(`2·max − total`):

| seed | augmented | max kind bucket | Hall deficiency | achieved collisions |
|---|---|---|---|---|
| 42 | 3170 | 1837 | 504 | **504** (15.9%) |
| 43 | 3089 | 1797 | 505 | **505** (16.4%) |
| 44 | 3119 | 1817 | 515 | **515** (16.5%) |

The marginals stay matched, so the contrast is not confounded by a shifted kind
distribution, but for those ~16% of rows the Relation control is inert. This is
reported as `shuffled_relation_identity_collision_count` in
`matched_control_audit` and must not be described as a full derangement.

### `pair_wrong_relation` — 100% semantic mismatch

This control keeps the correct Pair, input and target, omits Operation, and
replaces each Relation kind using the deterministic cycle
`invariance → entailment_to_contradiction → entailment_to_neutral → invariance`.
Every transformed row therefore has a different Relation kind and text. It is
complementary to `pair_shuffled_relation`: it guarantees semantic mismatch but
does not preserve the Relation marginal distribution.

---

## 5. Grounding invariance

`P` is the only thing the grounding manipulation may change. For the same row:

* the Operation payload is **byte-identical** in `operation_only` vs
  `pair_operation`, and in `operation_relation` vs `full_specification`;
* the Relation payload is **byte-identical** in `relation_only` vs
  `pair_relation`.

Only the presence of the Pair block differs.

---

## 6. Conversion report additions (v4)

`conversion_report.json` gains, only for v4 runs:

```text
composite_total                      composite_with_ordered_provenance
composite_missing_provenance         composite_provenance_coverage
operation_trace_count                unique_operation_trace_count
operation_arity_distribution         operation_unsupported_provenance_count
operation_trace_id_distribution      relation_kind_distribution_v4
relation_shortcut_risk_count         matched_control_audit
require_composite_provenance
```

Formal v4 requires `composite_operation_fallback_count == 0`.

---

## 7. Cohorts: v3 and v4 do not share examples

v4 points at a **different** training file:

| | v3 | v4 |
|---|---|---|
| training input | `data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json` | `data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json` |
| rows / sources | 5340 / 2033 | 5340 / 2033 |
| composites | 1949 | 486 |
| composite provenance | **0%** (and no log survives) | **100%** (ordered, arity 2) |
| converted / output | `RQ2/data/converted_v3/`, `RQ2/output/gemma3_4b_nli_grounded_v3/` | `RQ2/data/converted_v4/`, `RQ2/output/gemma3_4b_nli_grounded_v4/` |
| cohort / manifest | `RQ2/data/cohorts/` | `RQ2/data/cohorts_v4/` |

The two cohorts' **source rows are identical** (2033/2033 exact match), but every
composite and most atomic follow-ups differ: the two artifacts are different
generations (0 exact-identity overlap on composite rows). v3 and v4 results are
therefore **not** comparable example-for-example, and v4 cannot be described as
"v3 + provenance".

### Why not enrich the v3 cohort instead?

Metadata-only enrichment was the preferred route and was attempted first. It is
**impossible**:

* the v3 cohort's composites have no `component_mrs` in the artifact;
* no generation log for that run survives — 12 of 13 candidate logs in the
  MTrain repo have zero overlap with the cohort's composite text, and the one
  "hit" is a *rejected* attempt, not a produced row;
* the legacy generator composed composites with
  `random.randint(3, len(invariant_mrs))` and never persisted the selection;
* the provenance-rich artifact shares **0** composite identities with the v3
  cohort.

So the choice was: fabricate provenance, silently swap samples, or build v4 on a
cohort that actually has it. Fabricating was rejected; the third option is taken
and documented here.

---

## 8. Version policy

* `v2` — frozen, unchanged.
* `v3` — frozen. Regenerating v3 reproduces **all 33 output files**
  (`full_train.json`, `full_val.json`, `conversion_report.json` × 11 modes)
  byte-for-byte; verified by SHA-256 against a pre-refactor baseline and guarded
  permanently by `tests/fixtures/rq2_v3_instruction_golden.json`.
* `v4` — new behaviour, current default.

---

## 9. Inspecting v4 instructions

```bash
python scripts/inspect_rq2_instructions.py \
    --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json \
    --pair-id 0 --mr-id composite_flip
```

Prints the row's `mr_id`, `mr_type`, v4 relation kind, `component_mrs` **in
execution order**, `operation_trace_id`, arity, provenance, the shuffled-control
donor, and then the full instruction for every mode. Add `--json` for
machine-readable output, `--template-version 3` to compare against v3.

Provenance audit (no GPU):

```bash
python scripts/audit_rq2_v4_provenance.py \
    --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json
```

---

## 10. Known limitations

1. **`relation_only` / `operation_relation` shortcut risk persists and is not
   papered over.** The `E→C` / `E→N` constraint texts state a conditional
   mapping, which without the Pair block is close to naming the target label.
   On the v4 cohort this covers 2851 / 3307 augmented rows (86%). These cells
   remain **ungrounded metadata diagnostics**; primary Relation conclusions must
   come from `pair_relation`, `full_specification`, `pair_shuffled_relation`.
   The count is emitted as `relation_shortcut_risk_count`. Softening the wording
   to hide this would destroy the specification's meaning, so it is reported
   instead.
2. **The relation control is only ~84% deranged** (§4) — structurally forced by
   the kind imbalance.
3. **Strict wrong-Operation coverage is data-dependent.** The audit must be
   run before training; each `(relation_kind, arity)` stratum with one unique
   trace has zero eligible rows. No cross-stratum fallback is allowed.
4. **Token-length confound.** The v4 cohort's P=1 cells are longer (train p50:
   `none` 71 → `pair_relation` 162 → `full_specification` 203; max 371, 0 rows
   over the 512 cutoff). P contrasts remain length-unmatched.
5. **Arity is not experimentally varied.** Every composite in this cohort is
   arity 2, so v4 cannot separate "more steps" from "composite". A cohort with
   3+ step composites is needed for that question.
6. **Rare MRs.** `antonym_substitution` (13 rows), `voice_switch` (45),
   `synonym_replacement` (33), `composite_inv` (23) are too sparse for per-MR
   claims; only aggregate relation-kind effects are estimable.
7. **Composite share differs between cohorts** (v4: 486/3307 = 15% vs v3:
   1949/3307 = 59%), so absolute accuracy between v3 and v4 is not comparable
   even before the example mismatch.

---

## 11. Running v4

```bash
conda activate llmtrain310

# 8 core conditions x seeds 42/43/44 (24 fine-tunes)
python RQ2/run_rq2_snli.py --config RQ2/configs/rq2_snli_config_v4.json \
  --seeds 42 43 44 \
  --modes none operation_only relation_only operation_relation \
          pair_only pair_operation pair_relation full_specification

# + controls and diagnostic (14 conditions = 42 fine-tunes)
python RQ2/run_rq2_snli.py --config RQ2/configs/rq2_snli_config_v4.json --seeds 42 43 44
```

All conditions share one cohort file, one split manifest and therefore one set
of training examples per seed.
