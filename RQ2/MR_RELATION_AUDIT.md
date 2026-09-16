# RQ2 — MR relation audit (source of truth)

This is the evidence table behind `MR_OPERATION_DESCRIPTIONS` and
`MR_RELATION_SPECS` in `scripts/mr_instruction_design.py`.

**Method.** MR definitions were *not* inferred from the current data. Each
relation type is taken from the generator/oracle that produced the data, and only
then cross-checked against the observed `(source_label, follow-up_label)`
transitions. Empirical transitions alone would be circular — they are a
consistency check, not a definition.

## Source-of-truth hierarchy

| # | Source | What it defines |
|---|---|---|
| S1 | `/home/ubuntu/MTrain/augment_snli_all_label_mrs.py` → `OPERATION_SPECS` (and `COMPOSITE_STAGE_PLANS`) | the generator contract: `mr_type`, `target` (`"source"` = preserve, else a literal label), per-MR generation instructions, composite component plans |
| S2 | `/home/ubuntu/MTrain/MRAugment.py` → `NLIMRTool.MR_TYPE` | legacy generator relation taxonomy (used for the `MR_testing/*` files' `mr_category`) |
| S3 | `.research/MetTrain/bare_jrnl_new_sample4.tex` — `R_inv = {r | O_r(L)=L}`, `R_{E→C} = {r | O_r(E)=C}`, `R_{E→N} = {r | O_r(E)=N}` and `tab:allMRs` | the formal partition |
| S4 | `scripts/metamorphic_metrics.py` → `RELATION_TYPES` + the per-relation satisfaction rules | the project's *operational* oracle: `inv` → same prediction; `flip` → entailment↔contradiction, neutral→neutral; `neutral` → `neutral` |
| S5 | the cohort rows' own `mr_type` field | per-row declaration, used only as a consistency check |

The `inv` / `flip` / `neutral` triples in S1, S2, S3, S4 and S5 are mutually
consistent for every MR that actually occurs in the SNLI cohort.

## Audit table

Counts are for `RQ2/data/cohorts/snli_seed42/sampled.json` (5340 rows; 3307
augmented). `op_supported` = the input-side edit is defined by S1/S2.

| mr_id | n_all | n_train | operation spec (input-side) | relation type | relation spec | source of truth | deterministic? | supported for core RQ2 | notes |
|---|---:|---:|---|---|---|---|---|---|---|
| `none` | 2033 | 1932 | *(none — source row)* | — | *(no R block)* | S1 | n/a | yes (as source row) | always plain NLI instruction |
| `composite_inv` | 1433 | 1361 | *fallback* — "applying multiple input-side transformations" | `inv` | preserve source label | S1 `OPERATION_SPECS.composite_inv` (`target="source"`), S2, S3, S4 | yes | yes | component provenance missing in this cohort |
| `adding_contradiction` | 739 | 708 | inserting content that conflicts with the surrounding text | `flip` | fixed mapping; E→C (see S4 map) | S1 (`target="contradiction"`), S2, S3, S4 | yes (conditional on source label) | yes | — |
| `composite_flip` | 363 | 350 | *fallback* | `flip` | fixed mapping; E→C | S1 `...composite_flip` (`target="contradiction"`), S2, S3, S4 | yes | yes | component provenance missing; §limitations |
| `uninformative` | 285 | 274 | inserting content not informative about the rest of the text | `neutral` | follow-up label is `neutral` | S1 (`target="neutral"`), S2, S3, S4 | yes (conditional on source label) | yes | — |
| `conditional_clause` | 269 | 256 | adding a conditional clause to one component | `neutral` | follow-up label is `neutral` | S1 (`target="neutral"`), S2, S3, S4 | yes | yes | — |
| `composite_neutral` | 153 | 146 | *fallback* | `neutral` | follow-up label is `neutral` | S1 `...composite_neutral` (`target="neutral"`), S2, S3, S4 | yes | yes | component provenance missing |
| `synonym_replacement` | 63 | 60 | replacing one or more words with context-appropriate synonyms | `inv` | preserve source label | S1 (`target="source"`), S2, S3, S4 | yes | yes | — |
| `pronoun_substitution` | 1 | 0 | replacing a noun phrase or pronoun with a coreferential form | `inv` | preserve source label | S1 (`target="source"`), S2, S3, S4 | yes | yes (count ≈ 0) | single row, lands in validation |
| `voice_switch` | 1 | 0 | rewriting a sentence between active and passive voice | `inv` | preserve source label | S1 (`target="source"`), S2, S3, S4 | yes | yes (count ≈ 0) | single row, lands in validation |
| `antonym_substitution` | 0 | 0 | replacing content words with context-appropriate antonyms | `flip` | fixed mapping; E→C | S1 (`target="contradiction"`), S2, S3, S4 | yes | yes (not sampled) | defined but absent from this cohort |
| `negation_flip` | 0 | 0 | adding, removing, or reversing a negation marker | `flip` | fixed mapping; E→C | S1 (`target="contradiction"`), S2, S3, S4 | yes | yes (not sampled) | defined but absent from this cohort |
| `race_sensitive_transformation` | 0 | 0 | replacing/modifying demographic or identity-related terms | **unresolved** | **unresolved** | none — appears in no generator, no oracle, no dataset, no `MR_TYPE` entry | unknown | **no** | defensive whitelist only |
| `same_type_named_entity_substitution` | 0 | 0 | replacing named entities with others of the same category | **unresolved** | **unresolved** | none — same as above (nearest analogue `named_entity_replacement: "inv"` in S2 is not this id) | unknown | **no** | defensive whitelist only |
| `tense_shift` | 0 | 0 | shifting the tense of one or more verbs | **unresolved** | **unresolved** | none — same as above | unknown | **no** | defensive whitelist only |

### Conditional vs. deterministic

The `flip` and `neutral` families are **conditional**: S1's eligibility rule only
admits them when the *source* label is `entailment`
(`if int(source["label"]) != LABEL_IDS["entailment"]: return False`). The
relation texts therefore say "entailment maps to contradiction" / "the follow-up
label is neutral" and never claim a mapping from contradiction or neutral
sources. No `flip`/`neutral` row exists in the cohort whose source label is not
`entailment` (verified by transition count).

`inv` MRs are unconditional: the follow-up label equals the source label for any
source label.

No MR in the active set is **set-valued**. The only conditional/set-valued
entries in the whole codebase are `hypernym_substitution: "inv-neutral"` (dead —
shadowed by a duplicate key at `MRAugment.py:2158`) and
`counting_entailment: "inv-flip"`; neither occurs in any dataset here.

### Consistency check (empirical transitions)

`(source_label → follow-up_label)` counts over the full seed-42 cohort. These
**confirm** the definitions above; they were not used to derive them.

| relation type | source label | observed follow-up labels |
|---|---|---|
| `flip` | entailment (0) | contradiction (2) × 1102 |
| `neutral` | entailment (0) | neutral (1) × 707 |
| `inv` | entailment (0) | entailment (0) × 534 |
| `inv` | neutral (1) | neutral (1) × 504 |
| `inv` | contradiction (2) | contradiction (2) × 460 |

No `flip` or `neutral` row originates from a neutral or contradiction source, and
no `inv` row changes its label. The empirical map is fully consistent with the
generator contract (S1) and the project oracle (S4).

## Relation texts (as used in the prompt)

Shared within a relation type — this is intentional, so that the relation text
cannot be used to identify the specific MR (and hence cannot leak a composite's
relation class through wording):

* **inv**
  > For valid source-follow-up pairs generated by this metamorphic relation, the NLI label is preserved: the follow-up label is the same as the source label.

* **flip**
  > For valid source-follow-up pairs generated by this metamorphic relation, the follow-up label is determined from the source label by a fixed label mapping: entailment maps to contradiction, contradiction maps to entailment, and neutral maps to neutral.

* **neutral**
  > For valid source-follow-up pairs generated by this metamorphic relation, the follow-up label is determined from the source label by a fixed label mapping: the follow-up label is neutral.

None of these names the *current* sample's label; each states the abstract
source→follow-up mapping only.

## What was deliberately *not* done

* The old `MR_RELATION_EFFECTS` prose ("likely", "tend to", "may", "generally",
  "unlikely") was **not** kept. It described a probabilistic intuition rather than
  the MR's defined output relation, and for several MRs it was weaker than, or
  inconsistent with, S1/S2/S3/S4 (e.g. `adding_contradiction` "the original
  relation is likely disrupted" vs. the defined E→C mapping).
* No mapping was invented for an MR without a definition. The three unresolved
  ids keep a description but are marked `supported: False`; they occur in no
  dataset.
* Empirical transition counts were **not** used to derive any mapping.

## Composite Operation provenance

The resolver reads a `component_mrs` field when the dataset carries one and
otherwise falls back to one shared generic Operation text.

* **Current cohort** — `data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json`:
  no `component_mrs` on any row. All 1857 composite rows in the seed-42 train
  split use the fallback. Reported as `composite_operation_fallback_count`.
* **Newer artifact** — `data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json`:
  486 composite rows carry `component_mrs`, drawn from the nine 2-stage plans in
  S1's `COMPOSITE_STAGE_PLANS`. The resolver renders them as a numbered list:

  ```
  Input-transformation specification:
  The current sample was derived from its source sample by applying multiple input-side transformations:
  1. Replacing a noun phrase or pronoun with a coreferential form.
  2. Adding, removing, or reversing a negation marker.
  ```

  Switching the RQ2 cohort to that artifact would change the matched metamorphic
  examples and the split, so it was **not** done here. It is a drop-in change once
  the cohort is regenerated deliberately.

* Component plans never leak the output relation: they are input-side edits only.
  `composite_inv` / `composite_flip` / `composite_neutral` do **not** get
  relation-specific Operation wording, even when components are known.
