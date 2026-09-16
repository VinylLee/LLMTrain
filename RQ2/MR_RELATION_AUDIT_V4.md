# RQ2 v4 — MR relation audit (source of truth)

Evidence table behind `V4_ATOMIC_OPERATION_STEPS`, `MR_V4_RELATION_KINDS` and
`V4_RELATION_KIND_DESCRIPTIONS` in `scripts/mr_instruction_design.py`.

Method unchanged from the v3 audit (`MR_RELATION_AUDIT.md`): definitions come
from the generator/oracle, never from the observed data. Empirical transitions
are a consistency check only. This document records what v4 changed and why.

## Source-of-truth hierarchy

| # | Source | What it defines |
|---|---|---|
| S1 | `/home/ubuntu/MTrain/augment_snli_all_label_mrs.py` → `OPERATION_SPECS` + `COMPOSITE_STAGE_PLANS` | generator contract: `mr_type`, `target` (`"source"` = preserve, else a literal label), per-MR generation instructions, composite stage plans |
| S2 | `/home/ubuntu/MTrain/MRAugment.py` → `NLIMRTool.MR_TYPE` | legacy generator taxonomy (`inv` / `flip` / `neutral`) |
| S3 | `.research/MetTrain/bare_jrnl_new_sample4.tex` — `R_inv = {r \| O_r(L)=L}`, `R_{E→C} = {r \| O_r(E)=C}`, `R_{E→N} = {r \| O_r(E)=N}` | formal partition |
| S4 | `scripts/metamorphic_metrics.py` → `RELATION_TYPES` + satisfaction rules | the project's *metric operationalisation* |
| S5 | per-row `mr_type` in the cohort | per-row declaration, consistency check only |

---

## The `flip` decision: totalised map vs applicability-restricted branch

v3 rendered:

> …the follow-up label is determined from the source label by a fixed label mapping:
> entailment maps to contradiction, **contradiction maps to entailment, and neutral maps to neutral**.

v4 renders:

> For a valid source-follow-up pair **to which this relation applies**, the output labels must
> satisfy the following constraint: if the source label is entailment, the follow-up label must
> be contradiction.

Why the narrower form is the correct one:

* **The generator never produces the other branches.** S1's `OPERATION_SPECS`
  gives `target = "contradiction"` and its eligibility rule rejects any source
  whose label is not `entailment`:
  ```python
  if int(source["label"]) != LABEL_IDS["entailment"]:
      return False
  ```
* **The paper's formal definition is already restricted.** S3 defines
  `R_{E→C} = { r | O_r(E) = C }` — a relation over entailment sources, not a
  total function on all labels.
* **The 3-class map comes from the metric, not the MR.** S4's
  `entailment↔contradiction, neutral→neutral` table exists to make
  *satisfaction counting* total over whatever prediction the model emits. It is
  an evaluation convenience, not a statement about what the MR does.
* **Empirically only one branch occurs.** In the v4 cohort every `flip` row has
  `source_label = entailment` and `follow-up label = contradiction`.

Therefore v4 does **not** import `C→E` or `N→N` into the prompt. Writing them
would assert a mapping the generator never instantiates, i.e. it would be
inventing a relation.

The legacy `mr_type` field is untouched: the semantic resolver
(`mr_type`/`mr_id` → v4 kind) keeps the data compatible.

---

## Audit table

Counts from the v4 cohort (`snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json`,
5340 rows / 3307 augmented). `train n` is the seed-42 train split (5102 rows; 3170 augmented).

| mr_id | n_all | train n | v4 relation kind | legacy `mr_type` | applicability | operation step source | supported |
|---|---:|---:|---|---|---|---|---|
| `none` | 2033 | 1932 | *(source row — no R block)* | `inv` | n/a | — | yes (as source row) |
| `uninformative` | 1150 | 1100 | `entailment_to_neutral` | `neutral` | source label = entailment | S1 `target="neutral"` | yes |
| `conditional_clause` | 565 | 537 | `entailment_to_neutral` | `neutral` | source label = entailment | S1 | yes |
| `negation_flip` | 480 | 462 | `entailment_to_contradiction` | `flip` | source label = entailment | S1 `target="contradiction"` | yes |
| `pronoun_substitution` | 355 | 339 | `invariance` | `inv` | unconditional | S1 `target="source"` | yes |
| `composite_flip` | 255 | 246 | `entailment_to_contradiction` | `flip` | source label = entailment | S1 `COMPOSITE_STAGE_PLANS` + per-row `component_mrs` | yes |
| `composite_neutral` | 208 | 200 | `entailment_to_neutral` | `neutral` | source label = entailment | idem | yes |
| `adding_contradiction` | 180 | 175 | `entailment_to_contradiction` | `flip` | source label = entailment | S1 | yes |
| `voice_switch` | 45 | 45 | `invariance` | `inv` | unconditional | S1 | yes |
| `synonym_replacement` | 33 | 32 | `invariance` | `inv` | unconditional | S1 | yes |
| `composite_inv` | 23 | 22 | `invariance` | `inv` | unconditional | S1 + `component_mrs` | yes |
| `antonym_substitution` | 13 | 12 | `entailment_to_contradiction` | `flip` | source label = entailment | S1 | yes |
| `race_sensitive_transformation` | 0 | 0 | **unresolved** | — | unknown | none | **no** (defensive whitelist) |
| `same_type_named_entity_substitution` | 0 | 0 | **unresolved** | — | unknown | none | **no** |
| `tense_shift` | 0 | 0 | **unresolved** | unknown | none | **no** |

Relation-kind totals (all rows): `entailment_to_neutral` 1923,
`entailment_to_contradiction` 928, `invariance` 456.

### Consistency check (empirical transitions, v4 cohort)

| relation kind | source label | observed follow-up labels |
|---|---|---|
| `entailment_to_contradiction` | entailment (0) | contradiction (2) × 928 |
| `entailment_to_neutral` | entailment (0) | neutral (1) × 1923 |
| `invariance` | entailment (0) | entailment (0) × 429 |
| `invariance` | neutral (1) | neutral (1) × 7 |
| `invariance` | contradiction (2) | contradiction (2) × 20 |

No `E→C` or `E→N` row originates from a non-entailment source. Fully consistent
with S1/S3.

---

## Composite operation provenance

v4 renders the real ordered sequence; see `RQ2_INSTRUCTION_DESIGN_V4.md` §2.

* Coverage in the v4 cohort: **486 / 486 composites (100%)**, all arity 2, no
  nested composites, no unknown components, no repeated components.
* 9 distinct ordered traces come from S1's `COMPOSITE_STAGE_PLANS`.
* Component selection is **never** derived from the composite's own `mr_id`.
  If two different composite types share a `component_mrs`, their Operation
  blocks are byte-identical.
* If provenance is missing, v4 fails a formal run rather than falling back.

### Ordered traces present in the cohort

| ordered trace | arity | n | relation kind |
|---|---:|---:|---|
| `pronoun_substitution → negation_flip` | 2 | 248 | E→C |
| `pronoun_substitution → uninformative` | 2 | 177 | E→N |
| `voice_switch → uninformative` | 2 | 20 | E→N |
| `pronoun_substitution → synonym_replacement` | 2 | 18 | invariance |
| `synonym_replacement → uninformative` | 2 | 11 | E→N |
| `synonym_replacement → pronoun_substitution` | 2 | 4 | invariance |
| `voice_switch → adding_contradiction` | 2 | 4 | E→C |
| `synonym_replacement → negation_flip` | 2 | 3 | E→C |
| `voice_switch → synonym_replacement` | 2 | 1 | invariance |

Plus 8 atomic traces (arity 1): `uninformative` 1150, `conditional_clause` 565,
`negation_flip` 480, `pronoun_substitution` 355, `adding_contradiction` 180,
`voice_switch` 45, `synonym_replacement` 33, `antonym_substitution` 13.

Note the trace-based view makes a confound visible that the mr_id-based view
hides: the four `→ uninformative` composites (177+20+11 = 208) are rendered as
"insert irrelevant content" with the same first step as other composites, yet
they carry the E→N constraint. Trace composition and relation kind are
correlated by construction of the generator's stage plans.

---

## What was deliberately not done

* No mapping invented for an MR without a definition (the three unresolved ids).
* No `C→E` / `N→N` branch imported from the metric into the prompt.
* No empirical transitions used to derive any mapping.
* No hedging introduced to hide the ungrounded-Relation shortcut (§10 of the
  design doc); it is reported instead.
