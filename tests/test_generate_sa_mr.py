"""Offline tests for the SA MR generation engine.

No model, no network, no GPU: the generator and the validators are injected, so
every acceptance and rejection path can be driven deterministically.
"""

import json
import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_sa_mr as engine
import sa_mr_catalog as catalog
from sa_validators import ScriptedValidator


SOURCE_TEXT = "The director was brilliant and the whole cast delivered outstanding work throughout."
SOURCE_TEXT_NEG = "The director was dreadful and the whole cast delivered wooden work throughout."
FLIP_TEXT = "The director was not brilliant and the whole cast delivered poor work throughout."

_SOURCE_RE = re.compile(r"Original text:\n(.*?)\n\nReturn exactly one JSON object", re.DOTALL)
_LABEL_RE = re.compile(r"Original sentiment: (\w+)")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_rows(*texts_and_labels):
    return [
        {"source_id": f"sst2-test-{index:06d}", "text": text, "label": label}
        for index, (text, label) in enumerate(texts_and_labels, start=1)
    ]


class FakeMrGenerator:
    """Replays a scripted reply, deriving the source text from the prompt."""

    def __init__(self, reply=None, *, flatten_case=False, **kwargs):
        self.kwargs = kwargs
        self.reply = reply
        self.calls = []
        self.flatten_case = flatten_case

    def generate(self, messages, seed):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        source_text = _SOURCE_RE.search(prompt).group(1)
        source_label = _LABEL_RE.search(prompt).group(1)
        if self.reply is not None:
            return self.reply(source_text, source_label)
        is_flip = "OPPOSITE overall sentiment" in prompt
        if is_flip:
            label = "negative" if source_label == "positive" else "positive"
            return json.dumps({"text": FLIP_TEXT, "edited_label": label})
        return json.dumps({"text": source_text.replace(" was ", " is ")})


def quiet_factory(labels_by_text, default=None, names=("generator",)):
    """Build a validator_factory returning scripted validators.

    ``labels_by_text`` answers by exact text; ``default`` answers for anything
    not listed, which is what the case-reversal tests need since their follow-up
    differs from the source in letter case alone.
    """

    def factory(args, generator, model_reference):
        return {
            name: ScriptedValidator(labels_by_text, default=default, validator_id=name)
            for name in names
        }

    return factory


def run_engine(tmp_path, rows, *, mrs, labels_by_text, generator=None, default=None):
    """Drive the engine end-to-end with injected generator and validators."""
    source = tmp_path / "sources.jsonl"
    output = tmp_path / "out.jsonl"
    write_jsonl(source, rows)
    argv = [
        "--input", str(source),
        "--dataset", "sst2",
        "--output", str(output),
        "--mrs", *mrs,
        "--num-samples", str(len(rows)),
        "--seed", "42",
        "--label-strategy", "source",
        "--max-retries", "0",
        "--offline",
    ]
    factory = generator if generator is not None else FakeMrGenerator()
    code = engine.main(
        argv,
        generator_factory=lambda **kwargs: factory,
        validator_factory=quiet_factory(labels_by_text, default),
    )
    return code, output


# --------------------------------------------------------------------------- #
# Deterministic MR
# --------------------------------------------------------------------------- #

def test_case_reversal_writes_both_rows_without_calling_the_generator(tmp_path):
    generator = FakeMrGenerator()
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="positive",
        generator=generator,
    )
    assert code == 0
    assert generator.calls == [], "a deterministic MR must not prompt the generator"

    rows = read_jsonl(output)
    assert len(rows) == 2
    source_row, followup_row = rows
    assert source_row["is_source"] is True
    assert followup_row["is_source"] is False
    assert followup_row["text"].lower() == SOURCE_TEXT.lower()
    assert followup_row["text"] != SOURCE_TEXT
    assert followup_row["generator_id"] is None


def test_case_reversal_records_its_template_id_and_relation(tmp_path):
    _, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="positive",
    )
    followup = read_jsonl(output)[1]
    assert followup["template_id"] == "sa_mr/case_reversal/deterministic_v1"
    assert followup["relation_type"] == "preserve"
    assert followup["mr_type"] == "inv"


# --------------------------------------------------------------------------- #
# Row schema and metric bridge
# --------------------------------------------------------------------------- #

def test_accepted_group_carries_the_metric_bridge_keys(tmp_path):
    _, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="positive",
    )
    for row in read_jsonl(output):
        assert row["pair_id"] == row["group_id"]
        assert row["mr_type"] in {"inv", "flip"}
        assert row["task"] == "sentiment_analysis"
        assert row["dataset"] == "sst2"
        assert row["accepted"] is True
        assert row["provenance_hash"]


def test_source_and_followup_share_a_group_id_and_differ_in_sample_id(tmp_path):
    _, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="positive",
    )
    source_row, followup_row = read_jsonl(output)
    assert source_row["group_id"] == followup_row["group_id"]
    assert source_row["sample_id"].endswith("::source")
    assert followup_row["sample_id"].endswith("::followup")
    assert source_row["mr_id"] is None
    assert followup_row["mr_id"] == "sa_case_reversal"


def test_source_row_carries_the_groups_relation_type(tmp_path):
    _, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="positive",
    )
    assert read_jsonl(output)[0]["relation_type"] == "preserve"


# --------------------------------------------------------------------------- #
# Structural gate
# --------------------------------------------------------------------------- #

def test_unchanged_followup_is_rejected(tmp_path):
    generator = FakeMrGenerator(reply=lambda text, label: json.dumps({"text": text}))
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text={},
        generator=generator,
    )
    assert code == 2
    assert not output.exists() or read_jsonl(output) == []


def test_label_word_leak_is_rejected(tmp_path):
    generator = FakeMrGenerator(
        reply=lambda text, label: json.dumps(
            {"text": "The director is weak so this is a positive review overall."}
        )
    )
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text={},
        generator=generator,
    )
    assert code == 2


def test_label_words_already_in_the_source_are_not_treated_as_a_leak(tmp_path):
    """A review that legitimately says 'negative' must not poison its follow-up."""
    text = "The film explores a negative and bleak view of the whole world here."
    _, output = run_engine(
        tmp_path,
        source_rows((text, "negative")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="negative",
    )
    assert len(read_jsonl(output)) == 2


def test_meta_terms_already_in_the_source_are_not_treated_as_a_leak(tmp_path):
    """Regression: the source's own wording must not poison its follow-up.

    Five pilot groups were wrongly rejected because the review itself contained
    "here is the" / "I cannot"; for a deterministic case transform every word is
    inherited from the source, so those rejections could never be genuine.
    """
    text = "The difference here is the acting, and I cannot recommend this film to anyone."
    _, output = run_engine(
        tmp_path,
        source_rows((text, "negative")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="negative",
    )
    assert len(read_jsonl(output)) == 2


def test_meta_terms_introduced_by_the_generator_are_still_rejected(tmp_path):
    """The source-aware comparison must not open a hole for real meta commentary."""
    generator = FakeMrGenerator(
        reply=lambda text, label: json.dumps(
            {"text": "Here is the rewritten review you asked for about a director."}
        )
    )
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text={},
        default="positive",
        generator=generator,
    )
    assert code == 2


def test_meta_commentary_is_rejected(tmp_path):
    generator = FakeMrGenerator(
        reply=lambda text, label: json.dumps(
            {"text": "Here is the transformed version of the original text about a director."}
        )
    )
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text={},
        generator=generator,
    )
    assert code == 2


def test_generator_declining_is_a_skip_not_a_failure(tmp_path):
    generator = FakeMrGenerator(reply=lambda text, label: json.dumps({"text": None}))
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text={},
        generator=generator,
    )
    assert code == 0, "a truthful decline must not be reported as a failure"


def test_not_applicable_source_is_skipped(tmp_path):
    generator = FakeMrGenerator()
    code, _ = run_engine(
        tmp_path,
        source_rows(("Short.", "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text={},
        generator=generator,
    )
    assert code == 0
    assert generator.calls == []


def test_flip_mr_with_an_out_of_range_label_is_rejected(tmp_path):
    generator = FakeMrGenerator(
        reply=lambda text, label: json.dumps({"text": FLIP_TEXT, "edited_label": "positive"})
    )
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_controlled_negation_flip"],
        labels_by_text={},
        generator=generator,
    )
    assert code == 2


# --------------------------------------------------------------------------- #
# Relation gate -- the load-bearing part
# --------------------------------------------------------------------------- #

def test_flip_gate_accepts_a_reversal_the_validator_confirms(tmp_path):
    labels = {SOURCE_TEXT: "positive", FLIP_TEXT: "negative"}
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_controlled_negation_flip"],
        labels_by_text=labels,
    )
    assert code == 0
    followup = read_jsonl(output)[1]
    assert followup["label"] == "negative"
    assert followup["automatic_checks"]["relation_verdict"]["passed"] is True


def test_flip_gate_rejects_when_the_followup_did_not_flip(tmp_path):
    labels = {SOURCE_TEXT: "positive", FLIP_TEXT: "positive"}
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_controlled_negation_flip"],
        labels_by_text=labels,
    )
    assert code == 2


def test_flip_gate_rejects_when_the_source_prior_is_wrong(tmp_path):
    """Both texts read negative: that is not a validated reversal of a positive source."""
    labels = {SOURCE_TEXT: "negative", FLIP_TEXT: "negative"}
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_controlled_negation_flip"],
        labels_by_text=labels,
    )
    assert code == 2


def test_flip_gate_rejects_when_the_source_is_unjudgeable(tmp_path):
    labels = {FLIP_TEXT: "negative"}
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_controlled_negation_flip"],
        labels_by_text=labels,
    )
    assert code == 2


def test_preserve_gate_rejects_when_polarity_changed(tmp_path):
    labels = {SOURCE_TEXT: "positive"}
    generator = FakeMrGenerator(
        reply=lambda text, label: json.dumps(
            {"text": "The director is dull and the whole cast delivered wooden work throughout."}
        )
    )
    code, _ = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text=labels,
        generator=generator,
    )
    assert code == 2


def test_preserve_gate_accepts_when_polarity_is_preserved(tmp_path):
    labels = {SOURCE_TEXT: "positive"}
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text=labels,
        default="positive",
    )
    assert code == 0
    assert len(read_jsonl(output)) == 2


def test_gate_evaluation_records_every_configured_validator():
    verdicts = {
        "generator": {"source": None, "followup": None},
    }
    definition = catalog.get_mr("sa_tense_shift")
    report = engine.evaluate_relation(
        definition=definition,
        source_label="positive",
        verdicts=verdicts,
        gate_mode="generator",
    )
    assert report["passed"] is False
    assert report["validators"]["generator"]["reason"] == "generator_not_run"


def test_gate_evaluation_reports_an_unusable_followup_verdict():
    from sa_validators import ValidationVerdict

    definition = catalog.get_mr("sa_tense_shift")
    report = engine.evaluate_relation(
        definition=definition,
        source_label="positive",
        verdicts={"generator": {"followup": ValidationVerdict("v", error="boom")}},
        gate_mode="generator",
    )
    assert report["passed"] is False
    assert report["validators"]["generator"]["reason"] == "generator_followup_unusable"


# --------------------------------------------------------------------------- #
# Sampling and task construction
# --------------------------------------------------------------------------- #

def test_balanced_sampling_is_exactly_balanced():
    rows = source_rows(*[("A good film here.", "positive")] * 10)
    rows += source_rows(*[("A bad film here.", "negative")] * 10)
    for index, row in enumerate(rows, start=1):
        row["source_id"] = f"s-{index:06d}"
    sampled = engine.sample_sources(rows, 8, seed=42, label_strategy="balanced")
    labels = [row["label"] for row in sampled]
    assert labels.count("positive") == 4
    assert labels.count("negative") == 4


def test_sampling_without_replacement_is_enforced():
    rows = source_rows((SOURCE_TEXT, "positive"))
    with pytest.raises(ValueError, match="without replacement"):
        engine.sample_sources(rows, 5, seed=42, label_strategy="source")


def test_all_assignment_gives_every_source_every_mr():
    rows = source_rows((SOURCE_TEXT, "positive"), (SOURCE_TEXT_NEG, "negative"))
    sources = engine.sample_sources(rows, 2, seed=1, label_strategy="source")
    tasks = engine.select_tasks(sources, ["sa_tense_shift", "sa_voice_switch"], "sst2", "all")
    assert len(tasks) == 4
    assert {task.mr_id for task in tasks} == {"sa_tense_shift", "sa_voice_switch"}


def test_all_assignment_gives_every_mr_the_same_source_denominator():
    """The pilot's whole design rests on this: one shared source pool."""
    rows = source_rows((SOURCE_TEXT, "positive"), (SOURCE_TEXT_NEG, "negative"))
    sources = engine.sample_sources(rows, 2, seed=1, label_strategy="source")
    tasks = engine.select_tasks(sources, catalog.MR_ORDER, "sst2", "all")
    by_mr = {}
    for task in tasks:
        by_mr.setdefault(task.mr_id, []).append(task.source_id)
    reference = by_mr["sa_tense_shift"]
    for mr_id, ids in by_mr.items():
        assert ids == reference, f"{mr_id} sees a different source pool"


def test_round_robin_assignment_spreads_mrs():
    rows = source_rows(*[("A good film here.", "positive")] * 4)
    for index, row in enumerate(rows, start=1):
        row["source_id"] = f"s-{index:06d}"
    sources = engine.sample_sources(rows, 4, seed=1, label_strategy="source")
    tasks = engine.select_tasks(
        sources, ["sa_tense_shift", "sa_voice_switch", "sa_tense_shift", "sa_voice_switch"], "sst2", "round_robin"
    )
    assert len(tasks) == 4


def test_task_group_id_is_stable_and_structured():
    rows = source_rows((SOURCE_TEXT, "positive"))
    sources = engine.sample_sources(rows, 1, seed=1, label_strategy="source")
    task = engine.select_tasks(sources, ["sa_tense_shift"], "sst2", "all")[0]
    assert task.group_id == "sst2-test-000001::sa_tense_shift::v1"


# --------------------------------------------------------------------------- #
# Resume and dry-run
# --------------------------------------------------------------------------- #

def test_resume_does_not_regenerate_completed_groups(tmp_path):
    labels = {SOURCE_TEXT: "positive"}
    rows = source_rows((SOURCE_TEXT, "positive"))
    generator = FakeMrGenerator()
    code, output = run_engine(
        tmp_path,
        rows,
        mrs=["sa_tense_shift"],
        labels_by_text=labels,
        default="positive",
        generator=generator,
    )
    assert code == 0
    first_call_count = len(generator.calls)

    source = tmp_path / "sources.jsonl"
    code = engine.main(
        [
            "--input", str(source),
            "--dataset", "sst2",
            "--output", str(output),
            "--mrs", "sa_tense_shift",
            "--num-samples", "1",
            "--seed", "42",
            "--label-strategy", "source",
            "--max-retries", "0",
            "--offline",
            "--resume",
        ],
        generator_factory=lambda **kwargs: generator,
        validator_factory=quiet_factory(labels),
    )
    assert code == 0
    assert len(generator.calls) == first_call_count, "resume must not re-prompt"
    assert len(read_jsonl(output)) == 2


def test_resume_rejects_an_inconsistent_existing_row(tmp_path):
    source = tmp_path / "sources.jsonl"
    output = tmp_path / "out.jsonl"
    write_jsonl(source, source_rows((SOURCE_TEXT, "positive")))
    write_jsonl(
        output,
        [
            {"group_id": "sst2-test-000001::sa_tense_shift::v1", "is_source": True,
             "source_id": "wrong-id", "mr_id": None},
            {"group_id": "sst2-test-000001::sa_tense_shift::v1", "is_source": False,
             "source_id": "wrong-id", "mr_id": "sa_tense_shift"},
        ],
    )
    code = engine.main(
        [
            "--input", str(source), "--dataset", "sst2", "--output", str(output),
            "--mrs", "sa_tense_shift", "--num-samples", "1", "--seed", "42",
            "--label-strategy", "source", "--resume", "--offline",
        ],
        generator_factory=lambda **kwargs: FakeMrGenerator(),
        validator_factory=quiet_factory({}),
    )
    assert code == 2


def test_dry_run_never_constructs_a_model(tmp_path, capsys):
    source = tmp_path / "sources.jsonl"
    write_jsonl(source, source_rows((SOURCE_TEXT, "positive")))

    def exploding_factory(**kwargs):
        raise AssertionError("dry-run must not build a generator")

    code = engine.main(
        [
            "--input", str(source), "--dataset", "sst2",
            "--mrs", "sa_tense_shift", "--num-samples", "1",
            "--seed", "42", "--label-strategy", "source", "--dry-run",
        ],
        generator_factory=exploding_factory,
    )
    assert code == 0
    assert "sa_tense_shift" in capsys.readouterr().out


def test_dry_run_previews_the_deterministic_mr_without_a_prompt(tmp_path, capsys):
    source = tmp_path / "sources.jsonl"
    write_jsonl(source, source_rows((SOURCE_TEXT, "positive")))
    code = engine.main(
        [
            "--input", str(source), "--dataset", "sst2",
            "--mrs", "sa_case_reversal", "--num-samples", "1",
            "--seed", "42", "--label-strategy", "source", "--dry-run",
        ],
        generator_factory=lambda **kwargs: (_ for _ in ()).throw(AssertionError("no model")),
    )
    assert code == 0
    assert "deterministic" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def test_unparseable_response_is_retried_not_fatal(tmp_path):
    """Regression: one malformed reply used to abort the entire run.

    ``extract_json_object`` raises the base ``GenerationError`` while the loop only
    caught ``SAMRGenerationError`` / ``MRDeclined``, so a non-JSON reply escaped to
    ``main()`` and killed the pilot mid-flight.  A truncated IMDb rewrite is
    enough to trigger it, which is exactly what happened on the real run.
    """
    generator = FakeMrGenerator(
        reply=lambda text, label: "I'm sorry, I can't rewrite that review."
    )
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text={},
        default="positive",
        generator=generator,
    )
    assert code == 2, "a malformed reply is a failed group, not an aborted run"
    report = json.loads(Path(str(output) + ".report.json").read_text(encoding="utf-8"))
    assert report["accepted_groups"] == 0
    reasons = report["rejected_attempts_by_reason"]
    assert any("unparseable_response" in str(key) for key in reasons), reasons


def test_unparseable_response_leaves_other_groups_untouched(tmp_path):
    """A malformed reply for one source must not cost the others their result."""
    calls = {"n": 0}

    def reply(text, label):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps({"text": text.replace(" was ", " is ")})

    generator = FakeMrGenerator(reply=reply)
    rows = source_rows((SOURCE_TEXT, "positive"), (SOURCE_TEXT_NEG, "negative"))
    # The preserve gate needs each follow-up to read like its own source, and the
    # two sources have opposite labels, so key the validator by text.
    labels = {
        SOURCE_TEXT.replace(" was ", " is "): "positive",
        SOURCE_TEXT_NEG.replace(" was ", " is "): "negative",
    }
    code, output = run_engine(
        tmp_path,
        rows,
        mrs=["sa_tense_shift"],
        labels_by_text=labels,
        generator=generator,
    )
    followups = [row for row in read_jsonl(output) if not row["is_source"]]
    assert len(followups) == 1, "the healthy group must still be written"
    assert code == 2, "the malformed group is still reported as a failure"


def test_deterministic_gate_failure_is_terminal_not_a_crash(tmp_path):
    """Regression: a failed deterministic MR must not be queued for a retry.

    ``sa_case_reversal`` has no prompt, so sending it back through the retry loop
    used to abort the whole run with "deterministic and has no prompt" instead of
    recording one failed group.
    """
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_case_reversal"],
        labels_by_text={},
        default="negative",  # claims the case-reversed text flipped polarity
    )
    assert code == 2
    report = json.loads(Path(str(output) + ".report.json").read_text(encoding="utf-8"))
    assert report["accepted_groups"] == 0
    assert report["per_mr"]["sa_case_reversal"]["accepted_groups"] == 0


def test_deterministic_failure_does_not_block_other_mrs(tmp_path):
    """A failed deterministic group must leave the LLM-backed MRs unaffected."""
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_case_reversal", "sa_tense_shift"],
        labels_by_text={},
        default="positive",  # preserve gate passes for both
    )
    assert code == 0
    followups = [row for row in read_jsonl(output) if not row["is_source"]]
    assert {row["mr_id"] for row in followups} == {"sa_case_reversal", "sa_tense_shift"}


def test_report_records_catalog_hashes_and_per_mr_counts(tmp_path):
    labels = {SOURCE_TEXT: "positive"}
    code, output = run_engine(
        tmp_path,
        source_rows((SOURCE_TEXT, "positive")),
        mrs=["sa_tense_shift"],
        labels_by_text=labels,
        default="positive",
    )
    assert code == 0
    report = json.loads(
        Path(str(output) + ".report.json").read_text(encoding="utf-8")
    )
    assert report["catalog_version"] == catalog.MR_CATALOG_VERSION
    assert report["catalog_sha256"] == catalog.catalog_sha256()
    assert report["complete"] is True
    assert report["accepted_groups"] == 1
    assert report["per_mr"]["sa_tense_shift"]["accepted_groups"] == 1
    assert report["per_mr"]["sa_tense_shift"]["relation_type"] == "preserve"
    assert report["validator_ids"]


def test_structural_checks_report_similarity_and_length_ratio():
    definition = catalog.get_mr("sa_tense_shift")
    checks = engine.run_structural_checks(
        SOURCE_TEXT, SOURCE_TEXT.replace(" was ", " is "), definition, max_chars=6000
    )
    assert checks["diff_nonempty"] is True
    assert 0.0 < checks["similarity"] <= 1.0
    assert checks["length_ratio"] == pytest.approx(1.0, abs=0.05)


def test_temperature_override_runs_the_blind_pass_greedily_and_restores_state():
    class RecordingGenerator:
        def __init__(self):
            self.temperature = 0.8
            self.top_p = 0.95
            self.seen = []

        def generate(self, messages, seed):
            self.seen.append((self.temperature, self.top_p))
            return json.dumps({"predicted_label": "positive", "unambiguous": True, "reason": "r"})

    inner = RecordingGenerator()
    wrapper = engine.TemperatureOverrideGenerator(inner, temperature=0.0, top_p=1.0)
    wrapper.generate([{"role": "user", "content": "x"}], seed=1)

    assert inner.seen == [(0.0, 1.0)], "verification must sample greedily"
    assert (inner.temperature, inner.top_p) == (0.8, 0.95), "generation settings must be restored"


def test_temperature_override_restores_state_after_a_failure():
    class BoomGenerator:
        def __init__(self):
            self.temperature = 0.8

        def generate(self, messages, seed):
            raise RuntimeError("boom")

    inner = BoomGenerator()
    wrapper = engine.TemperatureOverrideGenerator(inner, temperature=0.0)
    with pytest.raises(RuntimeError):
        wrapper.generate([{"role": "user", "content": "x"}], seed=1)
    assert inner.temperature == 0.8


def test_generator_validator_is_built_with_greedy_verification(tmp_path):
    """The default validator backend must not inherit the generator's sampling."""
    args = engine.build_parser().parse_args(
        [
            "--input", "x", "--dataset", "imdb", "--dry-run",
            "--mrs", "sa_tense_shift",
        ]
    )
    assert args.verifier_temperature == 0.0


def test_structural_checks_reject_a_case_transform_that_changes_words():
    definition = catalog.get_mr("sa_case_reversal")
    with pytest.raises(engine.SAMRGenerationError, match="case_transform_changed_words"):
        engine.run_structural_checks(
            SOURCE_TEXT, "A completely different sentence entirely.", definition, max_chars=6000
        )
