import pytest
from pydantic import ValidationError

from knowledgator.models import (
    ExtractRequest,
    ExtractResults,
    SpanTaskOutput,
    AnalyzeTaskSpan,
    ScoreTaskOutput,
    AnalyzeTaskScore,
    BooleanTaskOutput,
    ComprehendRequest,
    ComprehendResults,
    AnalyzeTaskBoolean,
    ClassifyTaskOutput,
    RelationTaskOutput,
    AnalyzeTaskClassify,
    AnalyzeTaskRelation,
    parse_task_output,
)
from knowledgator.utils.dictionaries_op import get_inference_tasks

# =========================================================================
# 1. Interface Rules & Validation Tests
# =========================================================================

class TestInputValidationRules:
    """Tests corresponding to analyze_v1_phase1.interface_rules.md on AnalyzeRequest input validation."""

    def test_valid_input_string(self):
        req = ComprehendRequest(
            input="Valid text string",
            tasks={"topic": AnalyzeTaskClassify(type="classify", labels=["tech"])},
        )
        assert req.input.root == "Valid text string"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ComprehendRequest(
                input="",
                tasks={"topic": AnalyzeTaskClassify(type="classify", labels=["tech"])},
            )
        assert "invalid_input" in str(exc.value)

    def test_empty_list_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ComprehendRequest(
                input=[],
                tasks={"topic": AnalyzeTaskClassify(type="classify", labels=["tech"])},
            )
        assert "invalid_input" in str(exc.value)

    def test_empty_dict_rejected(self):
        with pytest.raises(ValidationError) as exc:
            ComprehendRequest(
                input={},
                tasks={"topic": AnalyzeTaskClassify(type="classify", labels=["tech"])},
            )
        assert "invalid_input" in str(exc.value)

    def test_valid_dict_and_list(self):
        req_dict = ComprehendRequest(
            input={"section1": "Text of section 1", "section2": "Text of section 2"},
            tasks={"topic": AnalyzeTaskClassify(type="classify", labels=["tech"])},
        )
        assert isinstance(req_dict.input.root, dict)

        req_list = ComprehendRequest(
            input=["Text 1", "Text 2"],
            tasks={"topic": AnalyzeTaskClassify(type="classify", labels=["tech"])},
        )
        assert isinstance(req_list.input.root, list)


class TestScoreTaskOrderValidation:
    """Tests rule: order must contain unique labels matching labels exactly."""

    def test_valid_score_task(self):
        task = AnalyzeTaskScore(
            type="score",
            labels=["low", "medium", "high"],
            order=["low", "medium", "high"],
        )
        assert task.labels.ids == ["low", "medium", "high"]
        assert task.order == ["low", "medium", "high"]

    def test_score_task_order_mismatch_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeTaskScore(
                type="score",
                labels=["low", "medium", "high"],
                order=["low", "medium"],  # missing 'high'
            )
        assert "invalid_order" in str(exc.value)

    def test_score_task_duplicate_order_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeTaskScore(
                type="score",
                labels=["low", "medium"],
                order=["low", "low"],
            )
        assert "invalid_order" in str(exc.value)


class TestBooleanTaskValidation:
    """Tests rule: boolean task labels must be subset of {'true', 'false'}."""

    def test_valid_boolean_task(self):
        task = AnalyzeTaskBoolean(type="boolean", instruction="Is this scientific?")
        assert task.type == "boolean"

        task_custom = AnalyzeTaskBoolean(
            type="boolean",
            instruction="Is this scientific?",
            labels=["true", "false"],
        )
        assert task_custom.labels.ids == ["true", "false"]

    def test_invalid_boolean_label_rejected(self):
        with pytest.raises(ValidationError) as exc:
            AnalyzeTaskBoolean(
                type="boolean",
                instruction="Is this true?",
                labels=["yes", "no"],
            )
        assert "invalid_label" in str(exc.value)


class TestEndpointTaskRestrictions:
    """ComprehendRequest vs ExtractRequest strict task types."""

    def test_comprehend_request_valid_tasks(self):
        req = ComprehendRequest(
            input="AI is advancing rapidly.",
            tasks={
                "topic": AnalyzeTaskClassify(type="classify", labels=["tech", "science"]),
                "is_ai": AnalyzeTaskBoolean(type="boolean", instruction="Is it about AI?"),
                "level": AnalyzeTaskScore(type="score", labels=["easy", "hard"], order=["easy", "hard"]),
            },
        )
        assert "topic" in req.tasks
        assert "is_ai" in req.tasks
        assert "level" in req.tasks

    def test_comprehend_request_rejects_span_task(self):
        with pytest.raises(ValidationError):
            ComprehendRequest(
                input="Some text",
                tasks={
                    "entities": AnalyzeTaskSpan(type="span", labels=["person"]),
                },
            )

    def test_extract_request_valid_tasks(self):
        req = ExtractRequest(
            input="Steve Jobs founded Apple.",
            tasks={
                "entities": AnalyzeTaskSpan(type="span", labels=["person", "org"]),
                "relations": AnalyzeTaskRelation(
                    type="relation",
                    labels=["founded"],
                    entity_labels=["person", "org"],
                ),
            },
        )
        assert "entities" in req.tasks
        assert "relations" in req.tasks

    def test_extract_request_rejects_classify_task(self):
        with pytest.raises(ValidationError):
            ExtractRequest(
                input="Some text",
                tasks={
                    "topic": AnalyzeTaskClassify(type="classify", labels=["tech"]),
                },
            )


# =========================================================================
# 2. Envelope & Helper Function Tests
# =========================================================================

class TestUnifiedHelper:
    """Tests for get_inference_tasks handling 'outputs' or root."""

    def test_extracts_from_outputs(self):
        payload = {"outputs": {"topic": {"value": "space"}}}
        assert get_inference_tasks(payload) == {"topic": {"value": "space"}}

    def test_fallback_to_raw(self):
        payload = {"topic": {"value": "space"}}
        assert get_inference_tasks(payload) == {"topic": {"value": "space"}}

    def test_handles_empty_outputs_list_or_dict(self):
        assert get_inference_tasks({"outputs": []}) == []
        assert get_inference_tasks({"outputs": {}}) == {}


# =========================================================================
# 3. Typed Output Parsing Tests
# =========================================================================

class TestOutputModelParsing:
    """Tests parsing live run JSON examples from new_api/api_examples_runs/."""

    def test_parse_classify_task_output(self):
        raw = {
            "type": "classify",
            "value": "space",
            "scores": {
                "space": 0.8657,
                "technology": 0.5059,
                "medicine": 0.0841,
                "sports": 0.0076,
            },
            "threshold": 0.5,
        }
        parsed = parse_task_output(raw)
        assert isinstance(parsed, ClassifyTaskOutput)
        assert parsed.value == "space"
        assert parsed.scores["space"] == 0.8657
        assert parsed.threshold == 0.5

    def test_parse_boolean_task_output(self):
        raw = {
            "type": "boolean",
            "value": True,
            "truth_score": 0.9272,
            "falsehood_score": 0.0728,
            "scores": {"true": 0.9272, "false": 0.0728},
            "threshold": 0.5,
        }
        parsed = parse_task_output(raw)
        assert isinstance(parsed, BooleanTaskOutput)
        assert parsed.value is True
        assert parsed.truth_score == 0.9272
        assert parsed.falsehood_score == 0.0728

    def test_parse_score_task_output(self):
        raw = {
            "type": "score",
            "value": "basic",
            "order": ["basic", "intermediate", "advanced"],
            "score": 0.0,
            "scores": {"basic": 0.9785, "intermediate": 0.0202, "advanced": 0.0013},
        }
        parsed = parse_task_output(raw)
        assert isinstance(parsed, ScoreTaskOutput)
        assert parsed.value == "basic"
        assert parsed.order == ["basic", "intermediate", "advanced"]
        assert parsed.score == 0.0

    def test_parse_span_task_output(self):
        raw = {
            "type": "span",
            "value": [
                {"label": "person", "text": "Steve Jobs", "start": 0, "end": 10, "score": 0.99},
                {"label": "organization", "text": "Apple Inc.", "start": 19, "end": 29, "score": 0.99},
            ],
            "threshold": 0.5,
        }
        parsed = parse_task_output(raw)
        assert isinstance(parsed, SpanTaskOutput)
        assert len(parsed.value) == 2
        assert parsed.value[0].label == "person"
        assert parsed.value[0].text == "Steve Jobs"

    def test_parse_relation_task_output(self):
        raw = {
            "type": "relation",
            "value": [
                {
                    "label": "founded",
                    "head": {"label": "person", "text": "Steve Jobs", "start": 0, "end": 10},
                    "tail": {"label": "organization", "text": "Apple Inc.", "start": 19, "end": 29},
                    "score": 0.98,
                }
            ],
        }
        parsed = parse_task_output(raw)
        assert isinstance(parsed, RelationTaskOutput)
        assert len(parsed.value) == 1
        assert parsed.value[0].label == "founded"
        assert parsed.value[0].head["text"] == "Steve Jobs"
        assert parsed.value[0].tail["text"] == "Apple Inc."

    def test_comprehend_results_dual_access_and_options(self):
        raw_response = {
            "outputs": {
                "topic": {
                    "type": "classify",
                    "value": "space",
                    "scores": {"space": 0.9},
                },
                "is_scientific": {
                    "type": "boolean",
                    "value": True,
                    "truth_score": 0.95,
                },
            },
            "usage": {"total_tokens": 42},
        }
        results = ComprehendResults(raw_response)

        # Dictionary access
        assert "topic" in results
        assert results["topic"].value == "space"

        # Attribute access
        assert results.topic.value == "space"
        assert results.is_scientific.value is True

        # .outputs property
        assert "topic" in results.outputs
        assert results.usage == {"total_tokens": 42}

    def test_extract_results_access(self):
        raw_response = {
            "outputs": {
                "entities": {
                    "type": "span",
                    "value": [
                        {"label": "person", "text": "Elon Musk", "start": 0, "end": 9, "score": 0.99}
                    ],
                }
            }
        }
        results = ExtractResults(raw_response)
        assert "entities" in results
        assert results.entities.value[0].text == "Elon Musk"
        assert results["entities"].value[0].label == "person"
