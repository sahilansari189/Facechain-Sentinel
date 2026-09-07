import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_accuracy.py"
SPEC = importlib.util.spec_from_file_location("check_accuracy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_evaluate_counts_correct_source_and_negative_case():
    results = [
        {"image_path": "a.jpg", "status": "matched", "source_url": "https://Example.com/post/"},
        {"image_path": "b.jpg", "status": "no_match", "source_url": ""},
    ]
    labels = [
        {"image_path": "a.jpg", "expected_source_url": "https://example.com/post"},
        {"image_path": "b.jpg", "expected_match": False},
    ]

    report = MODULE.evaluate(results, labels)

    assert report["accuracy"] == 1.0
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["errors"] == []


def test_evaluate_flags_wrong_positive_source_and_missing_result():
    results = [{"image_path": "a.jpg", "status": "matched", "source_url": "https://wrong.example"}]
    labels = [
        {"image_path": "a.jpg", "expected_source_url": "https://right.example"},
        {"image_path": "missing.jpg", "expected_match": True},
    ]

    report = MODULE.evaluate(results, labels)

    assert report["correct"] == 0
    assert report["fp"] == 1
    assert report["fn"] == 1
    assert len(report["errors"]) == 2