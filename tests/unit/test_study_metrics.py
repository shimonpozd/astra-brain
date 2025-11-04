import pytest

from prometheus_client import CollectorRegistry

from brain_service.services.study import metrics as study_metrics
from brain_service.services.study.logging import (
    log_bookshelf_built,
    log_daily_bg_loaded,
    log_daily_initial,
    log_prompt_trimmed,
    log_window_built,
)


@pytest.fixture()
def study_metrics_registry():
    registry = CollectorRegistry()
    metrics_obj = study_metrics.StudyMetrics(registry=registry)
    original = study_metrics.get_metrics()
    study_metrics.set_metrics(metrics_obj)
    try:
        yield registry
    finally:
        study_metrics.set_metrics(original)


def _sample(registry: CollectorRegistry, metric: str, labels: dict[str, str] | None = None) -> float | None:
    return registry.get_sample_value(metric, labels)


def test_log_window_built_records_metrics(study_metrics_registry: CollectorRegistry) -> None:
    log_window_built("Genesis 1:1", segments=5, window_size=7, duration_ms=250.0)

    assert _sample(
        study_metrics_registry,
        "study_window_build_duration_seconds_sum",
    ) == pytest.approx(0.25, rel=1e-6)
    assert _sample(study_metrics_registry, "study_window_segments_sum") == pytest.approx(5)
    assert _sample(study_metrics_registry, "study_window_requested_size_sum") == pytest.approx(7)


def test_log_daily_metrics(study_metrics_registry: CollectorRegistry) -> None:
    log_daily_initial("Genesis 1:1", loaded=6, total=18, duration_ms=150.0)
    log_daily_bg_loaded("Genesis 1:1", loaded=4, duration_ms=90.0, retry=True)

    assert _sample(
        study_metrics_registry,
        "study_daily_initial_duration_seconds_sum",
    ) == pytest.approx(0.15, rel=1e-6)
    assert _sample(study_metrics_registry, "study_daily_initial_loaded_segments_sum") == pytest.approx(6)
    assert _sample(study_metrics_registry, "study_daily_initial_total_segments_sum") == pytest.approx(18)
    assert _sample(
        study_metrics_registry,
        "study_daily_background_duration_seconds_sum",
    ) == pytest.approx(0.09, rel=1e-6)
    assert _sample(study_metrics_registry, "study_daily_background_segments_sum") == pytest.approx(4)
    assert _sample(
        study_metrics_registry,
        "study_daily_background_loads_total",
        {"retry": "true"},
    ) == pytest.approx(1)


def test_log_bookshelf_and_prompt_metrics(study_metrics_registry: CollectorRegistry) -> None:
    log_bookshelf_built("Genesis 1:1", items=12)
    log_prompt_trimmed("study", removed_tokens=25, remaining_tokens=125)

    assert _sample(study_metrics_registry, "study_bookshelf_items_sum") == pytest.approx(12)
    assert _sample(
        study_metrics_registry,
        "study_prompt_trimmed_total",
        {"segment": "study"},
    ) == pytest.approx(1)
    assert _sample(study_metrics_registry, "study_prompt_trimmed_tokens_sum") == pytest.approx(25)
    assert _sample(study_metrics_registry, "study_prompt_trimmed_remaining_tokens_sum") == pytest.approx(125)
