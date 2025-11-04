"""Prometheus-backed metrics for the study service."""

from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - optional dependency loaded at runtime
    from prometheus_client import CollectorRegistry, Counter, Histogram
except ModuleNotFoundError:  # pragma: no cover - graceful fallback
    CollectorRegistry = None  # type: ignore
    Counter = Histogram = None  # type: ignore


class StudyMetrics:
    """Container for study-domain Prometheus metrics."""

    def __init__(self, *, registry: CollectorRegistry | None = None) -> None:
        if Counter is None or Histogram is None:  # pragma: no cover - defensive
            raise RuntimeError("prometheus_client is required to use StudyMetrics")

        histogram_kwargs = {"registry": registry} if registry is not None else {}
        counter_kwargs = {"registry": registry} if registry is not None else {}

        # Window instrumentation -------------------------------------------------
        self.window_build_duration = Histogram(
            "study_window_build_duration_seconds",
            "Latency for assembling study windows.",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            **histogram_kwargs,
        )
        self.window_segments = Histogram(
            "study_window_segments",
            "Segments included in study window responses.",
            buckets=(0, 1, 2, 3, 5, 8, 12, 20, 30, 50),
            **histogram_kwargs,
        )
        self.window_requested = Histogram(
            "study_window_requested_size",
            "Requested window sizes.",
            buckets=(0, 1, 2, 3, 5, 8, 12, 20, 30, 50),
            **histogram_kwargs,
        )

        # Daily mode instrumentation --------------------------------------------
        self.daily_initial_duration = Histogram(
            "study_daily_initial_duration_seconds",
            "Latency for planning + loading initial daily segments.",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
            **histogram_kwargs,
        )
        self.daily_initial_loaded = Histogram(
            "study_daily_initial_loaded_segments",
            "Segments delivered in the initial daily payload.",
            buckets=(0, 5, 10, 15, 20, 30, 40, 60, 80, 120),
            **histogram_kwargs,
        )
        self.daily_initial_total = Histogram(
            "study_daily_initial_total_segments",
            "Total segments detected for a daily payload.",
            buckets=(0, 20, 40, 60, 80, 120, 160, 200, 300, 400),
            **histogram_kwargs,
        )
        self.daily_background_duration = Histogram(
            "study_daily_background_duration_seconds",
            "Latency for background segment loading batches.",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20),
            **histogram_kwargs,
        )
        self.daily_background_segments = Histogram(
            "study_daily_background_segments",
            "Segments appended during background loading batches.",
            buckets=(0, 1, 2, 3, 5, 8, 12, 20, 30, 50),
            **histogram_kwargs,
        )
        self.daily_background_loads = Counter(
            "study_daily_background_loads_total",
            "Number of background load executions.",
            ["retry"],
            **counter_kwargs,
        )

        # Bookshelf instrumentation ---------------------------------------------
        self.bookshelf_items = Histogram(
            "study_bookshelf_items",
            "Number of bookshelf items returned per request.",
            buckets=(0, 5, 10, 15, 20, 30, 40, 60, 80),
            **histogram_kwargs,
        )

        # Prompt instrumentation -------------------------------------------------
        self.prompt_trimmed_total = Counter(
            "study_prompt_trimmed_total",
            "Number of prompt segments trimmed due to budgeting.",
            ["segment"],
            **counter_kwargs,
        )
        self.prompt_trimmed_tokens = Histogram(
            "study_prompt_trimmed_tokens",
            "Tokens removed from prompts during trimming.",
            buckets=(0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200),
            **histogram_kwargs,
        )
        self.prompt_trimmed_remaining = Histogram(
            "study_prompt_trimmed_remaining_tokens",
            "Tokens remaining after trimming for a prompt segment.",
            buckets=(0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200),
            **histogram_kwargs,
        )

    # --------------------------------------------------------------------- API -
    def record_window_built(self, *, segments: int, window_size: int, duration_ms: float) -> None:
        self.window_build_duration.observe(max(duration_ms, 0.0) / 1000.0)
        self.window_segments.observe(max(segments, 0))
        self.window_requested.observe(max(window_size, 0))

    def record_daily_initial(self, *, loaded: int, total: int, duration_ms: float) -> None:
        self.daily_initial_duration.observe(max(duration_ms, 0.0) / 1000.0)
        self.daily_initial_loaded.observe(max(loaded, 0))
        self.daily_initial_total.observe(max(total, 0))

    def record_daily_background(
        self,
        *,
        segments: int,
        duration_ms: float,
        retry: bool,
    ) -> None:
        self.daily_background_duration.observe(max(duration_ms, 0.0) / 1000.0)
        self.daily_background_segments.observe(max(segments, 0))
        self.daily_background_loads.labels(retry="true" if retry else "false").inc()

    def record_bookshelf(self, *, items: int) -> None:
        self.bookshelf_items.observe(max(items, 0))

    def record_prompt_trim(self, *, segment: str, removed_tokens: int, remaining_tokens: int) -> None:
        self.prompt_trimmed_total.labels(segment=segment).inc()
        self.prompt_trimmed_tokens.observe(max(removed_tokens, 0))
        self.prompt_trimmed_remaining.observe(max(remaining_tokens, 0))


_default_metrics: Optional[StudyMetrics]
if Counter is None or Histogram is None:  # pragma: no cover - optional dependency missing
    _default_metrics = None
else:
    _default_metrics = StudyMetrics()

_metrics: Optional[StudyMetrics] = _default_metrics


def set_metrics(metrics: Optional[StudyMetrics]) -> None:
    """Override the global metrics collector (primarily for tests)."""

    global _metrics
    _metrics = metrics


def reset_metrics() -> None:
    """Reset the global metrics collector to the default instance."""

    global _metrics
    _metrics = _default_metrics


def get_metrics() -> Optional[StudyMetrics]:
    """Return the current metrics collector, if metrics are enabled."""

    return _metrics


def record_window_built(*, segments: int, window_size: int, duration_ms: float) -> None:
    metrics = get_metrics()
    if metrics is not None:
        metrics.record_window_built(segments=segments, window_size=window_size, duration_ms=duration_ms)


def record_daily_initial(*, loaded: int, total: int, duration_ms: float) -> None:
    metrics = get_metrics()
    if metrics is not None:
        metrics.record_daily_initial(loaded=loaded, total=total, duration_ms=duration_ms)


def record_daily_background(*, segments: int, duration_ms: float, retry: bool) -> None:
    metrics = get_metrics()
    if metrics is not None:
        metrics.record_daily_background(segments=segments, duration_ms=duration_ms, retry=retry)


def record_bookshelf(*, items: int) -> None:
    metrics = get_metrics()
    if metrics is not None:
        metrics.record_bookshelf(items=items)


def record_prompt_trim(*, segment: str, removed_tokens: int, remaining_tokens: int) -> None:
    metrics = get_metrics()
    if metrics is not None:
        metrics.record_prompt_trim(
            segment=segment,
            removed_tokens=removed_tokens,
            remaining_tokens=remaining_tokens,
        )


__all__ = [
    "StudyMetrics",
    "CollectorRegistry",
    "get_metrics",
    "record_bookshelf",
    "record_daily_background",
    "record_daily_initial",
    "record_prompt_trim",
    "record_window_built",
    "reset_metrics",
    "set_metrics",
]
