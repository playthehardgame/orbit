"""Optional OpenTelemetry instrumentation for Orbit.

Design contract
---------------
* OPT-IN. Nothing happens unless ``ORBIT_OTEL`` is ``1``/``true``/``yes``/``on``.
  Disabled (the default) every helper here is a constant-time no-op: no
  OpenTelemetry import, no network, no allocation beyond a shared singleton.
* OPTIONAL DEPENDENCY. ``opentelemetry-api``/``-sdk``/OTLP exporter are never
  required. If they are not installed, enabling telemetry silently stays a
  no-op. Install them with the ``otel`` extra (see docs/TELEMETRY.md).
* PRIVACY. Prompts, model output, file/sample contents, paths, secrets and
  environment values must never reach a span or metric. Every attribute goes
  through :func:`safe_attributes`, which enforces an explicit key allowlist and
  accepts only bools, numbers and short ``[A-Za-z0-9_.:+-]`` tokens (no spaces,
  no slashes, so prose and paths cannot pass even under an allowlisted key).
  Exception messages are never recorded, only the exception type name.
* NO BEHAVIOUR CHANGE. Telemetry failures are swallowed; instrumentation never
  alters return values, exceptions, exit codes or timing-sensitive logic.

Configuration is via ORBIT_OTEL plus the standard OTEL_* variables
(OTEL_SERVICE_NAME, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_TRACES_EXPORTER=console,
OTEL_METRICS_EXPORTER=console, ...). The exporter is only built when an
endpoint (or a console exporter) is explicitly configured.
"""

from __future__ import annotations

import atexit
import functools
import os
import re
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

_TRUE = {"1", "true", "yes", "on"}
MAX_STR_LEN = 64
_SAFE_STR = re.compile(r"[A-Za-z0-9_.:+\-]{1,%d}" % MAX_STR_LEN)

#: The only attribute keys that may be recorded. Structural metadata only.
ALLOWED_ATTRIBUTES = frozenset(
    {
        "orbit.command",
        "orbit.exit_code",
        "orbit.status",
        "orbit.outcome",
        "orbit.operation",
        "orbit.endpoint",
        "orbit.stream",
        "orbit.model",
        "orbit.tool",
        "orbit.tool.source",
        "orbit.error_type",
        "orbit.http.status",
        "orbit.tokens.prompt",
        "orbit.tokens.completion",
        "orbit.tokens.cached",
        "orbit.finish_reason",
        "orbit.tool_calls.count",
        "orbit.latency_ms",
        "orbit.ttft_ms",
        "orbit.tokens_per_second.prompt",
        "orbit.tokens_per_second.generation",
        "orbit.duration_s",
        "orbit.input_sha256",
        "orbit.code_sha256",
        "orbit.bytes",
        "orbit.count",
        "orbit.attempt",
        "orbit.shards",
        "orbit.resumed",
        "orbit.max_loops",
        "orbit.phase",
        "orbit.transition",
        "orbit.matrix.profile",
        "orbit.matrix.case",
    }
)


def safe_attributes(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only allowlisted, scalar, size-capped attributes; drop the rest."""
    out: dict[str, Any] = {}
    for key, value in (attrs or {}).items():
        if key not in ALLOWED_ATTRIBUTES or value is None:
            continue
        if isinstance(value, (bool, int, float)):
            out[key] = value
        elif isinstance(value, str) and _SAFE_STR.fullmatch(value):
            out[key] = value
    return out


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attrs: Mapping[str, Any]) -> None:
        pass

    def record_exception_type(self, exc: BaseException) -> None:
        pass


_NOOP = _NoopSpan()


class _SafeSpan:
    """Wraps a real span so every attribute passes through the allowlist."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        self.set_attributes({key: value})

    def set_attributes(self, attrs: Mapping[str, Any]) -> None:
        try:
            for key, value in safe_attributes(attrs).items():
                self._span.set_attribute(key, value)
        except Exception:
            pass

    def record_exception_type(self, exc: BaseException) -> None:
        self.set_attributes({"orbit.error_type": type(exc).__name__})


class _State:
    tracer: Any = None
    meter: Any = None
    instruments: dict[str, Any] = {}
    initialized = False
    tracer_provider: Any = None
    meter_provider: Any = None


_state = _State()


def enabled() -> bool:
    return os.environ.get("ORBIT_OTEL", "").strip().lower() in _TRUE


def _init() -> None:
    if _state.initialized:
        return
    _state.initialized = True
    try:
        from opentelemetry import metrics, trace
    except Exception:
        return
    try:
        _setup_sdk()
    except Exception:
        pass
    try:
        _state.tracer = trace.get_tracer("orbit")
        _state.meter = metrics.get_meter("orbit")
    except Exception:
        _state.tracer = _state.meter = None


def _setup_sdk() -> None:
    """Install SDK providers only on explicit exporter configuration."""
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

    env = os.environ
    resource = Resource.create({"service.name": env.get("OTEL_SERVICE_NAME", "orbit")})
    otlp = bool(
        env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or env.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    )
    tp = TracerProvider(resource=resource)
    exporter = None
    if env.get("OTEL_TRACES_EXPORTER", "").strip().lower() == "console":
        exporter = ConsoleSpanExporter()
    elif otlp:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    if exporter is not None:
        tp.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(tp)
        _state.tracer_provider = tp
    readers = []
    if env.get("OTEL_METRICS_EXPORTER", "").strip().lower() == "console":
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    elif otlp:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    if readers:
        mp = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(mp)
        _state.meter_provider = mp
    if _state.tracer_provider is not None or _state.meter_provider is not None:
        atexit.register(shutdown)


def shutdown() -> None:
    for provider in (_state.tracer_provider, _state.meter_provider):
        try:
            if provider is not None:
                provider.shutdown()
        except Exception:
            pass
    _state.tracer_provider = _state.meter_provider = None


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """Context manager yielding a span-like object (no-op when disabled)."""
    if not enabled():
        yield _NOOP
        return
    _init()
    if _state.tracer is None:
        yield _NOOP
        return
    try:
        cm = _state.tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False)
        raw = cm.__enter__()
    except Exception:
        yield _NOOP
        return
    wrapped = _SafeSpan(raw)
    wrapped.set_attributes(attrs)
    try:
        yield wrapped
    except BaseException as exc:
        wrapped.record_exception_type(exc)
        try:
            from opentelemetry.trace import Status, StatusCode

            raw.set_status(Status(StatusCode.ERROR))
        except Exception:
            pass
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def traced(name: str, **static_attrs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of :func:`span` with static (allowlisted) attributes."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not enabled():
                return fn(*args, **kwargs)
            with span(name, **static_attrs):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def _instrument(kind: str, name: str) -> Any:
    key = f"{kind}:{name}"
    inst = _state.instruments.get(key)
    if inst is None:
        if kind == "counter":
            inst = _state.meter.create_counter(name)
        else:
            inst = _state.meter.create_histogram(name)
        _state.instruments[key] = inst
    return inst


def count(name: str, value: int = 1, **attrs: Any) -> None:
    """Increment a counter (no-op when disabled)."""
    if not enabled():
        return
    _init()
    if _state.meter is None:
        return
    try:
        _instrument("counter", name).add(value, safe_attributes(attrs))
    except Exception:
        pass


def record(name: str, value: float, **attrs: Any) -> None:
    """Record a histogram sample (no-op when disabled)."""
    if not enabled():
        return
    _init()
    if _state.meter is None:
        return
    try:
        _instrument("histogram", name).record(value, safe_attributes(attrs))
    except Exception:
        pass


def _install_providers_for_tests(tracer_provider: Any = None, meter_provider: Any = None) -> None:
    """Test hook: bind explicit providers without touching global state."""
    _state.instruments = {}
    _state.initialized = True
    _state.tracer = tracer_provider.get_tracer("orbit") if tracer_provider is not None else None
    _state.meter = meter_provider.get_meter("orbit") if meter_provider is not None else None


def _reset_for_tests() -> None:
    _state.tracer = _state.meter = None
    _state.instruments = {}
    _state.initialized = False
