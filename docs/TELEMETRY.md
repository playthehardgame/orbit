# Optional OpenTelemetry

Orbit can emit OpenTelemetry traces and metrics. It is **off by default** and
**not a dependency**: with telemetry disabled, or with the `opentelemetry`
packages absent, every instrumentation call is a constant-time no-op (nothing
is imported, no network is touched, behavior, exit codes, cache keys and model
inputs are unchanged).

## Enable

    pip install -e '.[otel]'          # api + sdk + OTLP/HTTP exporter
    export ORBIT_OTEL=1
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    export OTEL_SERVICE_NAME=orbit     # optional, default "orbit"
    orbit ...

`OTEL_TRACES_EXPORTER=console` / `OTEL_METRICS_EXPORTER=console` print to
stdout instead. Exporters are only built when an endpoint or console exporter
is explicitly configured. An application that already installs its own
global providers is also honoured (Orbit only calls `get_tracer`/`get_meter`).

## What is emitted

Spans: `orbit.cli` (command, exit code), `orbit.runtime.tool_loop`,
`orbit.model.request` (prompt/completion/cached token counts, finish reason,
latency, TTFT, tokens/s), `orbit.server.health`, `orbit.server.request`
(method, known endpoint name), `orbit.server.model_load`,
`orbit.tool.execute` (tool name, source, outcome),
`orbit.analysis.sandbox_execute` (status, duration, input/code SHA-256),
`orbit.download.fetch` (resumed, bytes).

Metrics: `orbit.model.requests`, `orbit.model.request.duration_ms`,
`orbit.model.tokens.prompt|completion`, `orbit.tool.executions`,
`orbit.analysis.sandbox_runs`,
`orbit.download.completed`.

## Privacy

Prompts, model output, file or sample contents, paths, secrets and environment
values are never recorded. All attributes pass through
`orbit.telemetry.safe_attributes`: an explicit key allowlist, scalars only,
strings limited to 64 characters of `[A-Za-z0-9_.:+-]` (no spaces or slashes).
Exceptions record only the exception type name. Add new attribute keys to
`ALLOWED_ATTRIBUTES` deliberately; `tests/test_telemetry.py` guards this.

## Instrumenting a script (e.g. the model-matrix driver)

    from orbit import telemetry
    with telemetry.span("orbit.matrix.case", **{"orbit.matrix.profile": pid}) as sp:
        ...
        sp.set_attribute("orbit.status", "pass")

## Hash attributes are an opt-in disclosure

`orbit.input_sha256` and `orbit.code_sha256` on the ANALYSIS execution span are
SHA-256 digests of the analysed input and of the generated code. They carry no
content, but a digest can identify a known sample or confirm a guessed file.
Enable telemetry only toward collectors you trust to hold that information.
