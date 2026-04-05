# Changelog

All notable changes to this project will be documented in this file.
Please follow [the Keep a Changelog standard](https://keepachangelog.com/en/1.0.0/).

## [0.5.4] - 2026-04-02

### Changed
- Bump package version to `0.6.0`.
- Move Prometheus integration out of `BaseRequesterKit` into a dedicated `requester_kit.metrics` module.
- Replace global metrics toggle with the `@prometheus_metrics(...)` decorator and make metric collection explicit per method.
- Make payload and response size metrics optional via `count_payload_bytes` and `count_response_bytes`.
- Update README examples and release notes to match the new metrics workflow.

## [0.5.0] - 2026-02-24

### Changed
- Bump package version to `0.5.0`.
- Add optional `logger` argument to `BaseRequesterKit` constructor to allow injecting a custom logger instance.
- Rework request/response logging to structured events with `extra` payload fields (`event_name`, `method`, `url`, `status_code`, `total_time_seconds`, `response_body`).
