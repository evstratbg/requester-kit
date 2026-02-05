# Changelog

All notable changes to this project will be documented in this file.
Please follow [the Keep a Changelog standard](https://keepachangelog.com/en/1.0.0/).

### Added
- Add `headers` and `cookies` fields to `RequesterKitResponse`.
- Add `error_msg` field to `RequesterKitResponse` for error details.

### Changed
- Add `verify` support to `BaseRequesterKit` constructor for TLS verification configuration.


## [1.1.2] - 2024-11-05


### Added
- Add `py.typed` and package-data for enable mypy ([PEP 561](https://mypy.readthedocs.io/en/stable/installed_packages.html)) 

## [1.1.1] - 2024-11-05


### Added
- Add `__all__` imports

## [1.1.0] - 2024-11-05


### Added
- Add "raw_data" field for RequesterKitResponse to handle files

### Changed
- Rename "data" field to "parsed_data"


## [1.0.0] - 2024-11-01


### Added

- Base client for making http requests
