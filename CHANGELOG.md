# Changelog

## [Unreleased]

- Precomputed exact per-quality Bayesian likelihood vectors and specialized the canonical integer-Phred hot path while retaining accepted subclass semantics.
- Added a deterministic end-to-end heterogeneous-group consensus benchmark with exact result checksums.


## [1.0.0] - 2026-08-12

First stable release.

- Phred-aware Bayesian consensus calling for aligned reads.
- Added a deterministic property-based suite for consensus invariants and input-order independence.
- Reached 729/764 mutants killed (95.42%); reviewed the remaining 35 as behavior-equivalent under the validated public contract.
- Fixed duplicate prior keys being silently overwritten; duplicate keys are now rejected.
- Adopted strict mypy checking and shipped typed-package metadata.
- Expanded CI to Linux and macOS on Python 3.11–3.13 with 100% statement and branch coverage enforced.
