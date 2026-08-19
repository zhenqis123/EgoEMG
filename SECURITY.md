# Security Policy

## Scope

This policy covers the `egoemg` source code repository during the `0.1.0rc1`
code pre-release. The dataset release, hosted model checkpoints, and
third-party assets (MANO, WiLoR) are out of scope until they are published.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the repository
maintainers by opening a **draft** GitHub security advisory
(`Security > Report a vulnerability`) instead of opening a public issue.
Include reproduction steps and affected commits when possible.

We aim to acknowledge reports within 7 days. Fixes for accepted issues are
released in a patch release or documented as a known limitation in
`docs/PRERELEASE_LIMITATIONS.md`.

## Invariants

- Never commit credentials, dataset download tokens, or private links.
  Download instructions must not embed revocable access tokens.
- Scripts must not send telemetry or fetch code at runtime.
- Dataset handling code must not execute content loaded from data files.
