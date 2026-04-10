# Security Policy

## Reporting A Vulnerability

Please do **not** open a public GitHub issue for security-sensitive problems.

If you find a vulnerability, use one of these paths:

1. GitHub private vulnerability reporting / security advisory flow, if available for the repository
2. Email the maintainer at `whittakerbent@googlemail.com`

Please include:

- a clear description of the issue
- affected files or endpoints
- steps to reproduce
- impact you believe it may have
- any proof-of-concept details needed to verify it

## What To Expect

I will try to:

- confirm receipt
- reproduce the issue
- assess impact and scope
- fix it privately before broad public disclosure when appropriate

## Scope Notes

This project is a Docker-first local application. The most useful reports are likely to involve:

- API exposure or unsafe defaults
- file handling and upload paths
- container/runtime misconfiguration
- unsafe secrets handling
- dependency vulnerabilities with a realistic impact on this app

## Out Of Scope

These are usually not treated as security issues by themselves:

- bugs that only affect local development convenience
- theoretical issues without a practical attack path
- reports requiring bundled private voice data that the repo does not ship

