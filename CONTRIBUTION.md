# Contribution Guide

This document defines the Git commit message convention for this repository.

## Commit Format

Use the Conventional Commits style:

```text
<type>(<scope>): <subject>
```

Examples:

```text
feat(cli): add binder format option
fix(contracts): allow declared VHH inputs without light chain
docs(readme): clarify local ESMFold2 runtime usage
```

Notes:

- `scope` is optional, but recommended when the change is limited to one module.
- Use lowercase for `type` and `scope`.
- Keep `subject` short, specific, and action-oriented.

## Allowed Types

- `feat`: a new feature or user-visible capability
- `fix`: a bug fix or behavior correction
- `refactor`: code restructuring without intended behavior change
- `docs`: documentation-only changes
- `test`: new or updated tests
- `chore`: maintenance work such as config or tooling updates

## Scope Guidance

Prefer a scope that matches the main area changed. Typical scopes in this repository include:

- `cli`
- `contracts`
- `readme`
- `tests`
- `pyproject`

If a change spans multiple areas and no short scope fits well, omit the scope:

```text
chore: align development tooling defaults
```

## Subject Rules

Write the `subject` line with these rules:

- describe one logical change per commit
- use a short verb phrase, such as `add`, `fix`, `clarify`, `rename`, `remove`
- do not end the subject with a period
- keep the subject concise; prefer staying within 72 characters
- avoid vague messages such as `update code` or `fix bug`

Recommended:

```text
refactor(contracts): simplify antibody input normalization
test(contracts): cover scFv and VHH validation paths
```

Not recommended:

```text
update code
fix: bug.
Feat(cli): add option
```

## Optional Body

Add a commit body when the title alone is not enough.

Use the body to explain:

- why the change was needed
- important implementation context
- side effects, limitations, or follow-up work

Example:

```text
fix(contracts): allow declared VHH inputs without light chain

Treat VHH as a first-class binder format during schema validation.
Keep numbering failure degradable and preserve generic fallback scoring.
```

## Breaking Changes

If a commit introduces an incompatible change, add a footer:

```text
feat(contracts): rename binder metadata fields

BREAKING CHANGE: `binder_units` now replaces the previous `binder_chains` field.
```

## Quick Checklist

Before committing, confirm that:

- the message matches `<type>(<scope>): <subject>`
- the commit contains one logical change
- the subject is specific and easy to scan in `git log`
- a body or `BREAKING CHANGE` footer is included when needed
