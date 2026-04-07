# Contributing

## Getting started

Fork the repository on GitHub, then clone your fork:

```bash
git clone git@github.com:<your-username>/pardax.git
cd pardax
```

Add the upstream remote so you can pull in future changes:

```bash
git remote add upstream git@github.com:christianfenton/pardax.git
```

## Syncing dependencies

[uv](https://docs.astral.sh/uv/) is used for dependency management. Dependencies are split into groups so you only install what you need:

| Group | Purpose | Command |
|-------|---------|---------|
| `test` | pytest, beartype | `uv sync --group test` |
| `lint` | ruff, mypy | `uv sync --group lint` |
| `docs` | mkdocs and plugins | `uv sync --group docs` |
| `notebooks` | matplotlib, ipykernel | `uv sync --extra notebooks` |

To install everything at once:

```bash
uv sync --all-groups --all-extras
```

## Running tests

```bash
uv run pytest
```

Tests use a 1D heat equation with Dirichlet BCs validated against an 
analytical Gaussian solution. Runtime type checking is enabled 
through [jaxtyping](https://github.com/patrick-kidger/jaxtyping) 
and [beartype](https://github.com/beartype/beartype)

## Linting

Run ruff for style and import checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

## Documentation

Serve the docs locally while you work:

```bash
uv run mkdocs serve
```

Build a static site:

```bash
uv run mkdocs build
```
