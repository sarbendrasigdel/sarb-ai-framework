# Distribution

How `sarb` gets from this repository into a product that uses it.

## The mental model

`sarb` is a **package**, not a template. You do not clone it to start a project, and you
do not copy files out of it. You add it as a dependency, exactly as you would add
`fastapi` or `pydantic`, and it lands in your project's `.venv` alongside them.

```
sarb            (this repo)      ← cloned only by people working ON the framework
  │
  │ published as a versioned package
  ▼
support-bot     contract-search     internal-assistant     code-helper
   │                  │                     │                   │
   └──────────────────┴──── all depend on sarb ────────────────┘
```

That arrangement is the entire point. When a bug is fixed in the framework, four
products bump a version number. When the bug is fixed in four *copies* of the
framework, it is fixed in one of them and forgotten in three.

Cloning the repo into each project would give you the copies. So: don't.

## Using it in a project

A new product starts empty and adds `sarb` to it:

```bash
mkdir contract-search && cd contract-search
uv init
uv add "sarb[anthropic,qdrant] @ git+ssh://git@github.com/sarbendra/sarb.git@v0.1.0"
mkdir config
```

`uv` clones the tag, builds the wheel, and installs it. The result in your
`pyproject.toml`:

```toml title="contract-search/pyproject.toml"
[project]
name = "contract-search"
dependencies = [
    "sarb[anthropic,qdrant] @ git+ssh://git@github.com/sarbendra/sarb.git@v0.1.0",
]
```

Or, more legibly, using `uv`'s own source table — same thing, and it keeps the version
in one place:

```toml
[project]
dependencies = ["sarb[anthropic,qdrant]"]

[tool.uv.sources]
sarb = { git = "ssh://git@github.com/sarbendra/sarb.git", tag = "v0.1.0" }
```

The extras go **before** the `@`, and they work exactly as they do on a published
package: `sarb[anthropic,qdrant]` pulls in the Anthropic SDK and the Qdrant client and
nothing else. A product that calls a hosted API never installs PyTorch.

Your project then looks like this, and contains no framework code at all:

```
contract-search/
├── pyproject.toml       # depends on sarb
├── uv.lock              # pins sarb to an exact commit — reproducible builds
├── config/
│   ├── base.yaml        # llms:, embedders:, vector_stores:
│   └── production.yaml
└── src/contract_search/
    └── app.py           # from sarb.di import bootstrap
```

`uv.lock` records the resolved commit hash, not just the tag, so a build is reproducible
even if someone moves a tag.

## Cutting a release

A release is a version and a tag. Nothing else.

```bash
# 1. Bump the version. This is the only place it lives.
#    pyproject.toml:  version = "0.2.0"

# 2. Make sure it is actually releasable.
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run --group docs mkdocs build --strict

# 3. Tag it and push.
git commit -am "Release 0.2.0"
git tag v0.2.0
git push && git push --tags
```

Consumers then move at their own pace by changing `@v0.1.0` to `@v0.2.0` and running
`uv sync`. Nobody is forced to upgrade by a push.

You can confirm what will be built at any point:

```bash
uv build          # → dist/sarb-0.2.0-py3-none-any.whl
```

That wheel is what a consuming project installs. It contains `src/sarb/` and nothing
else — no tests, no docs, no config.

## Versioning

Semantic versioning, where "breaking" is judged **from the perspective of a product
built on the framework**:

| Change | Bump |
| --- | --- |
| A port gains a method, or a method gains an argument with a default | minor |
| A new provider, a new extra, a new config family | minor |
| An adapter option is renamed or removed | **major** — it breaks somebody's `base.yaml` |
| A port's signature changes | **major** — it breaks somebody's engine |
| A config key is renamed | **major** — a config file *is* an API |
| A bug fix, a doc, a faster adapter | patch |

That third and fifth row are the ones people get wrong. A YAML key is part of the public
interface just as much as a Python signature is — arguably more so, because it is edited
by people who never read the code.

Below `1.0.0`, the framework is allowed to break on a minor bump. Say so out loud in the
release notes when it does.

## Working on the framework and a product at the same time

This is the case the git-tag approach handles least gracefully, and it comes up
constantly: you are building `contract-search`, you find a bug in `sarb`, and you want
to fix it and see the fix immediately rather than tag a release for every keystroke.

Point the dependency at your local checkout:

```toml title="contract-search/pyproject.toml"
[tool.uv.sources]
sarb = { path = "../sarb", editable = true }
```

```bash
uv sync
```

Now `../sarb/src/sarb/` **is** the `sarb` your product imports. Edit the framework, rerun
your product, see the change. No reinstall, no tag, no version bump.

!!! warning "Take it back out before you commit"
    An editable path dependency is a pointer to a directory on *your* laptop. Merged into
    `main`, it breaks the build for everyone else and for CI. Fix the bug, restore the
    `git` source, release a version, and depend on that.

## Moving to a package index later

The git-tag approach costs nothing and needs no infrastructure, which is why it is the
right first step. It has two real limits, and you will feel both eventually:

- **Every consumer needs git access to this repository.** Fine for four internal teams;
  not fine for a contractor who should be able to install the framework without being
  able to read its source.
- **Resolution is slower.** `uv` clones and builds rather than downloading a prebuilt
  wheel. Noticeable in a cold CI job.

When either starts to hurt, the migration is small, because it is the same wheel. Stand
up a private index (GitHub Packages, AWS CodeArtifact, Artifactory, Gemfury), add a
release job that runs `uv build && uv publish`, and consumers change:

```diff
- sarb = { git = "ssh://git@github.com/sarbendra/sarb.git", tag = "v0.2.0" }
+ # nothing. Just: dependencies = ["sarb[anthropic,qdrant]>=0.2"]
```

with an index URL configured once, globally. Nothing about the framework itself changes —
`pyproject.toml` already declares everything a real package needs, and `uv build` already
produces the artifact. Only the destination is new.

## Before any of this

The repository has **no remote and no tags yet**. Everything above assumes it lives
somewhere the other projects can reach:

```bash
git remote add origin ssh://git@github.com/sarbendra/sarb.git
git push -u origin main
git tag v0.1.0 && git push --tags
```

Until that happens, the only way to depend on `sarb` is the local path source described
above — which is fine for one product on one laptop, and is not a platform.
