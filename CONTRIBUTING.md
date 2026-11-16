# Contributing to mini-rag

## Suggest a feature or report a bug

Open an issue before you write code. Pick the Feature request template for a new feature or a changed behaviour, and the Bug report template for something that doesn't work as the README describes.

A feature request that changes a default value must include the eval tables that justify it: the tables from `eval/results.md` after a `uv run mini-rag eval` run with the proposed value, next to the same tables with the current value.

## Make a change

The commands below were run on Windows with Windows PowerShell 5.1. No test has run on Linux or macOS.

1. Fork the repository on GitHub and clone your fork.
2. Create a branch for the change, for example with `git switch -c fix-query-note`.
3. Open a PowerShell window in the project root and dot-source the environment script with `. .\scripts\env.ps1`.
4. Install the dependencies with `uv sync`.
5. Run the test suite with `uv run pytest`. The expected output ends with `368 passed, 1 deselected`.
6. Make the change together with its tests and the documentation updates listed below.
7. Run `uv run pytest` again before you open the pull request. The run must end with 0 failed tests.

## Rules for changes

- New behaviour comes with tests that cover it. The default test suite must stay offline: no model download and no network call.
- Code is object-oriented, with one class per file.
- Every class and every method has a documentation comment (docstring).
- Python files use 2-space indentation.
- Every variable name, including parameters and attributes, starts with the prefix of its type. `self`, `cls` and ALL_CAPS constants are exempt.

| Prefix | Type |
|---|---|
| `s_` | str |
| `i_` | int |
| `f_` | float |
| `b_` | bool |
| `l_` | list |
| `d_` | dict |
| `t_` | tuple |
| `arr_` | numpy array |
| `obj_` | object instance |

- Default values stay unchanged unless the eval shows a gain of more than 2 questions on hit@5 at both chunk sizes (Table 1 of `eval/results.md`). A gain of 2 questions or fewer is inside the noise ([D22](docs/decisions.md#d22-fusion-tuning)).
- A new dependency needs an entry in [docs/decisions.md](docs/decisions.md) that states why it's needed.
- Every behaviour change updates [docs/decisions.md](docs/decisions.md), [docs/flow.md](docs/flow.md) and the Status diagram in [README.md](README.md#status) in the same pull request.

## Pull request title and description

Pull requests are squash-merged, and the commit message is built from the pull request's title and description.

- The title is one line that says what the change does.
- The description is a bulleted list of what was added, removed or changed, at most 50 words in total.
- The description holds no template text.

## Licence of contributions

A contribution is licensed under the repository's MIT License (see [LICENSE](LICENSE)).

## Open ideas

None of these exists today. Open an issue before you start on one.

- A reranking or answer-declining step for the questions the dense score can't separate. The three on-topic unanswerable questions, q26 to q28, pass the shipped threshold at chunk size 500 ([D23](docs/decisions.md#d23-threshold-decision)). mini-rag has no such step today.
- A POSIX shell equivalent of `scripts\env.ps1`. Only the PowerShell script exists today.
- A continuous-integration workflow. The repository has none today, and the tests run locally.
