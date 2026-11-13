# Design decisions

Each entry records one decision. Numbers come from the included corpus and the saved index unless a line says otherwise. The last section of each entry states whether it is built. Eval numbers come from the run recorded in `eval/results.md`.

## D01 Light dependency set

### Decision

Use fastembed on ONNX Runtime for embeddings instead of sentence-transformers and torch, and keep the vectors in a numpy array instead of a vector database.

### What it does

The runtime dependencies in `pyproject.toml` are `fastembed`, `numpy` and `rank-bm25`. The dense index is one float32 matrix held in memory and saved as a `.npy` file.

### How

`FastEmbedEmbedder` wraps `fastembed.TextEmbedding`, which runs an ONNX model on the CPU. `DenseIndex` stores the vectors in `arr_vectors` and scores a query with one matrix product (`arr_vectors @ arr_query`) followed by a sort.

### Why

torch isn't installed in the environment at all, which keeps the install small and CPU-only. The whole corpus produces 1,741 vectors of 384 values (`vectors.npy` is 2,674,304 bytes), so an exact search over every row is cheap and a database would add a service without a benefit.

### Trade-off or status

Search is brute force over every row. That's fine at this size but would need an approximate index for a much larger corpus. Built and merged.

## D02 Embedding model

### Decision

Use `BAAI/bge-small-en-v1.5`, with passages embedded by `embed()` and queries by `query_embed()`.

### What it does

Each chunk and each question becomes a 384-dimensional vector. The saved index has shape (1741, 384).

### How

`DenseIndex` passes the model name to `FastEmbedEmbedder`, which creates `TextEmbedding(model_name=...)` on first use. fastembed loads a quantised ONNX build of the model, stored in `.cache/fastembed/models--Qdrant--bge-small-en-v1.5-onnx-Q` (67,181,330 bytes, about 64 MB). `build()` calls `embed()` on chunk texts, and `search()` calls `query_embed()` on the question, which applies the model's query prefix.

### Why

It's a small English model that fastembed supports directly, so ingest runs on a CPU (about 160 seconds for the corpus, see D14). bge models expect a different treatment for queries and passages, which is why the two calls are kept apart.

### Trade-off or status

The model name is saved in `chunks.json` and checked on load (see D07), so an index built with another model is rejected instead of giving silently wrong scores. Built and merged.

## D03 Fixed-size character chunks

### Decision

Cut each document into windows of 500 characters that overlap by 100, with no word or sentence boundary logic.

### What it does

`Chunker` turns 20 documents into 1,741 chunks. Each chunk starts 400 characters after the previous one.

### How

`Chunker.chunk_document` slides a window of `i_size` characters forward by `i_size - i_overlap`. When a window reaches the end of the text it stops, so no final chunk made only of overlap is emitted. The constructor rejects a size below 1, a negative overlap, and an overlap that isn't smaller than the size. Both values are CLI options (`--size`, `--overlap`).

### Why

Character windows are simple to reason about and to test, and their behaviour doesn't depend on a sentence splitter. With a 100-character overlap, any phrase of up to 100 characters appears whole in at least one chunk.

### Trade-off or status

Chunks start and end mid-word and mid-sentence, which adds a little noise to both retrievers. The eval compares size 1000 against 500 (D18); the default stays at 500. Built and merged.

## D04 Chunk identity

### Decision

Identify a chunk by the pair (document id, chunk index). Keep `Document` and `Chunk` as frozen dataclasses, and use the file name stem as the document id.

### What it does

`tokamak.txt` becomes document `tokamak`, and its 95th chunk is `("tokamak", 94)`, shown as the marker `[tokamak#94]`.

### How

`Loader` sets `s_doc_id=obj_path.stem`. `RankFuser.key` returns `(s_doc_id, i_index)`, and `HybridRetriever` uses the same key to look up each retriever's score. `Answerer.marker` formats the citation marker.

### Why

The pair is unique, readable, and stable for a given corpus and chunk size. Frozen dataclasses can't be changed after creation, so no later step can alter a chunk once the chunker has made it.

### Trade-off or status

The index depends on chunk size and overlap, so the same marker points at different text after a re-ingest with other settings. Built and merged.

## D05 One shared tokenizer

### Decision

Use one tokenizer for BM25 indexing and BM25 queries: lowercase, then take runs matching `[a-z0-9]+`, with no stemming and no stop words.

### What it does

"The Tokamak's plasma" becomes `["the", "tokamak", "s", "plasma"]`.

### How

`Tokenizer` compiles the pattern once. `Bm25Index` creates one `Tokenizer` and uses it in both `build()` and `search()`.

### Why

The same rules on both sides mean a word in a question matches the same word in a chunk. The rules are deterministic and easy to test.

### Trade-off or status

"reactor" and "reactors" don't match each other, and characters outside a to z and 0 to 9, such as Greek letters or accented letters, act as separators. Built and merged.

## D06 BM25 keyword baseline

### Decision

Use `BM25Okapi` from `rank-bm25` as the keyword retriever and drop hits that score 0 or below.

### What it does

`Bm25Index.search` returns up to `i_top_k` chunks with a positive BM25 score, best first. Equal scores keep chunk order.

### How

`build()` tokenises every chunk and creates `BM25Okapi` with its default parameters. `search()` scores all chunks with `get_scores` and keeps only scores above 0. `build()` raises `ValueError` for an empty chunk list, and also when every chunk produces zero tokens, because `BM25Okapi` divides by the average document length.

### Why

BM25 is the standard keyword baseline and catches exact terms such as names and numbers that embeddings can blur. Dropping non-positive scores keeps chunks that share no term with the query out of the fusion.

On tiny corpora the IDF behaves oddly. rank-bm25 computes IDF as log((N - n + 0.5) / (n + 0.5)). In a 2-chunk corpus a term found in one chunk gets IDF 0, so the matching chunk scores 0 and is dropped (`test_two_chunk_corpus_gives_zero_idf` documents this). A term found in more than half the chunks gets a negative raw IDF, which rank-bm25 replaces with a small positive floor (0.25 times the average IDF). For that reason the ranking tests use a 6-chunk corpus in which each rare word appears in exactly one chunk.

### Trade-off or status

On the real corpus these effects don't arise for rare terms, but tests on very small corpora have to account for them. Built and merged.

## D07 Dense index details

### Decision

Store L2-normalised float32 rows so cosine similarity is a plain dot product. Do no score filtering inside the index. Save with numpy and JSON, check the files on load, and rebuild BM25 from the saved chunks at query time.

### What it does

`DenseIndex.search` returns the `i_top_k` chunks with the highest cosine score, whatever their value. `save()` writes `vectors.npy` and `chunks.json`, and `load()` restores them.

### How

`normalise()` scales every row to unit length and leaves all-zero rows as zeros. The query vector is normalised the same way. `save()` uses `np.save(..., allow_pickle=False)` and writes the chunks plus `s_model_name` as UTF-8 JSON. `load()` raises `FileNotFoundError` if either file is missing, and `ValueError` if the saved model name differs, the vectors aren't 2-D, or the row count doesn't match the chunk count. It reads with `np.load(..., allow_pickle=False)`. `Cli.query` loads the dense index and builds a new `Bm25Index` from `obj_dense.l_chunks`.

### Why

A dot product on unit vectors is the cosine, so one matrix product scores every chunk. Filtering is left to the answerer (D09) so the index stays a plain ranking. `allow_pickle=False` means loading a file can't run code. Rebuilding BM25 means there's only one saved copy of the chunks to keep consistent.

### Trade-off or status

BM25 is rebuilt on every query; that cost hasn't been measured separately. The saved index takes 3,762,469 bytes (about 3.6 MB). Built and merged.

## D08 Hybrid retrieval by reciprocal rank fusion

### Decision

Merge the BM25 and dense rankings with reciprocal rank fusion (RRF), k = 60, taking 20 candidates from each retriever, or top-k if that's larger.

### What it does

`HybridRetriever.search` returns `HybridHit` objects ordered by fused score. Each hit also carries its dense and BM25 scores, or `None` for a retriever that didn't return that chunk among its candidates.

### How

`RankFuser.fuse` adds 1 / (k + rank) for every ranking a chunk appears in, with 1-based ranks. If a chunk appears twice in one ranking, it counts once at its best rank. The sort is stable and the BM25 list is passed first, so equal fused scores keep first-appearance order with BM25 ahead. `HybridRetriever.score_table` records each retriever's own score per chunk key.

### Why

RRF uses only ranks, so it can combine BM25 scores (unbounded) and cosine scores (between -1 and 1) without tuning weights. A chunk ranked first by both retrievers gets 2 / 61, about 0.0328, so fused scores are small and can't be compared with BM25 or cosine scores. That's why `HybridHit` keeps the two original scores, and why the answerer gates on the dense one.

### Trade-off or status

A chunk outside a retriever's 20 candidates has `None` for that score, which means "not retrieved", not "score 0". Built and merged.

## D09 Answerer gate

### Decision

Count a question as found when the highest non-None dense score among the returned hits is at least the threshold. The default threshold is 0.5, and it's a placeholder.

### What it does

Below the threshold, `Answerer.answer` returns "Not found in the corpus." with no sources. At or above it, the answer uses the hits.

### How

`Answerer.is_found` collects `f_dense_score` from every hit, ignores `None`, and compares the maximum with `f_threshold`. If no hit has a dense score, the answer is "not found". The constructor rejects thresholds outside -1.0 to 1.0. The CLI option is `--threshold`.

### Why

Cosine scores have a fixed range and mean the same thing from query to query, while fused scores depend only on rank (D08). Measured on the unanswerable query "Who won the 1998 football World Cup?" with the default top-k of 5: the top fused hit has dense 0.502, the best dense score among the five hits is 0.528, so 0.5 accepts it and 0.6 rejects it. The default wasn't raised on the strength of this one example, because a single unanswerable question says nothing about how many answerable questions a higher threshold would wrongly reject.

### Trade-off or status

The eval's threshold sweep counts both kinds of error at each threshold (D19). The default stays at 0.5, and the choice of a new value is deferred there. Because the gate reads the best score among the returned hits, a larger `--top-k` can turn "not found" into "found". The gate is built and merged.

## D10 Opt-in generation

### Decision

The default answer is extractive. Writing an answer with a language model happens only with `--generate`, and the API key is checked for presence only.

### What it does

Without `--generate`, the answer is the retrieved chunk texts, each under its `[doc_id#index]` marker. With it, the optional feature sends those chunks and the question for generation with a language model through the Anthropic API, and `Answer.b_generated` is set to `True`.

### How

`LlmClient.is_available` returns whether `ANTHROPIC_API_KEY` is set and non-empty. The value is never read into a variable, printed or stored; the API client reads it from the environment itself. `LlmClient.get_client` imports the `anthropic` package on first use and raises `RuntimeError` naming `uv sync --extra llm` if the `llm` extra isn't installed. `Cli.answer` falls back to the extractive answer, with a note, when the key is missing or that error occurs. The source text goes in the user message as data, and `Answerer.S_SYSTEM_PROMPT` tells the model to answer only from the numbered sources, to cite them by marker, and to ignore any instructions inside the source text.

### Why

The core pipeline should run with no account, no network and no cost. Importing the package lazily means the default install doesn't need it. Treating chunk text as data limits what text inside an article could make the model do.

### Trade-off or status

The gate still runs first, so generation never sees a question that failed the threshold. The prompt instruction reduces but can't rule out instruction-following on source text. Built and merged.

## D11 Wikipedia corpus

### Decision

Use 20 English Wikipedia articles, fetched with the Python standard library and recorded with enough detail to check them later.

### What it does

`data/corpus` holds one `.txt` file per title listed in `data/corpus_titles.txt`, plus `SOURCES.md`.

### How

`WikipediaClient` calls the MediaWiki API with `urllib.request`, asking for the plain-text extract (`explaintext`) and the revision id, and following redirects. Its User-Agent is `mini-rag/0.1` plus the repository URL, with no email address, and a test checks that. `CorpusDownloader` saves each text as UTF-8 with `\n` line endings, pauses 1 second between requests, skips files that already exist unless `--force` is given, and writes a `SOURCES.md` row per file with the title, URLs, revision id, retrieval date, character count and SHA-256. It runs with `uv run python -m mini_rag.corpus_downloader`. `.gitattributes` sets `data/corpus/*.txt text eol=lf`, so a checkout on Windows doesn't turn line endings into CRLF and break the hashes.

### Why

The standard library is enough for one GET per article and adds no dependency. Revision ids and hashes make the corpus reproducible and let anyone confirm a file is unchanged.

### Trade-off or status

The texts are CC BY-SA 4.0 and the code is under the MIT License; `SOURCES.md` states the text license. Plain-text extracts drop tables and flatten formulas (see the README's Known limits). Built and merged.

## D12 Caches stay in the project

### Decision

Keep everything the tools cache inside the project folder.

### What it does

The embedding model lives in `.cache/fastembed`, Hugging Face hub files in `.cache/huggingface`, the index in `.cache/index`, and pytest temporary folders in `.cache/pytest-tmp`. `.gitignore` excludes `.cache/`.

### How

`scripts/env.ps1` sets `FASTEMBED_CACHE_PATH` and `HF_HOME` for the current PowerShell session only, creating the folders if needed. It changes nothing in the registry, PATH or profile. `pyproject.toml` passes `--basetemp=.cache/pytest-tmp` to pytest.

### Why

Setting `FASTEMBED_CACHE_PATH` alone isn't enough: `huggingface_hub`, which fastembed uses, writes to a `.cache\huggingface` folder in the user's home unless `HF_HOME` is set. Keeping caches local makes the project easy to delete without leftovers.

### Trade-off or status

The script must be dot-sourced in each new session. Setting `HF_HUB_OFFLINE=1` as well makes any accidental download fail instead of happening quietly; with the model already cached, `query` works with that flag set. Built and merged.

## D13 Test strategy

### Decision

The default test suite makes no model download and no network call. One test that uses the real model is marked `slow` and skipped by default.

### What it does

`uv run pytest` runs 202 tests and deselects 1. `uv run pytest -m slow` runs the deselected one.

### How

`pyproject.toml` adds `-m 'not slow'` to every run. Test doubles stand in for the outside world: `FakeEmbedder` (fixed vectors) and `BagOfWordsEmbedder` (word counts) replace the model, `FakeIndex` returns a fixed ranking, `FakeLlmClient` returns a fixed reply, and `FakeWikipediaClient` returns fixed pages. `WikipediaClient` tests check URL building and response parsing on hand-made JSON, and `LlmClient` tests set the key variable to dummy values. `DenseIndex` imports fastembed only when it needs the real model, so tests with a fake embedder never import it.

### Why

Tests should be fast, deterministic and runnable offline. `DenseIndex`, `Cli`, `Answerer` and `CorpusDownloader` take their model or network collaborator as a constructor argument, so tests pass a fake in directly.

### Trade-off or status

The fakes don't check how well the real model ranks text; the slow test covers one case of that, and the eval (D18) measures it on 25 questions with the real model. Built and merged.

## D14 Command-line interface

### Decision

Provide one `mini-rag` command with `ingest`, `query` and `eval` subcommands.

### What it does

`ingest` loads, chunks, embeds and saves the corpus, then prints the document and chunk counts, the settings, the index folder and the elapsed time. `query` prints each hit with its dense, BM25 and fused scores, then the answer and whether it was found and whether a model wrote it. `eval` runs the evaluation and writes its report (D21).

### How

`Cli` builds an `argparse` parser with subcommands; `pyproject.toml` maps the `mini-rag` script to `Cli.main`, which sets UTF-8 output and exits with the code from `run()`. Usage errors exit with code 2, which is argparse's own code, and the CLI uses the same code for a missing corpus, a missing index and invalid values such as `--top-k 0`. Success exits with 0.

### Why

One entry point keeps the README short, and returning exit codes from `run()` lets tests call it directly with a fake embedder. Measured once on a CPU, ingest of the corpus took about 160 seconds, and the saved index is about 3.6 MB.

### Trade-off or status

`ingest` and `query` are built and merged. `eval` is built; its options are in D21.

## D15 Eval design

### Decision

Measure retrieval and the answer gate with a hand-written question set. The eval runs offline and uses no language model.

### What it does

`mini-rag eval` compares BM25, dense and hybrid retrieval at chunk sizes 500 and 1000 with overlap 100. It also sweeps the gate threshold from 0.40 to 0.80 and writes every table to `eval/results.md`.

### How

Each answerable question has a `must_contain` phrase, and a chunk counts as relevant if it contains that phrase. `QuestionLoader` and `QuestionValidator` read and check the questions (D17). `Evaluator` ranks them with each method and `ReportTables` computes the metrics (D18). `ThresholdSweeper` counts the gate errors (D19). `EvalIndexStore` caches the embedded chunks per chunk size (D20), and `Cli.eval` ties the steps together (D21).

### Why

A phrase check needs no hand-labelled chunk ids, so the same questions work at any chunk size. Phrases are at most 60 characters, so the 100-character overlap (D03) puts each one whole in at least one chunk. On-topic unanswerable questions test the gate where it's hardest.

### Trade-off or status

Built. The numbers come from one run on 31 questions. A difference of 1 to 2 questions is noise, and hand-written questions tend to share words with the source text, which favours BM25.

## D16 Workflow

### Decision

Build one component per pull request, together with its tests.

### What it does

Each merged pull request adds one class or command plus the tests for it.

### How

Pull requests are squash-merged, and commit messages are short bullet points. When a pull request changes behaviour, it updates this file, `docs/flow.md` and the README status diagram in the same pull request.

### Why

Small pull requests are easy to review, and squash merging leaves one commit per component on `main`.

### Trade-off or status

In use.

## D17 Question set

### Decision

Keep 25 answerable and 6 unanswerable hand-written questions in `eval/questions.json`, and check them against the corpus before every eval run.

### What it does

An answerable question names its `source_doc` and a `must_contain` phrase copied from that document. An unanswerable question has a `topic` (`on_topic` or `off_topic`) and `absent_terms` that a correct answer would contain. The 25 answerable questions cover all 20 documents, and the unanswerable ones are 3 on topic and 3 off topic.

### How

`QuestionLoader.load` raises `FileNotFoundError` for a missing file and `ValueError`, with the question id, for a malformed entry or a duplicate id. `QuestionValidator.validate` returns a `ValidationResult` with a list of problems. It compares text after `RelevanceJudge.normalise`, which lowercases and collapses whitespace. The checks and their reasons:

- The `source_doc` must be a loaded document and must contain the phrase. Without that, no chunk could be relevant.
- The phrase must be 6 to 60 characters on one line. Shorter phrases match by accident, and longer ones might not fit whole in a chunk (D15).
- The phrase must not occur inside the question text, because the question would then hand the answer to keyword search.
- The phrase may occur in at most 2 documents, so a chunk from an unrelated article rarely counts as relevant. In the current set every phrase occurs in exactly one document.
- No absent term of an unanswerable question may occur anywhere in the corpus. This confirms the corpus can't answer it.
- The set needs 24 to 28 answerable and 5 to 7 unanswerable questions, with at least 2 of each topic.
- The answerable questions must cover at least 14 documents with at most 3 questions each, so no single article dominates.
- At least 8 answerable questions need an overlap of 0.40 or less, and at least 4 phrases must contain a digit. The first keeps paraphrased questions in the set, and the second keeps numbers and dates in it.

The overlap of a question is the share of its tokens longer than 3 characters that also occur in the source sentence around the phrase. A sentence ends at a full stop followed by whitespace, or at a line break.

### Why

The checks catch a broken question before it distorts a metric. The overlap figure makes the "shares words with the text" bias visible: the report splits its results at 0.40 (D18).

### Trade-off or status

Built. In the current set 14 of the 25 answerable questions have an overlap of 0.40 or less, and 13 phrases contain a digit. The questions were written after reading the source sentences, so they aren't a sample of what real users would ask.

## D18 Evaluator and metrics

### Decision

Rank every answerable question to depth 10 with BM25, dense and hybrid retrieval, and report hit@k, recall@k and MRR per method and chunk size.

### What it does

`Evaluator.run` returns an `EvalResult` with one `SizeResult` per chunk size. `ReportWriter` turns it into five tables and a per-question appendix.

### How

`Bm25Method`, `DenseMethod` and `HybridMethod` implement `RetrievalMethod.rank`. The hybrid method uses `HybridRetriever` with its defaults (D08). `RelevanceJudge.is_relevant` calls a chunk relevant when the normalised phrase occurs in the normalised chunk text. `MetricCalculator` defines the metrics:

- hit@k is true when the first relevant chunk is at rank k or better.
- recall@k is the number of relevant chunks in the top k divided by the number of relevant chunks in the whole chunk set. A phrase can sit in two overlapping chunks, and recall@1 is then at most 0.5 for that question.
- The reciprocal rank is 1 divided by the rank of the first relevant chunk, or 0 when none is in the top 10. MRR is its mean over the questions.

Table 2 compares two rankings question by question on first rank and counts wins, losses and ties, with "not retrieved" as the worst rank. Table 5 splits hit@5 and MRR by overlap group (D17).

### Why

Depth 10 is enough for hit@5 and recall@5 and lets MRR credit ranks 6 to 10. The pairwise counts show whether a gap in the means comes from many questions or from a few. No confidence interval is reported: the report gives raw counts out of 25, and its reading note says that 1 to 2 questions of difference is noise.

### Trade-off or status

Built. Measured results (hit@5, MRR):

| Chunk size | BM25 | dense | hybrid |
|---|---|---|---|
| 500 | 16/25, 0.528 | 21/25, 0.688 | 19/25, 0.587 |
| 1000 | 19/25, 0.656 | 21/25, 0.721 | 22/25, 0.735 |

Hybrid at size 1000 has the better first rank on 9 questions, hybrid at size 500 on 4, and 12 are ties. At size 500, BM25 finds 6 of the 14 low-overlap questions in its top 5 and 10 of the 11 high-overlap ones. The default chunk size stays at 500; the eval changes no default.

## D19 Gate sweep

### Decision

Sweep the gate threshold over the eval questions and report both kinds of error. Leave the default threshold at 0.5.

### What it does

For each threshold from 0.40 to 0.80 in steps of 0.05, Table 3 shows how many answerable questions the gate would reject and how many unanswerable ones it would accept.

### How

For every question, `Evaluator.best_dense` takes the top 5 hybrid hits and keeps the highest non-None dense score. `ThresholdSweeper.sweep` counts a question as found when that score is at least the threshold, which is the rule in `Answerer.is_found` (D09). A test checks that the two agree on the same hits.

### Why

One threshold trades wrong rejections against wrong acceptances, and the single World Cup example in D09 showed only one side.

### Trade-off or status

Built. The measured rows are (answerable rejected, unanswerable accepted):

| Threshold | Size 500 | Size 1000 |
|---|---|---|
| 0.40 | 0/25, 6/6 | 0/25, 6/6 |
| 0.50 | 0/25, 5/6 | 0/25, 6/6 |
| 0.60 | 0/25, 3/6 | 0/25, 3/6 |
| 0.70 | 3/25, 2/6 | 4/25, 2/6 |
| 0.80 | 16/25, 0/6 | 18/25, 0/6 |

At both sizes the total error count is lowest at 0.55 and 0.60, with 3 errors. The sweep shows that the two groups overlap: at size 500 the best dense score runs from 0.648 to 0.890 for answerable questions and from 0.489 to 0.733 for unanswerable ones, so no threshold separates them on this set.

The sweep can't show how a threshold performs on new questions, because a value picked from Table 3 is tuned on the questions it is measured on. It covers top-k 5 only, and the gate's result depends on top-k (D09). With 6 unanswerable questions, one question is a sixth of that group. For these reasons the default stays at 0.5 and the choice of a new value is deferred.

## D20 Eval index cache

### Decision

Cache the embedded chunks of each eval chunk setting in its own folder under `.cache/eval`, apart from the query index in `.cache/index`.

### What it does

The first eval run for a setting embeds the corpus and saves it. Later runs load the saved vectors, so only the first run pays for embedding.

### How

`EvalIndexStore.get_indexes` uses the folder `size_<size>_overlap_<overlap>`. It holds the two files written by `DenseIndex.save` plus `meta.json` with the size, overlap, model name, chunk count and corpus fingerprint. The fingerprint is a SHA-256 over the sorted lines `doc_id:sha256(text)`. The store loads the cache only when the size, overlap, model name and fingerprint all match and the files load cleanly; otherwise it rebuilds and overwrites the folder. `--rebuild` forces that. BM25 is rebuilt from the chunks on every call.

### Why

The eval needs two chunk sizes, and `.cache/index` holds one index that `query` depends on. A separate cache means an eval run can't replace the index a user queries. The fingerprint stops a cache from being loaded after the corpus changes; a test covers that case.

### Trade-off or status

Built. In one CPU run, building took 160 seconds for size 500 (1,741 chunks) and 154 seconds for size 1000 (780 chunks). The size 500 vectors are computed again even when `.cache/index` already holds the same chunks.

## D21 CLI changes

### Decision

Replace the `eval` placeholder with the real command, add `--brief` to `query`, and report a bad saved index as an error line.

### What it does

`eval` validates the questions, runs the evaluator, prints Table 1 and Table 3 and writes the report. `query --brief` cuts each hit's text to 160 characters and appends `...` only when it cut something. `query` on an index saved under another model name prints `error: <message>` and exits with code 2.

### How

The `eval` options are `--questions` (default `eval/questions.json`), `--corpus` (`data/corpus`), `--sizes` (`500,1000`), `--overlap` (100), `--gate-top-k` (5), `--cache-dir` (`.cache/eval`), `--out` (`eval/results.md`), `--rebuild` and `--build-only`. With `--build-only`, `Cli.build_only` calls `Evaluator.prepare` for each size and prints the chunk count, whether the index was built or loaded, and the elapsed seconds; it loads no questions and writes no report. If the validator finds problems, `Cli.fail_problems` prints up to 10 of them and the command exits with code 2. `FileNotFoundError` and `ValueError` become an error line with exit code 2. `Cli.query` now also catches the `ValueError` that `DenseIndex.load` raises for a model name mismatch or damaged files (D07). `Cli` passes its embedder to `EvalIndexStore`, so tests run the whole command with a fake embedder.

### Why

Validating first means a broken question set stops the run before any ranking. `--build-only` separates the slow embedding step from the evaluation. `--brief` keeps a query's output short enough to scan.

### Trade-off or status

Built. `--brief` shortens only the hit list; the extractive answer below it still prints the full chunk texts.
