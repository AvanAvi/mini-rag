# Design decisions

Each entry records one decision. Numbers come from the included corpus and the saved index unless a line says otherwise. Entries marked "planned" describe work that isn't built yet.

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

Chunks start and end mid-word and mid-sentence, which adds a little noise to both retrievers. Size 1000 is to be compared against 500 in the eval (D15). Built and merged.

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

The threshold will be calibrated with the eval's threshold sweep (D15). Because the gate reads the best score among the returned hits, a larger `--top-k` can turn "not found" into "found". The gate is built and merged; the calibration is planned.

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

`uv run pytest` runs 129 tests and deselects 1. `uv run pytest -m slow` runs the deselected one.

### How

`pyproject.toml` adds `-m 'not slow'` to every run. Test doubles stand in for the outside world: `FakeEmbedder` (fixed vectors) and `BagOfWordsEmbedder` (word counts) replace the model, `FakeIndex` returns a fixed ranking, `FakeLlmClient` returns a fixed reply, and `FakeWikipediaClient` returns fixed pages. `WikipediaClient` tests check URL building and response parsing on hand-made JSON, and `LlmClient` tests set the key variable to dummy values. `DenseIndex` imports fastembed only when it needs the real model, so tests with a fake embedder never import it.

### Why

Tests should be fast, deterministic and runnable offline. `DenseIndex`, `Cli`, `Answerer` and `CorpusDownloader` take their model or network collaborator as a constructor argument, so tests pass a fake in directly.

### Trade-off or status

The fakes don't check how well the real model ranks text; the slow test covers one case of that, and the eval (D15) is meant to cover it properly. Built and merged.

## D14 Command-line interface

### Decision

Provide one `mini-rag` command with `ingest`, `query` and `eval` subcommands. `eval` is a placeholder for now.

### What it does

`ingest` loads, chunks, embeds and saves the corpus, then prints the document and chunk counts, the settings, the index folder and the elapsed time. `query` prints each hit with its dense, BM25 and fused scores, then the answer and whether it was found and whether a model wrote it. `eval` prints "eval is not implemented yet" and exits with code 2.

### How

`Cli` builds an `argparse` parser with subcommands; `pyproject.toml` maps the `mini-rag` script to `Cli.main`, which sets UTF-8 output and exits with the code from `run()`. Usage errors exit with code 2, which is argparse's own code, and the CLI uses the same code for a missing corpus, a missing index and invalid values such as `--top-k 0`. Success exits with 0.

### Why

One entry point keeps the README short, and returning exit codes from `run()` lets tests call it directly with a fake embedder. Measured once on a CPU, ingest of the corpus took about 160 seconds, and the saved index is about 3.6 MB.

### Trade-off or status

`ingest` and `query` are built and merged. `eval` isn't built yet.

## D15 Eval design (planned)

### Decision

Measure retrieval with a hand-written question set. This entry is planned; nothing in it is built yet.

### What it does

It will compare BM25, dense and hybrid retrieval at chunk sizes 500 and 1000, and sweep the answer threshold.

### How

Each question has a `must_contain` phrase. A retrieved chunk counts as relevant if it contains that phrase, ignoring case. Phrases are kept under about 60 characters, so the 100-character overlap (D03) guarantees each one appears whole in some chunk. The metrics are hit@k and recall@k for k = 1, 3 and 5, plus MRR. The threshold sweep runs from 0.40 to 0.80 and reports, at each value, how many answerable questions are wrongly rejected and how many unanswerable ones are wrongly accepted. The set will have about 25 answerable and 5 unanswerable questions, with some unanswerable ones on topic for the corpus.

### Why

A phrase check needs no hand-labelled chunk ids, so it still works when the chunk size changes. On-topic unanswerable questions test the gate where it's hardest.

### Trade-off or status

Planned. With about 25 questions, a difference of 1 to 2 questions is noise, and hand-written questions tend to share words with the source text, which favours BM25.

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
