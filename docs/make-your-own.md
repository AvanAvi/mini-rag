# Make your own mini-rag

## What the template is

mini-rag is a small question-answering pipeline over a folder of text files. It searches the folder with keyword search (BM25) and with an embedding model, merges the two rankings, and answers only when the best embedding match clears a threshold. Everything you would swap lives in one config file, `mini-rag.toml`: the corpus folder, the embedding model, the model that writes answers, and the threshold. The included Wikipedia setup is the worked example, and the shipped `mini-rag.toml` reproduces it exactly.

## Quick start with the included example

These steps use Windows PowerShell from the project root, as in the README's Setup section.

1. Dot-source the environment script with `. .\scripts\env.ps1` so the model caches stay inside `.cache`.
2. Install the dependencies with `uv sync`.
3. Build the index with `uv run mini-rag ingest`.
4. Ask a question with `uv run mini-rag query "What is a tokamak?"`.

On the included corpus, ingest reports 20 documents and 1,741 chunks. The query's top hit is `[tokamak#0]` with a dense score of 0.880, and the last line is `found: yes, written by a model: no`.

Every command reads `mini-rag.toml` from the working directory. To use another file, pass `--config PATH` before or after the command name, for example `uv run mini-rag --config my.toml ingest`. A value given as a command-line option wins over the config file, and a key left out of the file falls back to the built-in default. Without any config file, mini-rag runs with the built-in defaults, which are the values in the shipped `mini-rag.toml`. A `--config` path that doesn't exist is an error.

## Swapping the corpus

Point `[corpus] dir` at a folder of UTF-8 text files. Each file becomes one document, and its file name without the extension becomes the document id, such as `tokamak` for `tokamak.txt`. Empty files are skipped, and subfolders are ignored.

By default only `.txt` files are read. Reading Markdown is opt-in with `extensions = [".txt", ".md"]`, because a Markdown file in the corpus folder that isn't part of the corpus would otherwise be loaded as a document. The included corpus folder holds such a file, `SOURCES.md`, which lists where each article came from. Markdown files are read as plain text, with the same treatment as `.txt` files. Two files that share a name, such as `notes.txt` and `notes.md`, are rejected because they would share a document id. No other file formats are read.

The licence of your texts is your responsibility. The included articles are CC BY-SA 4.0, as recorded in `data/corpus/SOURCES.md`.

## Swapping the embedding model

The shipped threshold of 0.55 belongs to the default embedding model, `BAAI/bge-small-en-v1.5`. After any model change, run `calibrate` and set `[gate] threshold` before you trust "Not found". Measured once on the 31 included questions: with `minishlab/potion-base-8M` (256 dimensions, 29.5 MB), `calibrate` gave its lowest error count at 0.35, 0.40 and 0.45 and suggested 0.35 without a warning. The shipped 0.55 would have rejected 8 of the 25 answerable questions.

`ingest` overwrites the index in the folder named in `[index] dir`, and `eval` and `calibrate` rebuild the caches in `[eval] cache_dir` when the embedder's name changes. To keep the default index and caches while you try another model, point both keys at new folders, as `examples/custom.toml` does.

### A fastembed model

Set `[embedder] kind = "fastembed"` and `model` to a name from fastembed's list of supported models. The installed fastembed (0.9.1) lists 41 models. mini-rag checks the name against that list before loading anything and rejects an unknown name with a message that shows how to print the list:

```powershell
uv run python -c "from fastembed import TextEmbedding; print([d_model['model'] for d_model in TextEmbedding.list_supported_models()])"
```

The first `ingest` with a new model downloads it into `.cache\fastembed`. If you set `HF_HUB_OFFLINE=1` in your session, clear it for that first run, because the download would fail.

### A class of your own

Set `kind = "custom"` and `class` to `"package.module:ClassName"`. The class must subclass `mini_rag.embedder.Embedder` and provide three things:

- `s_name`, a property that names the model and its settings. It's saved with the index and checked when the index is loaded.
- `embed(l_texts)`, which returns one vector per passage.
- `query_embed(s_query)`, which returns the vector of one question.

Vectors can be lists of floats or numpy arrays. mini-rag normalises them itself. The class is imported from the working directory, so a module in the project folder works without installing it. Keys in an `[embedder.options]` table are passed to the constructor as keyword arguments.

`examples/custom_embedder.py` is a complete example that uses only the standard library. `HashedBagOfWordsEmbedder` lowercases a text, splits it into words, and counts each word in one of `i_dim` buckets chosen by a CRC32 hash of the word. Its `s_name` includes `i_dim`, so an index built with 256 buckets is rejected by a config that asks for 64. `examples/custom.toml` uses it with `i_dim = 256` and keeps its index in `.cache/index_custom` and its eval cache in `.cache/eval_custom`, so the default index is left alone:

```powershell
uv run mini-rag --config examples/custom.toml ingest
uv run mini-rag --config examples/custom.toml query "What is a tokamak?"
```

On the included corpus this ingest took under a second for 1,741 chunks. The example matches only texts that share words, so it's a starting point to copy, not a useful model. For "What is a tokamak?" its best dense score is 0.446, below the shipped threshold of 0.55, so the answer is "Not found in the corpus.". That threshold was calibrated for `BAAI/bge-small-en-v1.5`, not for this embedder; see "How to read the calibration table" below.

## Swapping the generator

Generation only runs with `query --generate`, and only after the threshold check has passed. Without `--generate`, the answer is the retrieved chunk texts under their citation markers. The shipped `mini-rag.toml` sets `kind = "none"` and `model = ""`, so generation is off until you change `[generator]`. `mini-rag.toml` holds a commented block for each other kind. `[generator] kind` picks the client:

- `"anthropic"` uses the Anthropic API with the configured `model`. Set `model` to the model name from your provider; with an empty `model`, every command stops with `error: [generator] model is required for kind "anthropic"; set it to the model name from your provider` and exit code 2. It needs the `llm` extra (`uv sync --extra llm`) and the `ANTHROPIC_API_KEY` variable set in the session. The key check runs first: with the variable unset, `--generate` prints `note: ANTHROPIC_API_KEY is not set; showing the extractive answer` whether or not the extra is installed. With the variable set but the extra missing, it prints a note that the generator failed.
- `"openai_compatible"` posts to `<base_url>/chat/completions` with the model, the messages and `max_tokens` (500). Set `base_url`, such as `"http://localhost:8080/v1"` for a server on your own machine, and `model`; an empty `model` stops every command with the same error, naming kind `"openai_compatible"`. If the server needs a key, put the name of the environment variable that holds it in `api_key_env`. The key is read from that variable for each request and isn't stored. If the variable named in `api_key_env` is unset, the request is sent without an `Authorization` header and no note is printed, which suits a local server that needs no key. This client follows the commonly used chat-completions request format and was not tested against a live server in this project.
- `"none"` turns generation off and is the default. `--generate` then prints `note: no generator is configured ([generator] kind is "none"); showing the extractive answer` and shows the extractive answer.
- `"custom"` needs no `model`. It imports a `Generator` subclass named in `class`, with keyword arguments from `[generator.options]`. The class needs an `s_name` property and `generate(s_system, s_user)`, which returns the reply text.

If the client fails, for example because the server can't be reached, the query prints a note and shows the extractive answer.

## Re-run ingest after a model change

The index stores the vectors of one embedding model, and vectors from two models can't be compared. `ingest` saves the embedder's `s_name` in `.cache/index/chunks.json`, and `query` refuses an index built with another embedder, with a message that ends in "run ingest again". For a fastembed model the check runs before any model is loaded, so a refused query downloads nothing. A custom class is built before the check, so its constructor still runs. Changes under `[chunking]` also need a new `ingest`, because the saved index holds the chunks. The eval caches record the embedder's name in `meta.json`, and `eval` and `calibrate` rebuild a cache by themselves when the name doesn't match.

## Building your own eval in five steps

The eval measures how often retrieval finds the right passage, and the calibration picks a threshold for your embedding model. Both need a question file.

1. Create a starter file with `uv run mini-rag eval-init`. It writes `eval/my_questions.json` with one answerable and one unanswerable example, each explained in a `comment` field that the loader ignores. It won't replace an existing file unless you add `--force`. The two entries are placeholders, such as the `source_doc` value `your_document_name`, and `eval-check` fails on the file until you replace them.
2. Print candidate facts with `uv run mini-rag eval-suggest`. It lists up to 30 sentences (change it with `--limit`), at most 5 from each document, as `doc_id | sentence`. A candidate holds a number or a date and is 60 to 200 characters long. Headings are skipped, and a document is read only up to a heading such as "References" or "See also". The tool suggests facts; you write the questions.
3. Write your questions in the file and point `[eval] questions` at it, or pass `--questions eval/my_questions.json` to each command.
4. Check them with `uv run mini-rag eval-check --lenient`. It prints the question counts and every problem it finds. `--lenient` keeps the checks on each question and skips the rules for the set as a whole, so a set of 5 questions can pass. Without `--lenient` the check is strict and needs 24 to 28 answerable and 5 to 7 unanswerable questions, among other set-level rules.
5. Calibrate the threshold with `uv run mini-rag calibrate`. It warns when the file has fewer than 10 answerable or 3 unanswerable questions, prints the threshold sweep, and prints the line to put under `[gate]` in your config. It never edits the config itself.

A good answerable question:

- names its `source_doc` and a `must_contain` phrase of 6 to 60 characters copied from that document, which a correct answer needs;
- doesn't contain that phrase itself, since that would hand the answer to keyword search;
- uses a phrase found in at most 2 documents, so a passage from an unrelated document rarely counts as a hit;
- is worded in your own words where you can. Questions that copy the source sentence favour keyword search.

A good unanswerable question has a `topic` of `on_topic` (about your subject, but the corpus has no answer) or `off_topic`, and `absent_terms` that a correct answer would contain. eval-check confirms that none of those terms occurs anywhere in the corpus. On-topic questions are the hard case for the threshold and worth including.

When the set is large enough, `uv run mini-rag eval` compares BM25, dense and hybrid search at the chunk sizes in `[eval] sizes` and writes `eval/results.md`.

## Choose a retrieval mode

`[retrieval] mode` sets how `query` and `calibrate` search: `"hybrid"` (the default), `"dense"` or `"bm25"`. The `--mode` option overrides it for one command, for example `uv run mini-rag query "What is a tokamak?" --mode dense`.

- Table 1 of the report that `uv run mini-rag eval` writes compares BM25, dense and hybrid search on your questions. Read it before you change the mode.
- The mode changes `query` and `calibrate` only. `eval` always reports all three methods and takes its threshold sweep from the hybrid hits.
- `ingest` builds the same index in every mode, so a mode change needs no new ingest.
- `rrf_k`, `bm25_weight` and `dense_weight` apply to hybrid mode only.
- In hybrid mode the gate compares the best dense score among the fused hits with the threshold.
- In dense mode the gate compares the best dense score among the dense hits with the threshold.
- In bm25 mode no hit has a dense score, so the threshold isn't applied. The answer counts as found when at least one chunk shares a term with the question, and `query` prints a note saying so. The question is never embedded, and `calibrate` stops with exit code 2.

In bm25 mode the question "Who won the 1998 football World Cup?" returns 5 BM25 hits on the included corpus, so the gate can't reject it. In hybrid mode the same question is reported as not found.

## How to read the calibration table

`calibrate` uses the configured embedder, chunk size, overlap, fusion settings, `top_k` and retrieval mode. It uses only the chunk size in `[chunking]`, not the sizes in `[eval] sizes`. For each question it takes the best dense score among the top hits (the fused hits in hybrid mode, the dense hits in dense mode), then counts two kinds of error at each threshold, in steps of 0.05 across the range of the measured scores. "Answerable rejected" counts questions the corpus answers that the threshold would turn away. "Unanswerable accepted" counts questions it can't answer that would get through. The line under the table names the thresholds with the fewest total errors, and the suggested config line uses the lowest of them.

Look at more than the lowest total. When two thresholds tie, the lower one rejects fewer real answers, and its distance above the lowest answerable score tells you how much room it leaves for new questions. The included set went through this: 0.55 and 0.60 tied at 3 errors, and 0.55 was kept for its larger margin (decision D23 in [decisions.md](decisions.md)).

The table can also show that no threshold works. With the example custom embedder on the included questions, the fewest errors (6) are at 0.20, where every one of the 6 unanswerable questions is accepted, and every higher threshold has more errors in total (7 at 0.25, 24 at 0.55). That embedder can't tell the two groups apart, so the threshold can't either.

A threshold picked from the table is tuned on the same questions it's measured on, so it will look better there than on new questions. With a few unanswerable questions, one question moves the acceptance rate a lot. The threshold belongs to the embedding model it was calibrated with: after you change `[embedder]`, run `ingest` and `calibrate` again before you trust `[gate] threshold`.

## What the template doesn't do

- It reads only `.txt` files and, on request, `.md` files. PDF, HTML and word-processor files aren't supported.
- It has no reranker. The threshold reads an embedding score, which measures how close the topic is, not whether the answer is present, so a question on your subject that the corpus can't answer often gets through.
- `ingest` embeds the whole corpus every time; it doesn't update an index in place.
- Search compares the question with every chunk, which is fine for a small corpus but slow for a very large one.
- `eval-suggest` doesn't write questions, and `calibrate` doesn't change your config.
- The `openai_compatible` client hasn't been tested against a live server.
