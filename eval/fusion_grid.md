# Fusion grid

- Embedding model: BAAI/bge-small-en-v1.5
- Overlap: 100
- Depth: 10
- Gate top-k: 5
- Setting: RRF constant k and the weights w as BM25/dense.
- vs dense: questions where hybrid ranks its first relevant chunk better / worse / the same as dense alone.
- Unanswerable accepted at 0.60: unanswerable questions the gate lets through at that threshold, judged on the top 5 hybrid hits of the setting.

## Chunk size 500 (1741 chunks)

| Setting | hit@1 | hit@3 | hit@5 | MRR | vs dense | Unanswerable accepted at 0.60 |
|---|---|---|---|---|---|---|
| k=5 w=1.0/1.0 | 12/25 | 19/25 | 20/25 | 0.600 | 3 / 7 / 15 | 3/6 |
| k=5 w=0.5/1.0 | 12/25 | 19/25 | 20/25 | 0.628 | 2 / 6 / 17 | 3/6 |
| k=5 w=0.25/1.0 | 14/25 | 19/25 | 21/25 | 0.669 | 1 / 2 / 22 | 3/6 |
| k=10 w=1.0/1.0 | 12/25 | 18/25 | 20/25 | 0.595 | 3 / 7 / 15 | 3/6 |
| k=10 w=0.5/1.0 | 12/25 | 19/25 | 20/25 | 0.621 | 2 / 6 / 17 | 3/6 |
| k=10 w=0.25/1.0 | 13/25 | 19/25 | 21/25 | 0.655 | 2 / 3 / 20 | 3/6 |
| k=20 w=1.0/1.0 | 12/25 | 18/25 | 19/25 | 0.587 | 2 / 7 / 16 | 3/6 |
| k=20 w=0.5/1.0 | 11/25 | 18/25 | 19/25 | 0.593 | 2 / 7 / 16 | 3/6 |
| k=20 w=0.25/1.0 | 13/25 | 19/25 | 20/25 | 0.648 | 2 / 4 / 19 | 3/6 |
| k=60 w=1.0/1.0 (default) | 12/25 | 18/25 | 19/25 | 0.587 | 2 / 7 / 16 | 3/6 |
| k=60 w=0.5/1.0 | 11/25 | 18/25 | 18/25 | 0.589 | 2 / 7 / 16 | 3/6 |
| k=60 w=0.25/1.0 | 12/25 | 18/25 | 18/25 | 0.616 | 2 / 5 / 18 | 3/6 |

## Chunk size 1000 (780 chunks)

| Setting | hit@1 | hit@3 | hit@5 | MRR | vs dense | Unanswerable accepted at 0.60 |
|---|---|---|---|---|---|---|
| k=5 w=1.0/1.0 | 16/25 | 19/25 | 23/25 | 0.735 | 6 / 4 / 15 | 3/6 |
| k=5 w=0.5/1.0 | 15/25 | 21/25 | 23/25 | 0.729 | 4 / 1 / 20 | 3/6 |
| k=5 w=0.25/1.0 | 16/25 | 20/25 | 22/25 | 0.738 | 5 / 0 / 20 | 3/6 |
| k=10 w=1.0/1.0 | 16/25 | 19/25 | 22/25 | 0.731 | 6 / 4 / 15 | 3/6 |
| k=10 w=0.5/1.0 | 15/25 | 21/25 | 23/25 | 0.730 | 5 / 1 / 19 | 3/6 |
| k=10 w=0.25/1.0 | 15/25 | 21/25 | 23/25 | 0.730 | 5 / 1 / 19 | 3/6 |
| k=20 w=1.0/1.0 | 16/25 | 20/25 | 22/25 | 0.734 | 6 / 3 / 16 | 3/6 |
| k=20 w=0.5/1.0 | 15/25 | 20/25 | 22/25 | 0.722 | 5 / 1 / 19 | 3/6 |
| k=20 w=0.25/1.0 | 15/25 | 21/25 | 23/25 | 0.723 | 5 / 1 / 19 | 3/6 |
| k=60 w=1.0/1.0 (default) | 16/25 | 20/25 | 22/25 | 0.735 | 6 / 3 / 16 | 3/6 |
| k=60 w=0.5/1.0 | 15/25 | 20/25 | 22/25 | 0.721 | 5 / 1 / 19 | 3/6 |
| k=60 w=0.25/1.0 | 15/25 | 20/25 | 22/25 | 0.714 | 5 / 1 / 19 | 3/6 |

## How to read this

The grid is tuned on the same 25 questions it is measured on, so a gain of 2 questions or fewer on hit@5 is noise.
