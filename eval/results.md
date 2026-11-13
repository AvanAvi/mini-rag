# Evaluation results

- Date (UTC): 2026-11-13
- Corpus fingerprint: b827fbef198e
- Embedding model: BAAI/bge-small-en-v1.5
- Chunk sizes: 500, 1000
- Overlap: 100
- Depth: 10
- Gate top-k: 5
- Questions: 25 answerable, 6 unanswerable

## Table 1: Retrieval metrics

### Chunk size 500 (1741 chunks)

| Method | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| BM25 | 12/25 | 13/25 | 16/25 | 0.46 | 0.52 | 0.62 | 0.528 |
| dense | 15/25 | 20/25 | 21/25 | 0.56 | 0.80 | 0.82 | 0.688 |
| hybrid | 12/25 | 18/25 | 19/25 | 0.46 | 0.70 | 0.74 | 0.587 |

### Chunk size 1000 (780 chunks)

| Method | hit@1 | hit@3 | hit@5 | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| BM25 | 14/25 | 17/25 | 19/25 | 0.54 | 0.68 | 0.74 | 0.656 |
| dense | 16/25 | 19/25 | 21/25 | 0.62 | 0.76 | 0.84 | 0.721 |
| hybrid | 16/25 | 20/25 | 22/25 | 0.64 | 0.76 | 0.86 | 0.735 |

## Table 2: Pairwise comparison by first rank

### Chunk size 500 (1741 chunks)

| Comparison | First better | Second better | Ties |
|---|---|---|---|
| hybrid vs BM25 | 8 | 1 | 16 |
| hybrid vs dense | 2 | 7 | 16 |

### Chunk size 1000 (780 chunks)

| Comparison | First better | Second better | Ties |
|---|---|---|---|
| hybrid vs BM25 | 9 | 4 | 12 |
| hybrid vs dense | 6 | 3 | 16 |

### Across chunk sizes

| Comparison | First better | Second better | Ties |
|---|---|---|---|
| hybrid at 500 vs hybrid at 1000 | 4 | 9 | 12 |

## Table 3: Threshold sweep

### Chunk size 500 (1741 chunks)

| Threshold | Answerable rejected | Unanswerable accepted | Total errors |
|---|---|---|---|
| 0.40 | 0/25 | 6/6 | 6 |
| 0.45 | 0/25 | 6/6 | 6 |
| 0.50 | 0/25 | 5/6 | 5 |
| 0.55 | 0/25 | 3/6 | 3 |
| 0.60 | 0/25 | 3/6 | 3 |
| 0.65 | 1/25 | 3/6 | 4 |
| 0.70 | 3/25 | 2/6 | 5 |
| 0.75 | 8/25 | 0/6 | 8 |
| 0.80 | 16/25 | 0/6 | 16 |

Lowest total errors (observation only): 0.55, 0.60

### Chunk size 1000 (780 chunks)

| Threshold | Answerable rejected | Unanswerable accepted | Total errors |
|---|---|---|---|
| 0.40 | 0/25 | 6/6 | 6 |
| 0.45 | 0/25 | 6/6 | 6 |
| 0.50 | 0/25 | 6/6 | 6 |
| 0.55 | 0/25 | 3/6 | 3 |
| 0.60 | 0/25 | 3/6 | 3 |
| 0.65 | 2/25 | 2/6 | 4 |
| 0.70 | 4/25 | 2/6 | 6 |
| 0.75 | 12/25 | 0/6 | 12 |
| 0.80 | 18/25 | 0/6 | 18 |

Lowest total errors (observation only): 0.55, 0.60

## Table 4: Best dense score

### Chunk size 500 (1741 chunks)

| Group | Min | Median | Max | None |
|---|---|---|---|---|
| Answerable | 0.648 | 0.771 | 0.890 | 0 |
| Unanswerable | 0.489 | 0.592 | 0.733 | 0 |

### Chunk size 1000 (780 chunks)

| Group | Min | Median | Max | None |
|---|---|---|---|---|
| Answerable | 0.614 | 0.754 | 0.832 | 0 |
| Unanswerable | 0.506 | 0.585 | 0.732 | 0 |

## Table 5: Results by overlap group

### Chunk size 500 (1741 chunks)

| Overlap group | Questions | Method | hit@5 | MRR |
|---|---|---|---|---|
| <= 0.40 | 14 | BM25 | 6/14 | 0.229 |
| <= 0.40 | 14 | dense | 10/14 | 0.479 |
| <= 0.40 | 14 | hybrid | 8/14 | 0.310 |
| > 0.40 | 11 | BM25 | 10/11 | 0.909 |
| > 0.40 | 11 | dense | 11/11 | 0.955 |
| > 0.40 | 11 | hybrid | 11/11 | 0.939 |

### Chunk size 1000 (780 chunks)

| Overlap group | Questions | Method | hit@5 | MRR |
|---|---|---|---|---|
| <= 0.40 | 14 | BM25 | 9/14 | 0.458 |
| <= 0.40 | 14 | dense | 11/14 | 0.655 |
| <= 0.40 | 14 | hybrid | 11/14 | 0.599 |
| > 0.40 | 11 | BM25 | 10/11 | 0.909 |
| > 0.40 | 11 | dense | 10/11 | 0.806 |
| > 0.40 | 11 | hybrid | 11/11 | 0.909 |

## Appendix: Per-question results

### Chunk size 500 (1741 chunks)

| id | BM25 | dense | hybrid | Best dense score |
|---|---|---|---|---|
| q01 | - | 1 | 3 | 0.826 |
| q02 | - | 3 | 3 | 0.816 |
| q03 | - | - | - | 0.757 |
| q04 | 3 | - | 5 | 0.718 |
| q05 | 1 | 1 | 1 | 0.648 |
| q06 | 1 | 1 | 1 | 0.890 |
| q07 | 1 | 1 | 1 | 0.723 |
| q08 | 4 | 1 | 2 | 0.780 |
| q09 | 1 | 1 | 1 | 0.768 |
| q10 | - | - | - | 0.741 |
| q11 | - | - | - | 0.789 |
| q12 | 5 | 1 | 3 | 0.771 |
| q13 | 1 | 1 | 1 | 0.858 |
| q14 | 1 | 1 | 1 | 0.816 |
| q15 | 1 | 1 | 1 | 0.765 |
| q16 | - | 3 | 7 | 0.732 |
| q17 | 1 | 1 | 1 | 0.815 |
| q18 | 6 | 2 | 3 | 0.834 |
| q19 | 1 | 1 | 1 | 0.812 |
| q20 | 1 | 2 | 1 | 0.660 |
| q21 | - | 1 | 6 | 0.771 |
| q22 | 4 | 3 | 3 | 0.714 |
| q23 | 1 | 1 | 1 | 0.658 |
| q24 | - | 5 | - | 0.777 |
| q25 | 1 | 1 | 1 | 0.815 |

### Chunk size 1000 (780 chunks)

| id | BM25 | dense | hybrid | Best dense score |
|---|---|---|---|---|
| q01 | - | 1 | 1 | 0.820 |
| q02 | 2 | 3 | 1 | 0.832 |
| q03 | 8 | - | - | 0.722 |
| q04 | 3 | 7 | 4 | 0.714 |
| q05 | 1 | 1 | 1 | 0.623 |
| q06 | 1 | 6 | 2 | 0.808 |
| q07 | 1 | 1 | 1 | 0.715 |
| q08 | 2 | 1 | 1 | 0.771 |
| q09 | 1 | 5 | 2 | 0.728 |
| q10 | 1 | 1 | 1 | 0.684 |
| q11 | 7 | 9 | 6 | 0.775 |
| q12 | 1 | 1 | 1 | 0.754 |
| q13 | 1 | 1 | 1 | 0.822 |
| q14 | 1 | 1 | 1 | 0.785 |
| q15 | 1 | 2 | 1 | 0.756 |
| q16 | 4 | 1 | 3 | 0.758 |
| q17 | 1 | 1 | 1 | 0.722 |
| q18 | 7 | 4 | 5 | 0.811 |
| q19 | 1 | 1 | 1 | 0.742 |
| q20 | 1 | 1 | 1 | 0.614 |
| q21 | - | 1 | 10 | 0.737 |
| q22 | 4 | 3 | 3 | 0.723 |
| q23 | 1 | 1 | 1 | 0.665 |
| q24 | 6 | 1 | 1 | 0.801 |
| q25 | 1 | 1 | 1 | 0.825 |

## How to read this

The question set is small, so a difference of 1 to 2 questions is noise. Hand-written questions tend to share words with the text, which favours keyword search. A threshold picked from Table 3 is tuned on the same questions it is measured on and will look better than it performs on new questions. The gate was swept at top-k 5 only.
