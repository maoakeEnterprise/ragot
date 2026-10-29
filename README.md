*This project has been created as part of the 42 curriculum by mteriier.*

# RAG against the machine

## Description

This project is a **Retrieval-Augmented Generation (RAG)** system that answers
natural-language questions about a codebase: the
[vLLM 0.10.1](https://github.com/vllm-project/vllm) repository.

It covers the four steps of a RAG pipeline:

1. **Indexing**: read the useful files of the vLLM repository, split them into
   chunks of at most 2000 characters and build a lexical BM25 index.
2. **Retrieving**: given a question, return the top-k most relevant source
   locations (`file_path` + character range).
3. **Augmenting**: put the retrieved snippets into the model's context window
   within a token budget.
4. **Generating**: answer the question with **Qwen/Qwen3-0.6B**, grounded in the
   retrieved context.

Every result is written as JSON that follows the pydantic models of the subject
(`StudentSearchResults`, `StudentSearchResultsAndAnswer`), so it can be scored by
the provided moulinette.

## Instructions

### Requirements

- Python 3.13 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/)
- The vLLM corpus extracted in `data/raw/vllm-0.10.1/`
- A few GB of free disk space (PyTorch + Qwen3-0.6B weights). A GPU is optional;
  the system runs on CPU.

### Installation

```bash
make install            # uv sync
unzip vllm-0.10.1.zip -d data/raw/
```

### Makefile rules

| Rule | Description |
|---|---|
| `make install` | Install dependencies with `uv sync` |
| `make run` | Run the main module (`uv run python -m src`) |
| `make debug ARGS="..."` | Run the main module under `pdb` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `make lint` | `flake8` + `mypy` with the flags required by the subject |
| `make index` | Build the index with `--max_chunk_size 2000` |
| `make moulinette_doc` / `make moulinette_code` | Search a dataset and score it with the moulinette |
| `make answer Q="..."` | Answer a single question |
| `make gpu_available` | Check whether CUDA is available |

## Example usage

Every command runs through `uv run python -m src <command> [options]`.

**1. Index the corpus**

```bash
uv run python -m src index --max_chunk_size 2000
```

**2. Search a single question**

```bash
$ uv run python -m src search "How do I configure the KV cache in vLLM?" --k 5
data/raw/vllm-0.10.1/examples/others/logging_configuration.md [0:1590]
data/raw/vllm-0.10.1/docs/usage/security.md [3354:4366]
data/raw/vllm-0.10.1/examples/others/logging_configuration.md [5059:5617]
data/raw/vllm-0.10.1/vllm/logger.py [0:1863]
data/raw/vllm-0.10.1/tests/v1/kv_connector/unit/test_multi_connector.py [1172:3152]
```

**3. Search a whole dataset**

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

**4. Score it with the moulinette**

```bash
./moulinette/moulinette-ubuntu evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

**5. Generate answers**

```bash
uv run python -m src answer "How do I enable LoRA adapters?" --k 5

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

**6. Evaluate recall@k locally (no moulinette)**

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Every input and output path is a CLI argument. `--index_dir` (default
`data/processed/`) can also be passed to choose which index to use.

## System architecture

```
data/raw/vllm-0.10.1
        │
        ▼
   Chunker  ──(.py → Python splitter, .md → Markdown splitter, .txt → text splitter)
        │      list[Chunk]  (file_path, first/last_character_index, content)
        ▼
   DataManager ──► data/processed/chunks.json   (all chunks, pydantic ChunkIndex)
        │      └─► data/processed/bm25/         (bm25s index)
        ▼
   Retriever   (Tokenizer + BM25 → top-k MinimalSource)
        │
        ▼
   ChunkLookUp (MinimalSource → chunk text)
        │
        ▼
   ManagerLLM  (builds the context within a token budget → Qwen3-0.6B → answer)
        │
        ▼
   JSON output (StudentSearchResults / StudentSearchResultsAndAnswer)
```

| Module | Role |
|---|---|
| `src/__main__.py` | Entry point: exposes `Rag` through Python Fire and turns every expected exception into a clean error message (no traceback) |
| `utils/rag.py` | `Rag` class: the CLI commands (`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`) |
| `utils/chunker.py` | Lists the corpus files and splits them into chunks |
| `utils/tokenizer.py` | Code-aware tokenizer used both at indexing and at query time |
| `utils/data_manager.py` | Saves and loads the chunks and the BM25 index |
| `utils/retrievier.py` | BM25 retrieval of the top-k chunks |
| `utils/manager_llm.py` | Context building and answer generation with Qwen3-0.6B |
| `utils/models.py` | pydantic data models |

## Chunking strategy

Only `.py`, `.md` and `.txt` files are indexed (1847 files → 14 528 chunks).
Each file type has its own strategy, all built on LangChain's
`RecursiveCharacterTextSplitter`:

- **Python code**: `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`.
  It splits first on `class` and `def` boundaries, then on blank lines, then on
  lines, so a chunk usually holds a whole function or class.
- **Markdown**: `from_language(Language.MARKDOWN)`. It splits first on headings
  (`#`, `##`, ...), then on code blocks and paragraphs, so a chunk usually holds
  one section of the documentation.
- **Plain text**: the default recursive splitter (paragraphs → lines → words).

Settings:

- `chunk_size = --max_chunk_size` (default 2000, allowed range `[201, 2000]`).
  The largest chunk is exactly 2000 characters, so a source never exceeds the
  moulinette's `max_context_length`.
- `chunk_overlap = 100` characters, so text cut at a chunk border also appears
  in the next chunk.
- `add_start_index=True` gives the exact character offset of every chunk in its
  file. `last_character_index = start + len(content)`.
- `file_path` is stored exactly as it appears in the corpus
  (`data/raw/vllm-0.10.1/...`), because the grader compares paths verbatim.

## Retrieval method

Retrieval is lexical, with **BM25** (library `bm25s`, `k1 = 1.5`, `b = 0.75`).

BM25 scores a chunk by summing, over the query terms, the IDF of the term (rare
terms weigh more) multiplied by a term-frequency factor. That factor saturates
(controlled by `k1`) and is normalised by the chunk length (controlled by `b`).
The chunks are ranked by score and the top-k are returned.

Questions and code rarely use the same words, so the key part is the
**tokenizer**, which runs both at indexing and at query time:

1. Extract identifiers with `[A-Za-z_][A-Za-z0-9_]*` (punctuation and operators
   are dropped).
2. Remove English stop words (NLTK) and tokens shorter than 2 characters.
3. Split identifiers on `snake_case` and `CamelCase`
   (`gpu_memory_utilization` → `gpu`, `memory`, `utilization`;
   `LLMEngine` → `llm`, `engine`).
4. Keep the **full identifier as well** (`gpu_memory_utilization`), so a
   question that quotes an identifier verbatim gets an exact, high-IDF match,
   and a question that paraphrases it still matches its parts.
5. Lowercase everything.

## Performance analysis

Scored with the provided moulinette, `--max_chunk_size 2000`, `--k 10`:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Threshold (@5) |
|---|---|---|---|---|---|
| docs (100 questions) | 0.630 | 0.790 | **0.870** | 0.870 | ≥ 0.80 ✅ |
| code (99 questions) | 0.444 | 0.646 | **0.697** | 0.697 | ≥ 0.50 ✅ |

| Criterion | Measured | Requirement |
|---|---|---|
| Indexing time (whole corpus) | ~8 s | ≤ 5 min ✅ |
| Retrieval throughput | ~5 s for 100 questions (search + scoring) | ≤ 90 s for 200 questions ✅ |

Docs questions score better than code questions: their wording is close to the
Markdown docs, while code questions often paraphrase what a function does
without using its identifiers. Large chunks (2000 characters) help recall,
because a bigger window is more likely to overlap the reference span (IoU >
0.05). Smaller chunks give more precise sources but lower recall.

Answer generation with Qwen3-0.6B is slower: it depends on the hardware and is
much faster on GPU (`make gpu_available`).

## Design decisions

- **BM25 over TF-IDF**: length normalisation and term-frequency saturation work
  better with chunks of very different sizes (a short README section vs. a long
  Python class).
- **`bm25s`**: fast indexing and retrieval, and saving/loading to disk is built
  in, so a query runs in milliseconds once the index is loaded.
- **LangChain language-aware splitters**: they follow the syntax of each file
  type (functions, headings) instead of cutting blindly every N characters.
- **Chunks stored in `chunks.json`**: the text of each source can be looked up
  (`ChunkLookUp`) at answer time without re-reading or re-chunking the corpus.
- **Context budget of 3000 tokens**: retrieved chunks are added in rank order
  until the budget is reached, which keeps the prompt within what Qwen3-0.6B
  handles well.
- **Prompt strategy**: the system prompt asks the model to answer *only* from
  the context, to say "I don't know" otherwise, and to be concise, which limits
  hallucination. Each snippet is prefixed with its file path. Thinking mode is
  disabled (`enable_thinking=False`) and decoding is greedy (`do_sample=False`)
  for short, deterministic answers.
- **Lazy model loading**: Qwen is loaded only when an answer is needed, so
  `index`, `search` and `evaluate` never load PyTorch weights.
- **Robust CLI**: inputs are validated (`k ≥ 1`, `max_chunk_size` in
  `[201, 2000]`, empty queries, missing index). All expected exceptions are
  caught in `__main__.py` and printed as readable errors.

## Challenges faced

- **Questions vs. code vocabulary**: plain word splitting missed questions that
  quote identifiers like `max_model_len`. The fix was splitting `snake_case` and
  `CamelCase` while also keeping the full identifier.
- **Exact source locations**: the grader needs exact character offsets and
  verbatim paths. `add_start_index=True` and storing the corpus path unchanged
  fixed both.
- **The 2000-character limit**: one source that is too long invalidates the whole
  output, so `max_chunk_size` is capped at 2000 and the splitter never returns
  bigger chunks.
- **Degenerate inputs**: empty queries, `k=0`, non-integer arguments, missing
  files and malformed JSON all had to end with a clean error instead of a
  traceback.
- **Typing with transformers / torch**: some third-party types cannot be
  resolved by mypy, so a few targeted `# type: ignore[...]` were needed to pass
  `make lint`.

## Resources

### Chunking

- https://docs.langchain.com/oss/python/integrations/splitters/code_splitter
- https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter
- https://reference.langchain.com/python/langchain-text-splitters

### BM25 / TF-IDF

- https://www.elastic.co/fr/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
- https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency
- https://fr.wikipedia.org/wiki/Okapi_BM25
- https://pypi.org/project/BM25/
- https://github.com/xhluca/bm25s

### LLM

- https://huggingface.co/Qwen/Qwen3-0.6B
- https://huggingface.co/docs/transformers/main/en/chat_templating
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020), https://arxiv.org/abs/2005.11401

### Use of AI

- **README**: an AI assistant (Claude) drafted this README from the subject and
  the existing code. I reviewed it and checked it against the project.
- **Learning the concepts**: I used AI as a tutor throughout the project. It
  explained each concept I needed (RAG pipeline, chunking strategies, BM25 and
  TF-IDF, tokenization of code identifiers, pydantic models, Python Fire, LLM
  generation with Qwen3-0.6B) and showed small examples.
- **Writing the code**: I then wrote the code myself, based on what I had
  understood from those explanations, and did not copy-paste generated code
  into the project.
- **Review**: AI was also used to check the project against the subject and
  list the missing requirements.
