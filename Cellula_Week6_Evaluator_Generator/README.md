# Evaluator-Generator Knowledge Platform (Project 3)

A question answering platform where two LLM agents work in a loop. The generator writes an
answer from retrieved documents, the evaluator scores it, and if the score is not good enough
the feedback goes back to the generator. The loop stops when the evaluator accepts the answer
or after 4 rounds.

Built with LangChain chains, LCEL orchestration, Chroma for retrieval and Redis for caching.

## Supported knowledge sources

| Input | Loader |
|---|---|
| PDF | pypdf (PyPDFLoader) |
| DOCX | python-docx |
| TXT / MD | direct read |
| Source code files | direct read, language kept in metadata |
| PPTX | python-pptx (old .ppt is not supported) |
| Web page URL | requests + BeautifulSoup |
| Wikipedia page (title or link) | Wikipedia REST API |
| WAV audio | faster-whisper in a separate process |

## Setup

Tested on macOS with Python 3.9.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

Redis is not installed on this machine, so the project keeps a local build:

```bash
mkdir -p third_party && cd third_party
curl -L https://download.redis.io/redis-stable.tar.gz | tar xz
cd redis-stable && make -j4 MALLOC=libc
```

Put the API key and model in `app.env` (this file is gitignored, so start from the template):

```bash
cp app.env.example app.env
```

```
OPENROUTER_API_KEY=...
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
REDIS_URL=redis://localhost:6379/0
```

## Run

```bash
./start.sh
```

This starts Redis (if it is not running) and the Streamlit app on port 8501.

## How the workflow runs

```
question
  -> retrieve top 4 chunks from Chroma (cached in Redis)
  -> generator writes an answer with [source] citations
  -> evaluator returns JSON: acceptable, score, relevance, completeness, grounding, feedback
  -> accepted? return the answer : send feedback back to the generator
  -> stop after 4 rounds and return the latest answer with a warning
```

The whole step is one LCEL chain in `workflow.py`:

```python
self.step = (
    RunnablePassthrough.assign(documents=RunnableLambda(self._retrieve))
    | self.generator.as_runnable()
    | self.evaluator.as_runnable()
    | RunnableLambda(self._record)
)
```

## Memory isolation

Generator memory lives in Redis db 1, evaluator memory in Redis db 2. Each agent only holds a
client for its own database, so neither can read the other's history even by accident. The
generator stores each attempt (question, answer, sources, feedback used). The evaluator stores
each review (scores, decision, feedback).

## Caching

| Key prefix | Content | Invalidated by |
|---|---|---|
| `retrieval:<kb version>:<hash>` | retrieval results per question | new upload (version bump) |
| `answer:<kb version>:<hash>` | final accepted answers | new upload |
| `processed:<file sha256>` | already ingested files | clear knowledge base |
| langchain llm cache | raw LLM responses | 24h TTL |

Every new upload bumps `version:kb`, so stale cached results are never reused.

## Project layout

| File | Purpose |
|---|---|
| `config.py` | paths and settings from app.env |
| `loaders.py` | one loader per source format |
| `stt_worker.py` | speech to text, runs as its own process |
| `ingest.py` | load, split, embed, store |
| `knowledge.py` | Chroma store and cached search |
| `cache.py` | Redis cache wrapper and metrics |
| `memory.py` | generator and evaluator memories |
| `prompts.py` | generator and evaluator prompt templates |
| `generator.py` | generator agent (LCEL step) |
| `evaluator.py` | evaluator agent (LCEL step) |
| `workflow.py` | feedback loop, max 4 iterations |
| `app.py` | Streamlit interface |
| `docs/` | workflow diagram and the architecture document |

## Notes

- The LLM runs on the OpenRouter free tier, so a request can fail with a rate limit. The
  workflow stops cleanly and shows the error, the answer is not cached.
- Audio transcription is CPU based, a short recording takes a few seconds.
- `sample_data/` contains one source of each format to test the pipeline.
