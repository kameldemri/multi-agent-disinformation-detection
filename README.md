# Multi-Agent Disinformation Detection

## Project Overview

This project aims to detect suspicious bot activity and disinformation on social networks using a multi-agent machine learning architecture.

The system combines different analysis approaches to examine content, account behavior, and potential coordination between accounts. The goal is to identify suspicious activity and support informed moderation decisions.

This project is developed as part of a university machine learning project.

## Objectives

- Analyze social media content to identify suspicious or misleading content.
- Examine temporal account behavior to detect patterns associated with automated activity.
- Combine both evidence sources into a single account-level risk decision.
- Produce interpretable predictions and moderation recommendations.
- Package the full stack so it runs anywhere with Docker Compose.

## Architecture

The implemented system consists of two trained machine learning agents, one deterministic decision agent, and one public coordination service.

| Component | Responsibility | Technique |
|---|---|---|
| Agent 1 | Fake-news content analysis | Naive Bayes |
| Agent 2 | Bot-like temporal behavior analysis | BiGRU recurrent neural network |
| Agent 3 | Final decision and explanation | Deterministic policy plus optional LLM explanation |
| Coordinator | Public API orchestration | FastAPI HTTP service |

Agent 3 uses only `fake_news_probability` and `bot_probability` for its decision. The LLM can only write the explanation; it cannot change the decision or risk score.

## Repository Structure

```text
agents/
    agent1_naive_bayes/
    agent2_rnn/
data/
    raw/
    processed/
    samples/
docs/
services/
    agent1_service/
    agent2_service/
    agent3_service/
    coordinator_service/
scripts/
tests/
web/
```

- `agents/`: Trained model code and artifacts.
- `data/`: Dataset instructions, processed features, and small sample files.
- `docs/`: Project documentation and informal worklog.
- `services/`: FastAPI services for agents and coordination.
- `scripts/`: Repeatable validation scripts.
- `tests/`: Service and end-to-end tests.
- `web/`: Dependency-free browser UI served by the coordinator.

## Datasets

The project uses the following datasets as its initial data sources:

- **Twitter Bot Detection Dataset** — for investigating automated account behavior.
- **Fake News Detection Dataset** — for investigating potentially misleading content.

Dataset preparation, suitability, and the availability of the labels required for each task will be documented before model training begins.

See [`docs/dataset-notes.md`](docs/dataset-notes.md) for further details.

## Development Workflow

1. Agree on the interfaces and responsibilities of each agent.
2. Train and evaluate each model independently.
3. Wrap each model in an internal HTTP service.
4. Implement the deterministic decision layer and optional explanation.
5. Expose the system through the coordinator API.
6. Package all services with Docker Compose.
7. Run service and end-to-end validation.

Development is organized through Git and GitHub. Interface changes must be communicated before implementation.

## Development Setup

### Prerequisites

- Git
- Python 3.12

### Installation

Clone the repository:

```bash
git clone https://github.com/kameldemri/multi-agent-disinformation-detection.git
cd multi-agent-disinformation-detection
```

Create a virtual environment using Python 3.12:

**Linux / macOS:**

```bash
python3.12 -m venv .venv
```

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
```

Activate the virtual environment:

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the Python version:

```bash
python --version
```

It should report Python 3.12.x.

### Environment Guidelines

- Always activate the virtual environment before working on the project.
- Install project dependencies using `requirements.txt`.
- Do not commit the `.venv/` directory.
- If you add or modify dependencies, update `requirements.txt` and communicate the changes to the team.

## Docker Quickstart

Everything needed to run the app is Dockerized: runtime dependencies, trained model artifacts, FastAPI services, and the browser UI. Docker Compose builds one application image and starts four containers by default. The optional `llm` profile adds Ollama as a fifth service. On a fresh PC, the first build needs internet access to pull Docker images and install dependencies; after that, the default stack runs offline.

### Linux

Prerequisites:

- Docker Engine
- Docker Compose plugin
- Git
- Internet access for the first build

Clone and start:

```bash
git clone https://github.com/kameldemri/multi-agent-disinformation-detection.git
cd multi-agent-disinformation-detection
cp .env.example .env
docker compose up --build -d
```

Open the UI:

```text
http://localhost:8000/
```

Check the API:

```bash
curl http://localhost:8000/health
docker compose ps
```

Optional validation, which requires Python 3.12 with `httpx` installed on the host:

```bash
PYTHON_BIN=python bash scripts/e2e_validation.sh
```

The Docker run above does not require a local Python installation.

Stop the stack:

```bash
docker compose down
```

### Windows

Prerequisites:

- Docker Desktop
- Git for Windows
- Linux containers enabled in Docker Desktop
- Internet access for the first build

Run in PowerShell:

```powershell
git clone https://github.com/kameldemri/multi-agent-disinformation-detection.git
cd multi-agent-disinformation-detection
Copy-Item .env.example .env
docker compose up --build -d
```

Open the UI:

```text
http://localhost:8000/
```

Check the API:

```powershell
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
docker compose ps
```

Stop the stack:

```powershell
docker compose down
```

If `docker compose` is not recognized, enable Docker Compose v2 in Docker Desktop. On older installs, use `docker-compose` instead.

If port `8000` is already in use, set `COORDINATOR_PORT` in `.env` to another port and use that port in the UI/API URL.

The Docker run above does not require a local Python installation. For optional end-to-end validation, install Python 3.12 with `httpx`, then run `PYTHON_BIN=python bash scripts/e2e_validation.sh` from WSL or Git Bash.

### Optional Local LLM

The default stack is free and offline. To enable the local Ollama model, edit `.env` before starting:

```env
LLM_MODE=api
LLM_ALLOW_NO_KEY=true
LLM_BASE_URL=http://llm:11434/v1
LLM_MODEL=qwen2.5:0.5b-instruct
LLM_LOCAL_MODEL=qwen2.5:0.5b-instruct
```

Then start the optional profile:

```bash
docker compose --profile llm up --build -d
```

This works the same in Linux shell and Windows PowerShell. The first start pulls the Ollama image and the `qwen2.5:0.5b-instruct` model, so it needs internet once. After that, the model is stored in the `ollama_models` Docker volume and can be reused offline.

The coordinator is the only published container. Agent services communicate over the internal Docker network.

## Local LLM

The default stack stays free and offline. To enable a free local LLM explanation backend, set these values in `.env`:

```env
LLM_MODE=api
LLM_ALLOW_NO_KEY=true
LLM_BASE_URL=http://llm:11434/v1
LLM_MODEL=qwen2.5:0.5b-instruct
LLM_LOCAL_MODEL=qwen2.5:0.5b-instruct
```

Start with the optional profile:

```bash
docker compose --profile llm up -d --build
```

The first start pulls the model into a named Docker volume. CPU inference works but is slower than GPU inference. If the model is not ready, Agent 3 falls back to the deterministic reason.

Agent 3 accepts JSON (`{"reason": "..."}`) or plain-text reason responses from the model. This was verified locally with `qwen2.5:0.5b-instruct`: `/health` returned `{"status":"ok"}` and `/analyze` returned `llm_status: "used"` with a model-written reason.

Successful local LLM explanations return `llm_status: "used"`. Offline mode returns `"disabled"`, and failed API/local calls return `"fallback"`.

Do not commit `.env` or API keys.

## Bring Your Own API Key

LLM explanations are optional. The default `LLM_MODE=offline` needs no API key and no model download. To use an OpenAI-compatible hosted API, edit `.env` before starting the stack:

```env
LLM_MODE=api
LLM_API_KEY=your_api_key_here
LLM_ALLOW_NO_KEY=false
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Then start or restart the default stack:

```bash
docker compose up --build -d
```

Do not use the `llm` profile for a hosted API; that profile is only for the local Ollama model. For another provider, change `LLM_BASE_URL` and `LLM_MODEL` to that provider's OpenAI-compatible values.

The API key affects only the explanation text. Agent 3's decision and risk score stay deterministic. If the key is missing or invalid, Agent 3 falls back to the deterministic reason and returns `llm_status: "fallback"`. A successful hosted call returns `llm_status: "used"`.

Never commit `.env` or API keys.

## Web UI

After starting the stack, open:

```text
http://127.0.0.1:8000/
```

The UI submits to the same `POST /analyze` endpoint used by API clients. Enter text, an optional title, a user ID, and at least three ISO-8601 tweet timestamps, one per line. The **Use sample** button fills a ready-to-run example.

The result panel shows the final decision, final/content/behavior risk, agent labels, the explanation, the LLM status, and the raw JSON response. It works with offline mode, BYO API key mode, and the optional local LLM profile.

## API

The public endpoint is:

```text
POST /analyze
```

It accepts one text sample and one account's temporal tweet data, then returns Agent 1, Agent 2, and Agent 3 outputs. Article-only and account-only public flows are not supported.

## Data and Model Artifacts

Raw datasets and large processed files are not committed by default.

Trained inference artifacts are tracked so the services can run without retraining or downloading data:

- `agents/agent1_naive_bayes/models/nb_fake_news.joblib`
- `agents/agent2_rnn/artifacts/best_bigru_safe.pt`
- `data/processed/agent2/temporal_scaler.npz`

Small, permitted sample files may be stored in `data/samples/` for testing and demonstration.

See [`data/README.md`](data/README.md) for data management rules.

## Project Status

**Current phase:** Implementation complete; final validation passed; optional local LLM backend and browser UI added.

The FastAPI services, deterministic decision layer, optional API/local LLM explanation, Docker Compose runtime, browser UI, and end-to-end validation script are implemented.
