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
```

- `agents/`: Trained model code and artifacts.
- `data/`: Dataset instructions, processed features, and small sample files.
- `docs/`: Project documentation and informal worklog.
- `services/`: FastAPI services for agents and coordination.
- `scripts/`: Repeatable validation scripts.
- `tests/`: Service and end-to-end tests.

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

Copy the example environment file if you want to override thresholds or enable LLM explanations:

```bash
cp .env.example .env
```

Build and start the stack:

```bash
docker compose up --build -d
```

Check the public API:

```bash
curl http://localhost:8000/health
```

Run the full end-to-end validation:

```bash
PYTHON_BIN=python bash scripts/e2e_validation.sh
```

Stop the stack:

```bash
docker compose down
```

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

**Current phase:** Implementation complete; final validation passed; optional local LLM backend added.

The FastAPI services, deterministic decision layer, optional API/local LLM explanation, Docker Compose runtime, and end-to-end validation script are implemented.
