# Multi-Agent Disinformation Detection

## Project Overview

This project aims to detect suspicious bot activity and disinformation on social networks using a multi-agent machine learning architecture.

The system combines different analysis approaches to examine content, account behavior, and potential coordination between accounts. The goal is to identify suspicious activity and support informed moderation decisions.

This project is developed as part of a university machine learning project.

## Objectives

- Analyze social media content to identify suspicious or misleading content.
- Examine temporal account behavior to detect patterns associated with automated activity.
- Analyze coordination patterns between accounts.
- Combine multiple sources of evidence to estimate campaign risk.
- Produce interpretable predictions and moderation recommendations.

## Architecture

The system consists of three machine learning agents and a shared coordination layer.

| Component | Responsibility | Technique |
|---|---|---|
| Agent 1 | Content analysis | Naive Bayes |
| Agent 2 | Temporal behavior analysis | Recurrent Neural Network (RNN/LSTM) |
| Coordination Layer | Cross-account analysis and coordination features | Python-based analysis |
| Agent 3 | Evidence integration and risk classification | Logistic Regression |

The coordination layer is shared infrastructure, not a fourth machine learning agent.

Each agent has a separate responsibility and will be developed independently before integration.

## Repository Structure

```text
agents/
  agent1_naive_bayes/
  agent2_rnn/
  agent3_logistic_regression/
coordination/
data/
  raw/
  processed/
  samples/
docs/
tests/
```

- `agents/`: Individual machine learning agents.
- `coordination/`: Shared logic for analyzing relationships and coordination between accounts.
- `data/`: Dataset instructions, local datasets, and small sample files.
- `docs/`: Project documentation and architectural decisions.
- `tests/`: Integration and system-level tests.

Each agent directory contains its own source code, tests, and documentation.

## Datasets

The project uses the following datasets as its initial data sources:

- **Twitter Bot Detection Dataset** — for investigating automated account behavior.
- **Fake News Detection Dataset** — for investigating potentially misleading content.

Dataset preparation, suitability, and the availability of the labels required for each task will be documented before model training begins.

See [`docs/dataset-notes.md`](docs/dataset-notes.md) for further details.

## Development Workflow

1. Agree on the interfaces and responsibilities of each agent.
2. Inspect and prepare the datasets.
3. Develop and evaluate each agent independently.
4. Implement the coordination layer.
5. Integrate the agents and train/evaluate the final classifier.
6. Run integration tests and prepare the demonstration.

Development is organized through Git and GitHub. Each team member should work on their assigned component and communicate interface changes before integration.

## Data and Model Artifacts

Raw datasets, processed datasets, and trained model artifacts are not committed to the repository by default.

Small, permitted sample files may be stored in `data/samples/` for testing and demonstration.

See [`data/README.md`](data/README.md) for data management rules.

## Project Status

**Current phase:** Architecture, documentation, dataset investigation, and team coordination.

Model implementation and integration will follow once the interfaces and data requirements are agreed upon.
