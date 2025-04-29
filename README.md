# AI Evaluation Kickstarter

AI Evaluation Kickstarter is a Streamlit-based application for comparing and evaluating AI models, prompts, and agents. It leverages Azure AI Evaluation SDK, Semantic Kernel, and OpenTelemetry to provide performance, quality, and safety metrics along with logging and monitoring.

## Features

- **Model Evaluation**: Compare outputs from two AI models across various metrics such as groundedness, relevance, coherence, fluency, and NLP scores (F1, BLEU, ROUGE, etc.).
- **Prompt Evaluation**: Evaluate and compare different system messages (prompts) on the same model to determine the best prompting strategy.
- **Agent Evaluation**: Run conversational agents built with Semantic Kernel or OpenAI Agent SDK, view conversation history, and assess agent behavior with custom evaluators.
- **Configurable Metrics**: Select from performance, quality, risk and safety, and agent-specific metrics in the sidebar.
- **Azure Integration**: Uses Azure AI Project, Azure OpenAI, and Azure Monitor (via OpenTelemetry) for evaluation, logging, tracing, and metrics.
- **Download Responses**: Export generated responses and datasets as JSONL files for further analysis.

## Requirements

- Python 3.10+
- Azure subscription with:
  - Azure OpenAI resource
  - Azure AI Project enabled
  - Azure Monitor workspace
- Environment variables configured in `.env` or sample.env

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-org/ai-evaluation-kickstarter.git
   cd ai-evaluation-kickstarter
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\Activate.ps1  # PowerShell
   pip install -r requirements.txt
   ```

3. Copy `sample.env` to `.env` and set your Azure credentials and endpoints:

   ```env
   AZURE_SUBSCRIPTION_ID=...
   AZURE_RESOURCE_GROUP=...
   AZURE_PROJECT_NAME=...
   AZURE_OPENAI_ENDPOINT=...
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_API_VERSION=2024-10-21
   AZURE_OPENAI_INFERENCE_ENDPOINT=...
   SUPPORTED_MODELS=gpt-4o,gpt-4o-mini,gpt-4.1
   PROJECT_CONNECTION_STRING=...
   ```

## Usage

Run the Streamlit application locally:

```powershell
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser and:

1. Select evaluators in the sidebar.
2. Choose between Development and Production modes.
3. In Development:
   - **Evaluate Models**: Upload or enter test data, select two models and a judge LLM, then generate and evaluate responses.
   - **Evaluate Prompts**: Enter two system messages and a prompt, then compare model outputs and metrics.
   - **Evaluate Agents**: Run a chat agent, view conversations, and evaluate tool calls and agent behaviors.
4. Download JSONL datasets of responses for offline analysis.

## Architecture

A high-level overview: the Streamlit UI drives three evaluation flows (model comparison, prompt comparison, and agent evaluation), each using Semantic Kernel for response generation. Generated outputs are compiled into a JSONL dataset, which is fed into the Azure AI Evaluation SDK to run evaluations in Azure AI Foundry. Evaluation results, along with tracing and monitoring data, are then surfaced in the UI and Azure Monitor.

```mermaid
flowchart LR
    UI[Streamlit UI] --> MC(Model Comparison)
    UI --> PC(Prompt Comparison)
    UI --> AE(Agent Evaluation)
    MC --> SK[Semantic Kernel]
    PC --> SK
    AE --> SK
    SK --> RG[Response Generation]
    RG --> JSONL[JSONL Dataset]
    JSONL --> SDK[Azure AI Evaluation SDK]
    SDK --> Foundry[Azure AI Foundry]
    Foundry --> Results[Evaluation Results]
    Foundry --> Mon[Tracing & Monitoring]
``` 

## Contributing

Contributions welcome! Please open issues or pull requests for enhancements or bug fixes.

