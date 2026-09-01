# AI Research & Fact-Verification Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://docs.docker.com/compose/)
[![MCP](https://img.shields.io/badge/MCP-1.0+-purple.svg)](https://modelcontextprotocol.io/)

A multi-agent AI research system built with **LangGraph** that plans queries, retrieves information from a vector database and live web search, synthesizes answers with source attribution, and **verifies them for hallucination risk** before returning results. If verification flags high risk, the system automatically loops back to retrieval with a refined query.

---



### Docker Service Topology

```
┌─────────────────────────────────────────────────────────────┐
│                    docker-compose                           │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Qdrant  │  │ MCP: Qdrant  │  │ MCP: Tavily  │          │
│  │  :6333   │◄─│  :8001 HTTP  │  │  :8002 HTTP  │          │
│  └──────────┘  └──────┬───────┘  └──────┬───────┘          │
│                       │                  │                   │
│                 ┌─────┴──────────────────┴─────┐            │
│                 │    API (FastAPI + LangGraph)  │            │
│                 │         :8000                 │            │
│                 └──────────────────────────────-┘            │
│                                                             │
│  ┌───────────────────┐                                      │
│  │ MCP: Verifier     │  ← Stretch goal (Claude Desktop)    │
│  │  :8003 HTTP       │                                      │
│  └───────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
         External APIs: Groq, Tavily
```

---

## Agents

| Agent | Role | Key Logic |
|-------|------|-----------|
| **Planner** | Decomposes query into sub-questions + search terms | Groq LLM with structured JSON output |
| **Retriever** | Searches Qdrant (local docs) + conditional Tavily (web) | Confidence scoring, merge/dedup |
| **Synthesizer** | Drafts answer with inline `[Source N]` citations | Groq LLM with formatted source context |
| **Verifier** | Detects hallucination risk | Semantic entropy (60%) + Ensemble disagreement (40%) |



1. **Semantic Entropy**: Generate N samples at high temperature → cluster using a **hybrid approach** (embedding cosine similarity + bidirectional NLI contradiction veto via local `microsoft/deberta-large-mnli`) → compute Shannon entropy → normalized risk score.
2. **Ensemble Disagreement**: Extract claims from answers by two Groq models (`openai/gpt-oss-20b` + `openai/gpt-oss-120b`) → check all claim pairs for NLI contradictions → contradiction ratio.
3. **Combined**: `0.6 × entropy_risk + 0.4 × disagreement_score`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph (StateGraph with conditional edges) |
| LLMs | Groq API (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| NLI | `microsoft/deberta-large-mnli` (Local via transformers) |
| Vector DB | Qdrant (Docker) |
| Web Search | Tavily API |
| MCP | Official `mcp` SDK — Streamable HTTP transport |
| Backend | FastAPI (async) |
| Frontend | Vanilla HTML/CSS/JS |
| Containers | Docker Compose |

---

## Setup Instructions

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- API keys (all free tiers):
  - [Groq](https://console.groq.com/keys)
  - [Tavily](https://tavily.com/)

### Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/Research-Agent.git
cd Research-Agent

# 2. Configure
cp .env.example .env
# Edit .env with your API keys:
#   GROQ_API_KEY=gsk_...
#   TAVILY_API_KEY=tvly-...

# 3. Run
docker-compose up --build

# 4. Open
# Dashboard: http://localhost:8000
# API docs:  http://localhost:8000/docs
# Qdrant:    http://localhost:6333/dashboard
```

### Local Development (without Docker)

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (Docker still needed for this)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Update .env for local dev
# QDRANT_HOST=localhost
# MCP_TRANSPORT=stdio

# Run the API
python -m src.api.app
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/upload` | Upload a document (PDF/TXT) → chunk → embed → Qdrant |
| `POST` | `/query` | Run the full agent pipeline → verified answer |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

### Example: Upload + Query

```bash
# Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@my_paper.pdf"
# Response: {"collection_id": "doc_abc123", "n_chunks": 42, ...}

# Query with the uploaded document
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What methodology was used?", "collection": "doc_abc123"}'
```

---

## Example Scenarios

### 1. High Local Confidence → No Web Search → Low Risk

```
Question: "What methodology is described in the uploaded paper?"
→ Retriever: 5 chunks, confidence 0.82 — skipping web search
→ Verifier: entropy risk 0.08, disagreement 0.00
→ Combined: 5% (LOW) ✅
```

### 2. Low Local Confidence → Tavily Triggered → Merged Results

```
Question: "What is the latest research on quantum error correction?"
→ Retriever: 0 local chunks, confidence 0.00 — triggering Tavily
→ Web search: 9 results merged from Tavily
→ Verifier: entropy risk 0.22, disagreement 0.10
→ Combined: 17% (LOW) ✅
```

### 3. High Risk → Critique Loop → Refined Query

```
Question: "What were the outcomes of the 2027 Geneva AI Summit?"
→ Retriever: 0 local, 3 web (low quality)
→ Verifier: entropy risk 0.85, disagreement 0.60
→ Combined: 75% (HIGH) ⚠️ → Refining query...
→ Refined: "2027 Geneva AI Governance Summit outcomes declarations"
→ Retriever (retry 1): 5 web results (better quality)
→ Verifier: entropy risk 0.40, disagreement 0.15
→ Combined: 30% (MEDIUM) — returning with risk score shown
```

---

## MCP Integration

### Using the Verification MCP Server with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-verifier": {
      "url": "http://localhost:8003/mcp"
    }
  }
}
```

Then in Claude Desktop, you can invoke:
> "Use the verify_answer tool to check if this claim is reliable: ..."

---

## Project Structure

```
Research-Agent/
├── docker-compose.yml          # 5-service orchestration
├── Dockerfile.api              # FastAPI + LangGraph
├── Dockerfile.mcp-qdrant       # Qdrant MCP server
├── Dockerfile.mcp-tavily       # Tavily MCP server
├── Dockerfile.mcp-verifier     # Verification MCP server
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
│
├── src/
│   ├── config.py               # Pydantic BaseSettings
│   ├── agents/                 # LangGraph node implementations
│   │   ├── planner.py          # Query decomposition
│   │   ├── retriever.py        # Qdrant + Tavily retrieval
│   │   ├── synthesizer.py      # Answer synthesis with citations
│   │   └── verifier.py         # Hallucination detection
│   ├── graph/
│   │   ├── state.py            # AgentState TypedDict
│   │   └── workflow.py         # StateGraph + conditional edges
│   ├── verification/           # Ported from llm-hallu/
│   │   ├── nli.py              # Local transformers NLI inference
│   │   ├── semantic_entropy.py # Hybrid embedding/NLI clustering
│   │   └── ensemble_disagreement.py
│   ├── services/               # Qdrant, Tavily, embeddings, doc processing
│   ├── mcp_client/             # MCP client for Streamable HTTP
│   └── api/                    # FastAPI routes + models
│
├── mcp_servers/                # Standalone MCP server processes
│   ├── qdrant_server.py
│   ├── tavily_server.py
│   └── verifier_server.py
│
└── frontend/
    └── index.html              # Vanilla HTML/CSS/JS dashboard (UI)
```

---

## Related Projects

- **[llm-hallu](../llm-hallu/)** — The original hallucination detection pipeline that this project's verification module is built upon.
