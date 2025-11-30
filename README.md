# Multi-Agent RAG & Routing System

This project implements an intent‑aware, multi‑agent Retrieval Augmented Generation (RAG) system for three internal domains: **HR**, **IT/Tech**, and **Finance**. An Orchestrator decides which domain agent(s) to invoke. Each specialized agent retrieves relevant internal documentation chunks and (optionally) uses an LLM to synthesize an answer. A lightweight **offline / deterministic mode** enables running without any external model or network connectivity.

## 1. Quick Start (Windows PowerShell)

```powershell
# (Optional) Create & activate virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1

# Install dependencies (may emit numpy conflict warning; see Section 7)
python -m pip install --upgrade pip
pip install -r requirements.txt

# Offline deterministic run (no network, no LLM, no Langfuse, no evaluation)
$env:DISABLE_LLM="1"; $env:DISABLE_EVALUATION="1"; $env:DISABLE_LANGFUSE="1"
python -m src.multi_agent_system "What are the vacation leave policies?"

# Re‑enable LLM + evaluation (requires working OpenRouter / OpenAI connectivity)
Remove-Item Env:DISABLE_LLM -ErrorAction SilentlyContinue
Remove-Item Env:DISABLE_EVALUATION -ErrorAction SilentlyContinue
Remove-Item Env:DISABLE_LANGFUSE -ErrorAction SilentlyContinue
python -m src.multi_agent_system "How do I submit an expense report?"
```

## 2. Modes of Operation

| Mode | Description | How to Enable |
|------|-------------|---------------|
| LLM Mode | Uses ChatOpenAI via OpenRouter/OpenAI for intent classification + answer synthesis + evaluator scoring | Ensure API key present; do NOT set `DISABLE_LLM` or `DISABLE_EVALUATION` |
| Offline Retrieval Mode | No external calls; returns structured sentence extraction (deterministic) | Set `DISABLE_LLM=1` (evaluation auto-skipped or also set `DISABLE_EVALUATION=1`) |
| Evaluation Disabled | Skips quality scoring completely | Set `DISABLE_EVALUATION=1` |
| Observability Disabled | Skips Langfuse tracing (avoids network DNS failures) | Set `DISABLE_LANGFUSE=1` |

If LLM requests hang (e.g. slow network / blocked DNS), switch to offline mode.

## 3. Architecture Overview

```
User Query
   ↓
Orchestrator (LLM or keyword intent)
   ↓                     ┌──────────────┐
   ├─ HR Agent (RAG) ───►│ HR TF‑IDF    │
   ├─ Tech Agent (RAG) ─►│ Tech TF‑IDF  │
   └─ Finance Agent ────►│ Finance TF‑IDF│
                        │
                  Structured / LLM Answer
                        │
                 (Optional) Evaluator (LLM)
                        │
                   Final Response JSON
```

Key implementation notes:
- Vector store replaced with an **in‑memory TF‑IDF retriever** (no Chroma persistence; fast startup; zero heavy dependencies).
- Fallback summaries (offline mode) are **section‑structured** for each domain to improve relevance and completeness.
- Multi‑domain queries aggregate responses from multiple agents when ≥2 domain keyword groups appear.

## 4. Repository Structure

```
Assigment03/
├─ README.md
├─ requirements.txt
├─ .env (user-provided; not committed)
├─ src/
│  ├─ multi_agent_system.py          # CLI entrypoint
│  ├─ config.py                      # Configuration & env loading
│  ├─ agents/                        # Orchestrator + specialized agents
│  ├─ utils/                         # Document loader, vector store, Langfuse setup
│  └─ evaluator/                     # Optional LLM-based evaluator
├─ data/                             # Domain document folders (each ≥50 chunks)
├─ tests/                            # Test queries JSON
└─ multi_agent_system.ipynb          # Notebook for interactive exploration
```

### Detailed Tree

```
multi_agent_system.ipynb
output_hr.txt
README.md
requirements.txt
test_llm.py
data/
   finance_docs/
      accounting_procedures.txt
      budget_guidelines.txt
      expense_policy.txt
      financial_reporting.txt
      invoice_procedures.md
      payment_processing.txt
   hr_docs/
      compensation_policy.txt
      employee_benefits.txt
      employee_handbook.txt
      employee_relations.txt
      leave_policy.txt
      onboarding_process.txt
      performance_review.md
      recruitment_policy.txt
      termination_policy.txt
      training_development.txt
      workplace_conduct.md
   tech_docs/
      it_support_guide.txt
      network_administration.txt
      security_policies.txt
      software_access.txt
      system_access.txt
      troubleshooting.md
src/
   __init__.py
   config.py
   multi_agent_system.py
   __pycache__/
   agents/
      __init__.py
      base_agent.py
      finance_agent.py
      hr_agent.py
      orchestrator.py
      tech_agent.py
      __pycache__/
   evaluator/
      __init__.py
      evaluator_agent.py
      __pycache__/
   utils/
      __init__.py
      document_loader.py
      langfuse_setup.py
      vector_store.py
      __pycache__/
tests/
   __init__.py
   test_queries.json
```

## 5. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested with 3.13)
- Optional: OpenRouter or OpenAI API key for LLM features
- Optional: Langfuse keys for tracing

### Install Dependencies
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables (.env)
Create `.env` file (or set directly in shell):
```dotenv
# Choose one: OpenRouter or OpenAI
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Optional explicit model (falls back to openrouter/auto)
OPENAI_MODEL=openrouter/auto

# Optional Langfuse (omit or set DISABLE_LANGFUSE=1 to skip)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

If you only want offline deterministic behavior you may skip all keys and set:
```powershell
$env:DISABLE_LLM="1"; $env:DISABLE_EVALUATION="1"; $env:DISABLE_LANGFUSE="1"
```

## 6. Running Queries

### Single-Domain Examples
```powershell
# HR
python -m src.multi_agent_system "What are the vacation leave policies?"
# Tech
python -m src.multi_agent_system "How do I reset my password?"
# Finance
python -m src.multi_agent_system "How do I submit an expense report?"
```

### Multi-Domain Example (aggregates sections)
```powershell
python -m src.multi_agent_system "How do I reset my password, submit an expense report, and what are the vacation leave policies?"
```

### Offline Mode (guaranteed deterministic)
```powershell
$env:DISABLE_LLM="1"; $env:DISABLE_EVALUATION="1"; $env:DISABLE_LANGFUSE="1"
python -m src.multi_agent_system "How do I submit an expense report?"
```

### Rebuild Vector Stores (rarely needed)
All stores are ephemeral TF‑IDF; to force reload just run again. Set `rebuild_vector_stores=True` in code if you alter loader logic.

## 7. Dependency Notes & Known Conflicts

Current `requirements.txt` pins `numpy==2.3.5` but **LangChain 0.1.20 requires `numpy < 2`**. If you encounter resolution errors:
```powershell
pip install "numpy<2" --force-reinstall
```
Or edit `requirements.txt` to use:
```
numpy==1.26.4
```

Because we replaced embedding-based vector stores with TF‑IDF + scikit‑learn, you can remove heavy packages (e.g., chromadb, tiktoken) if not needed.

## 8. Configuration Summary (src/config.py)

| Setting | Purpose | Default |
|---------|---------|---------|
| CHUNK_SIZE | Character chunk size for docs | 1000 |
| CHUNK_OVERLAP | Overlap between chunks | 200 |
| TOP_K_RETRIEVAL | Docs per retrieval call | 5 |
| OPENAI_MODEL | LLM model alias | openrouter/auto |
| OPENAI_BASE_URL | Auto-select OpenRouter if key present | dynamic |

Environment toggles (not in config class): `DISABLE_LLM`, `DISABLE_EVALUATION`, `DISABLE_LANGFUSE`.

## 9. Evaluation & Observability

- Evaluator uses an LLM to score relevance / completeness / accuracy.
- Auto-skipped when `DISABLE_LLM=1` or `DISABLE_EVALUATION=1`.
- Langfuse disabled automatically if keys missing or `DISABLE_LANGFUSE=1`.

**If network is blocked** (DNS failures like `getaddrinfo failed`): disable Langfuse & LLM as shown.

## 10. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Hanging on query | LLM network timeout | Set `DISABLE_LLM=1` |
| DNS error (`cloud.langfuse.com`) | No network / blocked | Set `DISABLE_LANGFUSE=1` |
| Numpy conflict | `numpy==2.3.5` vs LangChain constraint | Install `numpy<2` |
| Repeated fragments in offline answer | Overlapping chunk retrieval | Reduce `k` or refine section extraction |
| Low evaluation scores | Fragmented fallback output | Enable LLM or refine docs / chunking |

## 11. Extending the System

1. Add new domain folder under `data/` (e.g. `legal_docs/`).
2. Load & chunk via existing `DocumentLoader`.
3. Add new agent subclass of `BaseRAGAgent` replicating structured fallback pattern.
4. Update `MultiAgentSystem._setup_vector_stores` and orchestrator keyword heuristics.
5. (Optional) Add domain to multi-domain detection lists.

## 12. Roadmap / Future Improvements
- Replace deprecated `retriever.get_relevant_documents` with `.invoke()` usage.
- Sentence-aware chunk splitter (avoid mid-sentence truncation).
- De-duplicate lines in structured fallback sections.
- Confidence score for multi-domain detection.
- REST API / FastAPI wrapper.
- Persistent vector store option (if embeddings restored).

## 13. License & Purpose
Educational assignment project; not production-hardened. For production: add auth, rate-limits, retries, observability hardening, validation, secure secret management.

## 14. Support
If you have questions: check code comments, open an issue (if repository hosting allows), or review agent fallbacks to understand offline behavior.

---
**Tip:** Start offline (deterministic) to verify retrieval and multi-domain logic, then enable LLM for improved fluency once connectivity is stable.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Langfuse account (free tier available at https://cloud.langfuse.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Assigment03
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   LANGFUSE_PUBLIC_KEY=pk-lf-xxx
   LANGFUSE_SECRET_KEY=sk-lf-xxx
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

5. **Get Langfuse API keys**
   - Sign up at https://cloud.langfuse.com
   - Navigate to Settings → API Keys
   - Copy your Public Key and Secret Key

## Usage

### Option 1: Jupyter Notebook (Recommended for Development)

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Open `multi_agent_system.ipynb`**

3. **Run cells in order**:
   - Section 1: Setup & Imports
   - Section 2: Document Loading & Vector Stores
   - Section 3: Agent Definitions
   - Section 4: Orchestrator & Routing
   - Section 5: Testing & Examples
   - Section 6: Langfuse Integration
   - Section 7: Evaluator Agent (Bonus)

### Option 2: Python Script

```python
from src.multi_agent_system import MultiAgentSystem

# Initialize system
system = MultiAgentSystem(rebuild_vector_stores=False)

# Process a query (evaluation is automatic!)
result = system.process_query("What are the vacation leave policies?")

print(f"Intent: {result['intent']}")
print(f"Agent: {result['agent']}")
print(f"Answer: {result['answer']}")

# Evaluation is automatically included
if 'evaluation' in result and result['evaluation']:
    eval_data = result['evaluation']
    print(f"\nAutomatic Evaluation:")
    print(f"  Overall Score: {eval_data['overall_score']}/10")
    print(f"  Relevance: {eval_data['relevance']}/10")
    print(f"  Completeness: {eval_data['completeness']}/10")
    print(f"  Accuracy: {eval_data['accuracy']}/10")
```

### Option 3: Command Line

```bash
# Evaluation is automatic by default
python -m src.multi_agent_system "What are the vacation leave policies?"

# To disable evaluation (not recommended)
python -m src.multi_agent_system "What are the vacation leave policies?" --no-evaluate
```

## Configuration

Key configuration options in `src/config.py`:

- **OPENAI_MODEL**: LLM model (default: "gpt-4")
- **EMBEDDING_MODEL**: Embedding model (default: "text-embedding-3-small")
- **CHUNK_SIZE**: Document chunk size (default: 1000)
- **CHUNK_OVERLAP**: Chunk overlap (default: 200)
- **TOP_K_RETRIEVAL**: Number of documents to retrieve (default: 5)

## Testing

The project includes 25 test queries covering:
- HR queries (benefits, leave, policies, onboarding)
- Tech queries (software, troubleshooting, access)
- Finance queries (expenses, invoices, budgets)
- Edge cases (ambiguous, multi-domain, unrelated)

Run tests:
```python
import json
from src.multi_agent_system import MultiAgentSystem

system = MultiAgentSystem()

with open('tests/test_queries.json', 'r') as f:
    test_data = json.load(f)
    
for query_data in test_data['test_queries']:
    result = system.process_query(query_data['query'])
    print(f"Query: {query_data['query']}")
    print(f"Expected: {query_data['expected_intent']}, Got: {result['intent']}")
    print()
```

## Observability with Langfuse

All operations are automatically traced in Langfuse:

1. **View Traces**: Go to https://cloud.langfuse.com → Traces
2. **Filter by Operation**: 
   - `intent_classification`: Orchestrator classification
   - `hr_agent_query`: HR agent operations
   - `tech_agent_query`: Tech agent operations
   - `finance_agent_query`: Finance agent operations
   - `evaluate_response`: **Automatic** evaluator operations (runs for every query)
3. **View Scores**: **Automatic** evaluation scores appear in the Scores tab for every query:
   - `response_quality`: Overall score (1-10)
   - `relevance`: Relevance score (1-10)
   - `completeness`: Completeness score (1-10)
   - `accuracy`: Accuracy score (1-10)
4. **Debug Issues**: Inspect full execution path, inputs, outputs, and timings

## Technical Decisions

### Why LangChain?
- **Production-Grade**: Battle-tested components reduce bugs and maintenance
- **Standard Framework**: Industry-standard approach ensures maintainability
- **Integration**: Seamless integration with OpenAI, vector stores, and observability tools
- **Extensibility**: Easy to add new agents or modify existing functionality

### Routing Strategy
- **LLM-Based Classification**: GPT-4 provides flexible, accurate intent classification
- **Conditional Routing**: Simple if-else routing based on classified intent
- **Fallback Handling**: Unknown intents receive helpful guidance

### RAG Configuration
- **Chunk Size**: 1000 characters with 200 overlap for context preservation
- **Top-K Retrieval**: 5 documents balance relevance and context window
- **Domain-Specific Stores**: Separate vector stores prevent cross-domain contamination
- **Embedding Model**: text-embedding-3-small for cost-effective, high-quality embeddings

### Observability
- **Langfuse Tracing**: Automatic tracing of all operations
- **Structured Logging**: Clear trace names for easy filtering
- **Automatic Evaluation**: Evaluator runs automatically for every query
- **Score API**: Evaluator integrates with Langfuse for quality metrics (scores logged automatically)
- **Full Visibility**: Complete execution path visible in dashboard, including automatic evaluation scores

## Known Limitations

1. **Intent Classification**: May misclassify ambiguous queries (e.g., queries spanning multiple domains)
2. **Document Coverage**: Answers limited to content in provided documents
3. **Vector Store**: ChromaDB stores data locally; for production, consider cloud vector stores
4. **Cost**: Uses GPT-4 which can be expensive at scale; consider GPT-3.5-turbo for cost savings
5. **Evaluation**: Evaluator uses LLM-based scoring which may have some variance (but runs automatically for every query)

## Future Improvements

- [ ] Add more specialized agents (Legal, Sales, etc.)
- [ ] Implement confidence scores for intent classification
- [ ] Add conversation history and context
- [ ] Implement feedback loop for improving classification
- [ ] Add support for multi-turn conversations
- [ ] Deploy as API service
- [ ] Add caching for common queries
- [ ] Implement A/B testing for different routing strategies

## Dependencies

- `langchain>=0.1.0`: Core framework for chains, retrievers, agents
- `langchain-openai>=0.0.5`: OpenAI integration
- `langfuse>=2.0.0`: Observability and tracing
- `openai>=1.0.0`: OpenAI API client
- `chromadb>=0.4.0`: Vector store
- `tiktoken>=0.5.0`: Token counting
- `python-dotenv>=1.0.0`: Environment variable management
- `jupyter>=1.0.0`: Notebook environment
- `pypdf>=3.0.0`: PDF document loading

## License

This project is for educational purposes as part of the M3 - PI Project Assignment.

## Contact

For questions or issues, please refer to the assignment guidelines or contact the course instructor.

---

**Note**: This system is designed for demonstration purposes. For production use, additional considerations around security, scalability, and error handling should be implemented.

