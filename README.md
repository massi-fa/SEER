# SEER (Semantic Extraction and Enrichment Reasoning)

An advanced framework for the automated analysis of journalistic content, leveraging **AI Agents**, **Large Language Models (LLMs)**, and **Knowledge Graphs**.

## 📖 Project Description

**SEER** is a comprehensive pipeline that automates the extraction and analysis of claims from news articles, identifies the actors involved, classifies their types, and constructs a semantic knowledge graph to represent relationships between entities, claims, and topics.

The system demonstrates its capabilities through analysis of articles on sensitive and politically relevant topics such as climate change, UK immigration, Ukraine war, and US politics. This curated dataset serves as a proof-of-concept for research purposes, but the architecture is designed to be **domain-agnostic and highly extensible**. The pipeline can be easily adapted to analyze news articles from any topic or domain - from healthcare and technology to economics and social issues - simply by providing new article datasets. 

**Modular Architecture & Extensibility**: The system's **modular design** allows researchers and developers to easily modify, improve, or replace individual components without affecting the entire pipeline. Each component (semantic search, claim extraction, entity linking, knowledge graph creation) operates independently with well-defined interfaces. Crucially, the **LangGraph-based workflow** allows for the seamless addition of **new analysis nodes**. Researchers can plug in additional modules—such as **Sentiment Analysis**, **Stance Detection**, or **Political Bias Evaluation**—simply by defining new graph nodes and connecting them to the existing flow, transforming SEER into a multi-perspective media analysis tool.

**LangGraph Implementation Benefits**: Built on **LangGraph**, the system leverages a state-machine architecture that provides exceptional advantages: **transparent workflow orchestration**, **automatic state persistence**, **error recovery mechanisms**, and **easy debugging through visual graph representation**. This makes the pipeline both robust for production use and transparent for research purposes, allowing researchers to precisely track data flow and implement complex conditional logic with full observability.

The combination of modular design and LangGraph implementation makes this a versatile tool for media analysis, fact-checking, and discourse studies across multiple domains and languages.

## 🧠 Research Context

This project represents an innovative convergence of **Semantic Web technologies**, **AI agents**, and **information retrieval** applied to journalism and media analysis. Built at the intersection of multiple cutting-edge research domains:

**🌐 Semantic Web & Knowledge Graphs**: Leveraging RDF ontologies and linked data principles to create machine-readable representations of journalistic discourse, enabling automated reasoning and knowledge discovery across large-scale news corpora.

**🤖 AI Agents & LLM Orchestration**: Implementing intelligent agents with LangGraph state machines that coordinate complex workflows, from semantic search to claim extraction, demonstrating how AI systems can autonomously process and understand media content.

**🔍 Advanced Information Retrieval**: Combining semantic embeddings, vector databases, and neural information retrieval techniques to enable context-aware search and content discovery beyond traditional keyword-based approaches.

**📊 Computational Journalism**: Bridging the gap between AI research and practical journalism applications, creating tools for automated fact-checking, bias detection, and discourse analysis that can scale to real-world media monitoring scenarios.

## 🏗️ System Architecture

The project implements a modular architecture based on **LangGraph** that orchestrates different specialized components:

```
📰 Articles → 🔍 Semantic Search → 📄 Claim Extraction → 👥 Agent Analysis → 📊 KG Creation
```

**Claim Extraction Versions:**
- **Version 1.0** (`claim_extraction.py`): Single LLM extraction with retry logic
- **Version 2.0** (`claim_extraction2_0.py`): Advanced 2-phase workflow with parallel extraction from 3 LLMs and consensus-based aggregation for improved accuracy and reliability

## 📂 Repository Structure

```
SEER/
├── 📁 Code/                          # Main pipeline code
│   ├── semantic_db.py                # Semantic database for article search
│   ├── claim_extraction.py           # Claim extraction v1.0 (single LLM)
│   ├── claim_extraction2_0.py        # Claim extraction v2.0 (parallel + aggregation)
│   ├── entity_information_extraction.py # Entity information extraction
│   ├── knowledge_graph_creator.py    # Knowledge graph creation
│   ├── PersonInfoExtractor.py        # Person information extraction
│   ├── OrganizationInfoExtractor.py  # Organization information extraction
│   ├── agent_utilities.py            # Agent utilities
│   ├── Ontology/                     # Owl Ontology and Schema mapping
│   └── Prompts/                      # YAML prompt templates
├── 📁 GoldStandards/                 # Manually annotated datasets
│   ├── ClaimExtraction/              # Gold Standard for Claim Extraction
│   └── EntityLinking/                # Gold Standard for Entity Linking
├── 📁 KnowledgeGraph/                # Generated Knowledge Graph
│   ├── abox.ttl                      # ABox (instances)
│   ├── tbox.ttl                      # TBox (ontological schema)
│   └── kg.ttl                        # Complete Knowledge Graph
├── 📁 SemanticArticlesDB/            # Articles database
│   └── news_semantic_db.json         # Serialized semantic database
├── 📁 backup_extractions/            # Automatic processing backups
├── FullPipeline.py                   # Main pipeline orchestration
├── run.py                            # CLI entry point for the pipeline
├── Run_FullPipeline.ipynb           # Notebook to run the pipeline
├── requirements.txt                  # Python dependencies
└── README.md                        # Documentation
```

## 🛠️ Technology Stack

### Core Frameworks and Libraries
- **LangGraph**: Workflow orchestration and state management
- **LangChain**: Integration with Large Language Models
- **RDFLib**: Knowledge Graph manipulation and serialization
- **spaCy**: Named Entity Recognition and linguistic processing

### Machine Learning and AI
- **Sentence Transformers**: Semantic embeddings for textual similarity
- **FAISS**: Efficient vector search
- **OpenRouter/DeepSeek**: Large Language Models for claim extraction

### Database and Storage
- **JSON**: Article and metadata persistence
- **Turtle (TTL)**: RDF Knowledge Graph serialization
- **FAISS Index**: Vector indexing for semantic search

### Data Processing
- **NLTK**: Text tokenization and preprocessing
- **ftfy**: Encoding correction and text normalization
- **PyTorch**: Backend for embedding models

## ⚙️ Main Components

### 1. 🔍 Semantic Database (`semantic_db.py`)

The **SemanticNewsDB** implements a semantic database for intelligent article search with two main access modes:

**Article Retrieval Modes:**

#### a) **Semantic Search by Query**
- **Text query**: Search articles through semantic similarity
- **Multi-modal support**: Queries on titles, body, or combined content
- **Intelligent ranking**: Results ordered by semantic relevance
- **Advanced filtering**: Configurable Top-K results

#### b) **Direct Retrieval by ID**
- **Specific ID list**: Direct access through article identifiers
- **Batch processing**: Efficient handling of multiple lists
- **Automatic validation**: Existence verification and missing ID reporting
- **Order preservation**: Maintaining requested ID sequence

**Core Functionality:**
- **Automatic loading** of articles from JSON structures organized by topic
- **Semantic embedding creation** using Sentence Transformers models
- **FAISS indexing** for efficient vector search
- **Persistence** and database loading for reuse
- **Complete metadata management** (title, body, summary, source, date)

**Technical Features:**
- Automatic GPU/CPU support for acceleration
- Advanced preprocessing with encoding correction (ftfy)
- Multi-modal search (title, body, combined)
- Fallback handling for missing components
- Intelligent caching for optimal performance

### 2. 📄 Claim Extraction

#### Version 1.0 (`claim_extraction.py`)

The **ClaimExtractor** uses a single LLM to identify and structure claims:

**Extraction Process:**
1. **Text analysis** with specialized prompt engineering
2. **Structured extraction** of claims with metadata:
   - `agent_name`: Who makes the claim
   - `utterance_type`: Type (direct/indirect/partially-direct)
   - `utterance_text`: Content of the claim
   - `source_context`: Original reference text
   - `agent_type`: Agent classification (person/organization)
   - `agent_description`: Contextual description of the agent

**Robustness:**
- Retry system with configurable attempt limit
- Resilient JSON parsing with regex fallback
- Error handling and detailed logging
- LangGraph workflow for traceability

#### Version 2.0 (`claim_extraction2_0.py`) - **Advanced Multi-LLM Workflow**

The **ClaimExtractorWorkflow** implements a sophisticated 2-phase architecture:

**Phase 1: Parallel Extraction**
- **3 independent LLMs** extract claims simultaneously
- Default models: `z-ai/glm-4.5`, `moonshotai/kimi-k2`, `deepseek/deepseek-chat-v3.1`
- Each LLM provides independent analysis
- ThreadPoolExecutor for concurrent execution

**Phase 2: Consensus Aggregation**
- **Aggregator LLM** consolidates results using specialized prompt
- Identifies duplicate claims across extractions
- Resolves conflicts through consensus protocol
- Automatic JSON repair system with fallback parsing

**Advanced Features:**
- **Parallel processing** with configurable LLM models
- **Quality validation** with structured claim verification
- **Robust error handling** with dirty JSON parsing
- **Processing statistics** (retention rate, per-LLM metrics)
- **Graph visualization** support

**Output Structure:**
```python
{
  "claims": [...],           # Final aggregated claims
  "processing_log": [...],   # Detailed execution log
  "status": "success",       # Processing status
  "pipeline_stats": {        # Performance metrics
    "llm_A_claims": N,
    "total_extracted": M,
    "final_claims": K,
    "retention_rate": X%
  }
}
```

### 3. 👥 Entity Information Extraction (`entity_information_extraction.py`)

The **EntityInfoExtractor** implements a multi-stage system for agent information enrichment:

**Processing Workflow:**
```
Direct Search → NER Search → LLM Validation → Taxonomic Classification
```

**Main Phases:**

#### a) **Direct Wikidata Search**
- Immediate search of agent name on Wikidata
- Support for people and organizations
- Automatic extraction of structured metadata

#### b) **Named Entity Recognition (NER)**
- Use of spaCy (en_core_web_lg) for entity identification
- **Thread-safe model loading** with singleton pattern
- **Automatic model download** if not available
- Automatic extraction of PERSON and ORG from context
- Secondary search on Wikidata with identified entities

#### c) **LLM Validation**
- **Single entity**: Direct validation with specialized prompt
- **Multiple entities**: Best selection through LLM reasoning
- Contextual analysis with original article
- Structured output with explicit reasoning

#### d) **Taxonomic Classification**

**People (NewsSpeakerType)**

| Type | Description |
| :--- | :--- |
| `Politician` | Elected or campaigning political figures |
| `PublicOfficial` | Non-elected government/state officials |
| `Expert` | Recognized specialists or professionals |
| `BusinessRepresentative` | Private sector representatives |
| `UnionRepresentative` | Labor union representatives |
| `Journalist` | Media professionals |
| `Activist` | Individuals involved in organized activism |
| `Celebrity` | Entertainment, sports, or public media figures |
| `OrdinaryCitizen` | Quoted people in a personal capacity |
| `AnonymousSource` | Speakers not identified by name |
| `Spokesperson` | Official communicators |
| `Other` | Fallback category |

**Organizations (OrganizationalSpeakerType)**

| Type | Description |
| :--- | :--- |
| `Airline` | Transports passengers or cargo by air |
| `Consortium` | Group of entities collaborating on a common goal |
| `Cooperative` | Business owned and run by its members |
| `Corporation` | For-profit business entity |
| `EducationalOrganization` | Institution providing education (schools, universities) |
| `FundingScheme` | Organization providing financial resources |
| `GovernmentOrganization` | Part of a national, regional, or local government |
| `LibrarySystem` | Manages collections of knowledge resources |
| `LocalBusiness` | Small business serving a local community |
| `MedicalOrganization` | Provides healthcare services (hospitals, clinics) |
| `NGO` | Non-profit, independent advocacy group |
| `NewsMediaOrganization` | Gathers, produces, and distributes news |
| `OnlineBusiness` | Company primarily operating via the internet |
| `PerformingGroup` | Group of artists who perform (theater, orchestra) |
| `PoliticalParty` | Organization competing in elections |
| `Project` | Temporary effort to create a unique product/service |
| `ResearchOrganization` | Institution focused on conducting research |
| `SearchRescueOrganization` | Finds and assists people in distress |
| `SportsOrganization` | Organizes, promotes, or regulates sports |
| `WorkersUnion` | Organization representing workers' interests |
| `Other` | Organization not fitting other categories |

- Automatic classification with confidence scoring
- Fallback system for invalid classifications

### 4. 🌐 Knowledge Graph Creator (`knowledge_graph_creator.py`)

The **KnowledgeGraphCreator** generates a semantic knowledge graph compliant with Semantic Web standards:

**RDF Architecture:**
- **TBox**: Ontological schema with classes and properties
- **ABox**: Concrete instances of articles, actors, and claims
- **Namespaces**: NCO (News Classification Ontology), Schema.org, Time Ontology

**Main Entities:**
- **NewsItem**: Articles with temporal metadata and sources
- **Agent**: People and organizations with Schema.org classifications
- **Claim**: Claims with context and type
- **AgentComponent**: Agent roles in specific claims
- **Utterance**: Linguistic representation of claims
- **Topic**: Thematic classification (climate_change, uk_immigration, ukraine_war, us_politics)

**Advanced Features:**
- **Consistent URIs** with MD5 hashing for uniqueness
- **Wikidata linking** for identified entities
- **Duplicate management** with internal mapping
- **Multiple serialization** (Turtle, RDF/XML, JSON-LD)
- **Temporal metadata** with W3C Time Ontology

### 5. 🔄 Full Pipeline (`FullPipeline.py`)

The **FullPipeline** orchestrates the entire process through LangGraph State Machine:

**Workflow States:**
```python
class FullPipelineState(TypedDict):
    query: str                          # Search query
    num_articles: int                   # Number of articles to process
    article_ids: List[str]              # Specific IDs (optional)
    articles: List[Dict[str, Any]]      # Retrieved articles
    extracted_claims: List[Dict[str, Any]]  # Extracted claims
    agents_info: List[Dict[str, Any]]   # Agent information
    summary: Dict[str, Any]             # Analysis summary
    knowledge_graph: str                # Serialized knowledge graph
```

**Execution Flow:**
1. **search_articles**: Semantic search or retrieval by ID
2. **process_articles**: Parallel or sequential processing with automatic backup
3. **create_summary**: Result aggregation and statistics
4. **create_knowledge_graph**: Final KG generation

**Processing Modes:**
- **Sequential Mode** (`parallel=False`): Articles processed one at a time
- **Parallel Mode** (`parallel=True`): Concurrent processing with ThreadPoolExecutor
- **Configurable workers**: Control parallelism with `max_workers` parameter

**Robustness Management:**
- **Backup system** for each processed article in `backup_extractions/`
- **Automatic recovery** from previous processing (resumes from checkpoints)
- **Thread-safe operations** with locks for concurrent processing
- **Error handling** for individual articles without blocking the pipeline
- **Progress tracking** with detailed progress bars (tqdm)
- **Structured logging** for debugging and monitoring

## 🚀 Setup and Installation

### Prerequisites
- Python 3.11
- spaCy model: `en_core_web_lg` (automatically downloaded on first run)
- OpenRouter API access (for LLM)
- Optional: CUDA-compatible GPU for faster embeddings

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/massi-fa/ClaimExtractionAgent.git
cd ClaimExtractionAgent
```

2. **Install dependencies:**

   The `run.py` script automatically handles the environment setup (virtual environment creation and dependency installation) on the first run.
   
   Alternatively, you can install them manually:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**

   The `run.py` script includes an **interactive configuration wizard**. If the `.env` file is missing, the script will automatically prompt you to select your preferred LLM provider (**OpenRouter**, Google, or OpenAI) and enter the API key.

   Alternatively, you can manually create a `.env` file in the root directory:
   ```bash
   # Example .env content (choose one)
   OPENROUTER_API_KEY=your_api_key_here
   # GOOGLE_API_KEY=your_google_key
   # OPENAI_API_KEY=your_openai_key
   ```

**Note:** The spaCy model `en_core_web_lg` will be automatically downloaded on first pipeline run if not available.

## 📊 Usage

### 🖥️ Command Line (CLI)

The easiest way to run the pipeline is using the `run.py` script.

#### 🔧 CLI Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--query` | str | - | TOPIC mode: Semantic search query (e.g., "climate change"). Mutually exclusive with `--ids`. |
| `--ids` | list | - | DIRECT mode: List of specific article IDs. Mutually exclusive with `--query`. |
| `--num` | int | 5 | Number of articles to retrieve when using `--query`. |
| `--model-claims` | str(s) | `google/gemini-2.5-flash-lite` | LLM model(s) for claim extraction. Pass multiple strings for V2 consensus. |
| `--model-entities` | str | `google/gemini-2.5-flash-lite` | LLM model for entity analysis and linking. |
| `--v2` | flag | False | Enable V2.0 extraction (Multi-LLM Consensus). |
| `--parallel` | flag | False | Enable parallel article processing. |
| `--workers` | int | Auto | Max concurrent workers for parallel processing. |
| `--temp` | float | 0.0 | LLM temperature (0.0 = deterministic). |
| `--output` | str | `summary_pipeline.json` | Path for the final summary output file. |

#### Examples

**1. Semantic Search (Standard Mode)**
Search for articles on a topic and extract claims.
- Without `--num`, defaults to extracting from **5 articles**.
```bash
python run.py --query "climate change policies" 
```

**2. High-Accuracy Mode (V2.0 + Parallel Processing)**
Use the advanced multi-LLM consensus system with parallel execution.
You can specify the 3 distinct LLMs to use for the ensemble.
```bash
python run.py --query "ukraine war" --v2 --parallel --model-claims "z-ai/glm-4.5" "moonshotai/kimi-k2" "deepseek/deepseek-chat-v3.1"
```

**3. Process Specific Articles**
Run analysis on specific article IDs (bypass search).
```bash
python run.py --ids "7570366282_climate_change" "5821571678_uk_immigration"
```

**4. Custom Models**
Specify different LLMs for claim extraction and entity analysis.
```bash
python run.py --query "US politics" --model-claims "openai/gpt-4o" --model-entities "anthropic/claude-3-5-sonnet"
```

### 💻 Python API

You can also use the pipeline programmatically in your Python scripts.

```python
from FullPipeline import FullPipeline

# Initialize the pipeline with default settings (v1.0, sequential)
pipeline = FullPipeline()

# Initialize with advanced settings
pipeline = FullPipeline(
    model_claim_extraction="google/gemini-2.5-flash-lite",  # LLM for claims
    model_entity_analysis="google/gemini-2.5-flash-lite",   # LLM for entities
    temperature=0,                    # Deterministic output
    claim_extraction_version=2,       # Use v2.0 (multi-LLM)
    parallel=True,                    # Enable parallel processing
    max_workers=5                     # Limit concurrent workers
)

# MODE 1: Semantic query analysis
results = pipeline.run(
    query="climate change policy", 
    num_articles=10
)

# MODE 2: Specific article analysis by ID
results = pipeline.run(
    article_ids=[
        '7570366282_climate_change', 
        '7570396554_climate_change',
        '7571453924_climate_change'
    ]
)

# MODE 3: Combination of both (query with limit + specific IDs)
# Note: If both are provided, article_ids takes precedence
results = pipeline.run(
    query="ukraine war",
    num_articles=5,
    article_ids=['7570019533_ukraine_war', '7570244481_ukraine_war']
)
```

### Accessing Results

```python
# Analyzed articles
articles = results['articles']

# Extracted claims
claims = results['extracted_claims']

# Agent information
agents = results['agents_info']

# Complete summary (includes metadata)
summary = results['summary']
# Summary contains:
# - query, num_articles, num_claims, num_agents
# - claim_extraction_version, claim_extraction_model
# - entity_linking_model, temperature
# - condensed article data with claims and agents

# Knowledge Graph (Turtle format)
kg_ttl = results['knowledge_graph']
```

### Interactive Notebook

Use `Run_FullPipeline.ipynb` for a complete interactive experience with visualizations and step-by-step analysis.

## 📈 System Output

### Structured Claims
```json
{
  "agent_name": "European Commission",
  "utterance_type": "direct",
  "utterance_text": "We need immediate action on climate change",
  "source_context": "The European Commission stated that...",
  "agent_type": "organization",
  "agent_description": "EU executive body"
}
```

### Agent Information
```json
{
  "name": "European Commission",
  "type_of_agent": "organization",
  "agent_description": "EU executive body",
  "wikidata_info": {
    "wikidata_id": "Q8880",
    "label": "European Commission",
    "description": "executive branch of the European Union"
  },
  "taxonomy_info": {
    "classification": "GovernmentOrganization",
    "confidence": "HIGH",
    "explanation": "Classified based on..."
  },
  "match_evaluation": "YES",
  "extraction_history": [...]  # Detailed processing log
}
```

### Knowledge Graph
- **Format**: RDF Turtle
- **Entities**: 1000+ for typical analysis
- **Relations**: hasAgentComponent, hasClaim, concernsTopic
- **Integration**: Wikidata linking for known entities

## 🎯 Use Cases

### Academic Research
- Political-media discourse analysis
- Journalistic bias and framing studies
- Public actor and position mapping

### Media Monitoring
- Claim tracking on sensitive topics
- Trend and narrative identification
- Credibility and sourcing analysis

### Fact-Checking
- Verifiable claim extraction
- Source and attribution mapping
- Automatic verification pipeline support

## 🔧 Advanced Configuration

### Pipeline Configuration
```python
# Version 1.0: Single LLM extraction
pipeline = FullPipeline(
    claim_extraction_version=1,
    model_claim_extraction="google/gemini-2.5-flash-lite",
    temperature=0
)

# Version 2.0: Multi-LLM with custom models
from Code.claim_extraction2_0 import ClaimExtractorWorkflow

extractor = ClaimExtractorWorkflow(
    llm_models=[
        "z-ai/glm-4.5",
        "moonshotai/kimi-k2",
        "deepseek/deepseek-chat-v3.1"
    ]
)
results = extractor.run_claim_extraction(article_text)

# Parallel vs Sequential Processing
pipeline_parallel = FullPipeline(
    parallel=True,      # Enable parallel processing
    max_workers=10      # 10 concurrent articles
)

pipeline_sequential = FullPipeline(
    parallel=False      # Process one article at a time
)
```

### Semantic Database
```python
# Use different embedding model
db = SemanticNewsDB(
    model_name="all-mpnet-base-v2",
    data_dir="custom_articles/"
)

# SEMANTIC SEARCH: Text query with similarity
results = db.search(
    query="immigration policy", 
    top_k=20, 
    search_type="title"  # "body", "combined"
)

# DIRECT RETRIEVAL: List of specific IDs
specific_articles = db.get_articles_by_ids([
    "7570366282_climate_change",
    "5821571678_uk_immigration", 
    "7572075644_us_politics"
])

# ADVANCED SEARCH: Parameter combination
climate_articles = db.search(
    query="global warming effects",
    top_k=15,
    search_type="combined"  # Search on title + body
)

# ERROR HANDLING: Retrieval with validation
article_ids = ["valid_id_123", "invalid_id_456", "another_valid_789"]
found_articles = db.get_articles_by_ids(article_ids)
# Automatically reports missing IDs and returns only found ones
```

### Knowledge Graph
```python
# Customize base URI
kg = KnowledgeGraphCreator(
    base_uri="http://yourorganization.org/kg/",
    base_tbox_file="custom_ontology.ttl"
)

# Multiple format serialization
kg.serialize_knowledge_graph(format="json-ld", destination="kg.jsonld")
```

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. This project is intended for research purposes, and we encourage the community to build upon this work.

## 📄 Citation

If you use this code or dataset in your research, please cite this repository and the associated research work.

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [News Classification Ontology (NCO)](http://data.open.ac.uk/ontology/newsclassification)
- [Schema.org Vocabulary](https://schema.org/)
- [W3C Time Ontology](https://www.w3.org/TR/owl-time/)

## 📜 License

This project is released under MIT license. See `LICENSE` for details.

---

For questions, support, or collaborations, open an issue or contact the development team.

## 🏅 Gold Standard

To ensure robust and multi-faceted evaluation, we constructed two distinct gold standard datasets tailored to specific tasks: one for Claim Extraction and another for Agent Entity Linking.

### 1. Claim Extraction Gold Standard

- **File:** `GoldStandards/ClaimExtraction/NewGS.xlsx`
- **Contents:** 93 news articles from reputable English-language outlets, manually annotated by five human annotators, resulting in 1,052 distinct claims.
- **Statistics:**

| Topic           | No. Articles | Total Claims | Mean Claims/Article | Std. Dev. |
|-----------------|--------------|--------------|---------------------|-----------|
| Climate Change  | 21           | 189          | 9.00                | 6.50      |
| UK Immigration  | 24           | 221          | 9.21                | 4.29      |
| War in Ukraine  | 24           | 367          | 15.29               | 5.56      |
| US Politics     | 24           | 275          | 11.46               | 5.30      |

- **Schema:** Each annotated claim includes:
  - `Title`: Title of the source news article
  - `Body`: Full text of the source article
  - `Agent`: Name of the person or organization making the claim
  - `TypeOfAgent`: "person" or "organization"
  - `Utterance`: Exact textual content of the claim
  - `TypeOfUtterance`: direct, indirect, or partially-direct
  - `SourceText`: Original sentence(s) containing the claim

This balanced dataset covers four contemporary topics and exhibits significant variation in claim density, ensuring the evaluation is not biased toward a single narrative style.

### 2. Entity Linking Gold Standard

- **File:** `GoldStandards/EntityLinking/final_gold_standard.xlsx`
- **Contents:** 90 unique agent instances sampled from claims (78 persons, 12 organizations), annotated for Wikidata linking.
- **Statistics:**
  - **Total annotated entities:** 90
  - **Persons:** 78
  - **Organizations:** 12
  - **Entities with Wikidata ID:** 45
  - **Entities without Wikidata ID:** 45

- **Distribution by challenge category:**

| Challenge Category         | Description                                                        | Count |
|---------------------------|--------------------------------------------------------------------|-------|
| **Cases with valid Wikidata link (45 total)** |                                                                    |       |
| wikidata_1_result         | Single, correct candidate found via name lookup                    | 23    |
| wikidata_gt1_result       | Correct entity among multiple candidates, requiring disambiguation  | 22    |
| **Cases with no valid Wikidata link (45 total)** |                                                            |       |
| no_wikidata_0_result      | No candidates found for a non-linkable agent                       | 19    |
| no_wikidata_1_result      | Single incorrect candidate found, must be rejected                 | 10    |
| no_wikidata_gt1_result    | Multiple incorrect candidates found, all must be rejected          | 16    |

- **Schema:** Each instance includes:
  - `agent_name`: Name of the agent as extracted from the claim
  - `agent_description`: Contextual description of the agent
  - `agent_type`: "person" or "organization"
  - `wikidata_id`: Ground truth Wikidata ID (NULL if no correct link exists)
  - `gs_group`: Category defining the specific linking challenge scenario
  - `title`: Title of the source news article
  - `body`: Full text of the source news article

This dataset is perfectly balanced between linkable and non-linkable cases, and is structured into distinct challenge categories to test both precision and the ability to reject false positives.

These gold standards enable both quantitative and qualitative evaluation of the system's claim extraction and entity linking capabilities, providing reliable benchmarks for measuring precision, recall, and robustness.
