from langgraph.graph import StateGraph, START, END
from typing import Dict, List, Any, TypedDict, Optional
from IPython.display import Image, display
from tqdm import tqdm
import json
import os
import sys
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root directory to Python path
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Also add Code subdirectory to Python path
code_dir = os.path.join(current_dir, 'Code')
if code_dir not in sys.path:
    sys.path.append(code_dir)


from semantic_db import SemanticNewsDB
from claim_extraction import ClaimExtractor
from claim_extraction2_0 import ClaimExtractorWorkflow
from entity_information_extraction import EntityInfoExtractor
from knowledge_graph_creator import KnowledgeGraphCreator

class FullPipelineState(TypedDict):
    """State for the full analysis pipeline"""
    query: str
    num_articles: int
    article_ids: List[str]
    articles: List[Dict[str, Any]]
    extracted_claims: List[Dict[str, Any]]
    agents_info: List[Dict[str, Any]]
    summary: Dict[str, Any]  # Added new field for summary
    knowledge_graph: str      # Added new field for the semantic graph


class FullPipeline:
    """
    A comprehensive pipeline that:
    1. Searches for articles based on a query using a semantic database
    2. Extracts claims from the articles
    3. Analyzes agents mentioned in the claims
    """
    
    @staticmethod
    def ensure_spacy_model(model_name: str = "en_core_web_lg") -> bool:
        """
        Ensure spaCy model is downloaded and ready before pipeline initialization.
        This should be called once before creating a FullPipeline instance.
        
        Args:
            model_name: Name of the spaCy model to ensure
            
        Returns:
            True if model is ready, False otherwise
        """
        import subprocess
        import sys
        import spacy
        
        def is_model_installed(name: str) -> bool:
            try:
                spacy.load(name)
                return True
            except OSError:
                return False
        
        if is_model_installed(model_name):
            print(f"✓ spaCy model '{model_name}' already installed and ready")
            return True
        
        print(f"📥 Downloading spaCy model '{model_name}'...")
        print("This is a one-time download (may take a few minutes)")
        
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "spacy", "download", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end="")
            
            process.wait()
            
            if process.returncode == 0:
                print(f"\n✓ Model '{model_name}' downloaded successfully")
                
                # Verify it's loadable
                print(f"📦 Verifying model...")
                nlp = spacy.load(model_name)
                print(f"✓ Model verified and ready to use")
                print(f"  Pipeline components: {nlp.pipe_names}")
                return True
            else:
                print(f"\n❌ Download failed (return code: {process.returncode})")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print(f"Please install manually: python -m spacy download {model_name}")
            return False
    
    def __init__(self, model_claim_extraction: Any = "google/gemini-2.5-flash-lite", model_entity_analysis: str = "google/gemini-2.5-flash-lite", temperature: float = 0, claim_extraction_version: int = 1, parallel: bool = False, max_workers: int = None):
        """
        Initialize the full analysis pipeline
        
        Args:
            model_claim_extraction: Model(s) for claim extraction. Can be a string (V1) or list of strings (V2).
            model_entity_analysis: Model for entity analysis
            temperature: Temperature for LLM
            claim_extraction_version: Version of claim extraction (1 or 2)
            parallel: If True, process articles in parallel; if False, sequential processing
            max_workers: Maximum number of parallel workers (None = auto-detect, respects parallel flag)
            
        Note:
            The spaCy model 'en_core_web_lg' will be automatically downloaded if needed.
        """
        
        # Ensure spaCy model is available BEFORE initializing entity extractor
        print("Checking spaCy model availability...")
        spacy_ready = self.ensure_spacy_model()
        
        if not spacy_ready:
            print("⚠️  WARNING: spaCy model unavailable. Entity linking will operate in limited mode.")

        self.claim_extraction_version = claim_extraction_version
        self.model_claim_extraction = model_claim_extraction
        self.model_entity_analysis = model_entity_analysis
        self.temperature = temperature
        self.parallel = parallel
        self.max_workers = max_workers
        self.semantic_search = SemanticNewsDB()

        if self.claim_extraction_version == 1:
            # If a list is provided for V1, use the first element
            if isinstance(self.model_claim_extraction, list):
                model_name = self.model_claim_extraction[0]
            else:
                model_name = str(self.model_claim_extraction)
                
            self.claim_extractor = ClaimExtractor(model_name=model_name, temperature=self.temperature)
        else:
            # V2 Logic
            if isinstance(self.model_claim_extraction, list):
                # Use provided list of models
                models = self.model_claim_extraction
                # Ensure we have 3 models by repeating if necessary (primitive fallback)
                if len(models) < 3:
                     # Fill remaining slots with the last model provided
                     models += [models[-1]] * (3 - len(models))
            else:
                # If single string provided for V2, use it for all 3 agents (or default fallback)
                # But safer to just use it as the 'default' override if we want consistency
                # However, usually V2 implies an ensemble. 
                # If user passed a single model for V2, we create an ensemble of 3 same models (which is still useful for temperature variation/self-consistency, though currently temp is fixed)
                models = [str(self.model_claim_extraction)] * 3
                
            self.claim_extractor = ClaimExtractorWorkflow(llm_models=models)
            self.model_claim_extraction = self.claim_extractor.get_exp_name()
        
        # Initialize entity extractor (spaCy model already loaded by ensure_spacy_model)
        print("Initializing entity information extraction system...")
        self.entity_information_extraction = EntityInfoExtractor(model_name=model_entity_analysis, temperature=temperature)
        
        # No need to check again - we already know the status from ensure_spacy_model()
        print("✓ Pipeline initialization complete")
        
        self.knowledge_graph_creator = None
        self.workflow = StateGraph(FullPipelineState)
        self.chain = None
        self._build_workflow()

    def _build_workflow(self):
        """
        Build the workflow for the full analysis pipeline
        """
        self.workflow.add_node("search_articles", self._search_articles)
        self.workflow.add_node("process_articles", self._process_articles)
        self.workflow.add_node("create_summary", self._create_summary)
        self.workflow.add_node("create_knowledge_graph", self._create_knowledge_graph)

        self.workflow.add_edge(START, "search_articles")
        self.workflow.add_edge("search_articles", "process_articles")
        self.workflow.add_edge("process_articles", "create_summary")
        #self.workflow.add_edge("create_summary", END)
        self.workflow.add_edge("create_summary", "create_knowledge_graph")
        self.workflow.add_edge("create_knowledge_graph", END)

        self.chain = self.workflow.compile()

    def _search_articles(self, state: FullPipelineState) -> FullPipelineState:
        """
        Search for articles based on the query and update the state.

        Args:
            state: The current state of the pipeline

        Returns:
            The updated state
        """
        print("Searching for articles...")  

        if state.get('article_ids') is not None:
            print(f"Article IDs: {state['article_ids']}")  
            articles = self.semantic_search.get_articles_by_ids(state["article_ids"])
            state['num_articles'] = len(articles)
            print(f"Articles found: {len(articles)}")  # Debug print
        else:
            print(f"Query: {state['query']}") 
            print(f"Number of articles: {state['num_articles']}") 

            articles = self.semantic_search.search(state["query"], state["num_articles"])
        state["articles"] = articles
        return state
    
    def _extract_claims_from_response(self, res_claims: Any, article_id: str, progress_bar) -> List[Dict[str, Any]]:
        """
        Extract claims from the response based on the version being used.
        Handles different output formats between v1 and v2 claim extractors.
        """
        if not isinstance(res_claims, dict):
            progress_bar.write(f"  Warning (Article {article_id}): Response is not a dict. Using empty list.")
            return []
        
        if self.claim_extraction_version == 1:
            # Version 1 uses 'extracted_claims' or 'validated_claims'
            claims = res_claims.get('extracted_claims', res_claims.get('validated_claims', []))
        else:  # version == 2
            # Version 2 can use 'claims', 'results', 'final_claims', or fallback options
            claims = res_claims.get('claims', res_claims.get('results', res_claims.get('final_claims', [])))
            
            # Debug logging for v2 to help troubleshoot
            if not claims:
                available_keys = list(res_claims.keys())
                progress_bar.write(f"  Debug (Article {article_id}): No claims found. Available keys: {available_keys}")
                
                # Log processing information if available
                if 'processing_log' in res_claims:
                    log_entries = res_claims['processing_log']
                    if log_entries:
                        progress_bar.write(f"  Debug (Article {article_id}): Last log entry: {log_entries[-1]}")
                
                if 'error_message' in res_claims and res_claims['error_message']:
                    progress_bar.write(f"  Debug (Article {article_id}): Error message: {res_claims['error_message']}")
        
        return claims if isinstance(claims, list) else []


    def _process_single_article(self, article_idx, article, backup_dir, progress_bar, lock):
        """
        Process a single article: extract claims and analyze agents.
        Thread-safe implementation for parallel processing.
        
        Args:
            article_idx: Index of the article
            article: Article data dictionary
            backup_dir: Directory for backup files
            progress_bar: tqdm progress bar instance
            lock: Threading lock for thread-safe operations
            
        Returns:
            Tuple of (claims_entry, agents_entry)
        """
        original_article_id = article.get("article_id")
        article_id_for_data = original_article_id if original_article_id is not None else f"generated_id_{article_idx}"

        if original_article_id:
            filename_safe_article_id = str(original_article_id).replace(os.sep, "_").replace("..", "_")
            if not filename_safe_article_id:
                filename_safe_article_id = f"empty_id_idx_{article_idx}"
        else:
            timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
            filename_safe_article_id = f"no_id_idx_{article_idx}_{timestamp_str}"
        
        backup_filename = f"article_{filename_safe_article_id}.json"
        backup_path = os.path.join(backup_dir, backup_filename)

        # Thread-safe progress bar write
        def safe_write(msg):
            with lock:
                progress_bar.write(msg)

        loaded_from_backup = False
        
        if os.path.exists(backup_path):
            try:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                if backup_data.get("status") == "success" and \
                   "claims" in backup_data and \
                   "agents_info" in backup_data:
                    safe_write(f"  Article '{article_id_for_data}' (Backup: {backup_filename}): Claims and Agents loaded from backup.")
                    return (
                        {"article_id": article_id_for_data, "claims": backup_data["claims"]},
                        {"article_id": article_id_for_data, "agents_info": backup_data["agents_info"]}
                    )
                else:
                    safe_write(f"  Article '{article_id_for_data}' (Backup: {backup_filename}): Backup invalid or incomplete. Reprocessing.")
            except (json.JSONDecodeError, IOError) as e:
                safe_write(f"  Article '{article_id_for_data}' (Backup: {backup_filename}): Error loading backup ({str(e)}). Reprocessing.")

        safe_write(f"  Article '{article_id_for_data}': Processing with LLM...")

        current_article_claims = []
        current_agents_info = {}
        article_status = "success"
        article_error_msg = None
        
        backup_payload = {
            "article_id": article_id_for_data, 
            "title": article.get("title", ""),
            "body": article.get("body", ""),
            "date": article.get("date", ""),
            "source_link": article.get("source_link", ""),
            "source_info": article.get("source_info", ""),
            "claims": [],
            "agents_info": {},
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "processing"
        }

        try:
            article_body = article.get("body", "")
            if not isinstance(article_body, str):
                safe_write(f"  Warning (Article {article_id_for_data}): Article body is not a string (type: {type(article_body)}). Converting to empty string.")
                article_body = ""

            if not article_body:
                safe_write(f"  Article '{article_id_for_data}': Body is empty. Claim/agent extraction will likely yield empty results.")

            res_claims = self.claim_extractor.run_claim_extraction(article_body)
            extracted_claims_data = self._extract_claims_from_response(res_claims, article_id_for_data, progress_bar)
            

            if not isinstance(extracted_claims_data, list):
                safe_write(f"  Warning (Article {article_id_for_data}): 'extracted_claims' is not a list. Using empty list.")
                extracted_claims_data = []
            
            current_article_claims = extracted_claims_data
            backup_payload["claims"] = current_article_claims
            
            if current_article_claims:
                unique_agents = {}
                for claim_idx, claim in enumerate(current_article_claims):
                    if not isinstance(claim, dict):
                        continue
                    
                    agent_name = claim.get("agent_name")
                    if agent_name and agent_name not in unique_agents:
                        unique_agents[agent_name] = {
                            "agent_type": claim.get("agent_type"),
                            "agent_description": claim.get("agent_description")
                        }
                
                if unique_agents:  # Only if there are agents to process
                    for agent_name, agent_details in unique_agents.items():
                        try:
                            res_entity = self.entity_information_extraction.run(
                                article_text=article_body,
                                raw_name_text=agent_name, 
                                type_of_agent=agent_details["agent_type"], 
                                agent_description=agent_details["agent_description"]
                            )
                            entity_info_data = res_entity.get('entity_info')
                            if not isinstance(entity_info_data, dict):
                                safe_write(f"  Warning (Article {article_id_for_data}, agent '{agent_name}'): 'entity_info' is not a dict. Storing basic info with error.")
                                current_agents_info[agent_name] = {"name": agent_name, "error": "Malformed entity_info from LLM"}
                            else:
                                current_agents_info[agent_name] = entity_info_data
                                
                        except Exception as e_agent:
                            safe_write(f"  Error (Article {article_id_for_data}, agent '{agent_name}'): Failed to analyze agent: {str(e_agent)}")
                            current_agents_info[agent_name] = {"name": agent_name, "error_in_agent_analysis": str(e_agent)}
            
            backup_payload["agents_info"] = current_agents_info
            backup_payload["status"] = "success"
            article_status = "success"

        except Exception as e_article:
            safe_write(f"  Error (Article '{article_id_for_data}'): Failed to process: {str(e_article)}")
            article_status = "error"
            article_error_msg = str(e_article)
            current_article_claims = []
            current_agents_info = {}
            backup_payload["claims"] = []
            backup_payload["agents_info"] = {}
            backup_payload["status"] = "error"
            backup_payload["error"] = article_error_msg
        
        final_claims_entry = {"article_id": article_id_for_data, "claims": current_article_claims}
        final_agents_entry = {"article_id": article_id_for_data, "agents_info": current_agents_info}
        
        if article_status == "error":
            final_claims_entry["error"] = article_error_msg
            final_agents_entry["error"] = article_error_msg
        
        backup_payload["timestamp"] = datetime.datetime.now().isoformat()
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_payload, f, ensure_ascii=False, indent=2)
        except IOError as e_io_backup:
            safe_write(f"  CRITICAL (Article '{article_id_for_data}'): Failed to save backup to '{backup_filename}': {str(e_io_backup)}")
        
        return (final_claims_entry, final_agents_entry)

    def _process_articles(self, state: 'FullPipelineState') -> 'FullPipelineState':
        """
        Process each article individually, extracting claims and analyzing agents.
        Uses ThreadPoolExecutor with max_workers=1 for sequential mode or max_workers>1 for parallel mode.

        Args:
            state: The current state of the pipeline

        Returns:
            The updated state
        """
        mode = "parallel" if self.parallel else "sequential"
        print(f"Processing articles in {mode} mode...")
        
        backup_dir = os.path.join(current_dir, "backup_extractions")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        processed_claims_list = []
        processed_agents_info_list = []
        
        articles = state["articles"]
        
        # Determine number of workers with safer defaults
        if self.parallel:
            if self.max_workers is not None:
                # User-specified limit
                max_workers = max(1, min(self.max_workers, len(articles)))
            else:
                # Conservative auto-detect: limit to 5 for API safety
                # Adjust based on your API provider's rate limits
                conservative_limit = 20
                max_workers = min(conservative_limit, len(articles)) if len(articles) > 0 else 1
        else:
            max_workers = 1  # Sequential mode: single worker
        
        print(f"Using {max_workers} worker(s) for processing")
        
        lock = Lock()
        
        with tqdm(total=len(articles), desc=f"Processing articles ({mode})", unit="article") as progress_bar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._process_single_article,
                        article_idx, article,
                        backup_dir, progress_bar, lock
                    )
                    for article_idx, article in enumerate(articles)
                ]
                
                for future in as_completed(futures):
                    try:
                        claims_entry, agents_entry = future.result()
                        with lock:
                            processed_claims_list.append(claims_entry)
                            processed_agents_info_list.append(agents_entry)
                            progress_bar.update(1)
                    except Exception as e:
                        with lock:
                            progress_bar.write(f"  Error processing article: {str(e)}")
                            progress_bar.update(1)
        
        state["extracted_claims"] = processed_claims_list
        state["agents_info"] = processed_agents_info_list
        return state

    def _create_summary(self, state: FullPipelineState) -> FullPipelineState:
        """
        Create a summary of the analysis and update the state.

        Args:
            state: The current state of the pipeline containing agents information

        Returns:
            The updated state with the summary added

        Raises:
            KeyError: If the 'agents_info' key is missing from state
            TypeError: If the agents information data structure is invalid
        """
        print("Creating summary...")  # Debug print

        condensed_info = []

        for article in state['articles']:
            article_id = article.get('article_id', '')
            claims = [claim for article_claims in state['extracted_claims'] 
                    if article_claims['article_id'] == article_id 
                    for claim in article_claims['claims']]

            agents_info = next((article_agents_info['agents_info'] 
                            for article_agents_info in state['agents_info'] 
                            if article_agents_info['article_id'] == article_id), {})

            condensed_info.append({
                "article_id": article_id,
                "title": article.get('title', ''),
                "body": article.get('body', ''),
                "summary": article.get('summary', ''),
                "date": article.get('date', ''),
                "source_link": article.get('source_link', ''),
                "source_info": article.get("source_info", ""),
                "claims": claims,
                "agents_info": agents_info
            })

        num_claims = sum(len(article['claims']) for article in condensed_info)
        num_agents = sum(len(article['agents_info']) for article in condensed_info)

        summary = {
            "query": state["query"],
            "num_articles": state["num_articles"],
            "num_claims": num_claims,
            "num_agents": num_agents,
            # Informazioni aggiuntive sulla versione e sui modelli
            "claim_extraction_version": self.claim_extraction_version,
            "claim_extraction_model": self.model_claim_extraction,
            "entity_linking_model": self.model_entity_analysis,
            "temperature": self.temperature,
            #"articles": state["articles"],
            #"extracted_claims": state["extracted_claims"],
            #"agents_info": state["agents_info"],
            "summary": condensed_info,
        }

        state["summary"] = summary
        return state

    def _create_knowledge_graph(self, state: FullPipelineState) -> FullPipelineState:
        """
        Create a knowledge graph from the analysis results and update the state.

        Args:
            state: The current state of the pipeline containing agents information

        Returns:
            The updated state with the knowledge graph added

        Raises:
            KeyError: If the 'agents_info' key is missing from state
            TypeError: If the agents information data structure is invalid
        """
        print("Creating knowledge graph...")  # Debug print
        self.knowledge_graph_creator = KnowledgeGraphCreator(data=state["summary"]["summary"])
        ABox = self.knowledge_graph_creator.serialize_abox(format="turtle", destination="./KnowledgeGraph/abox.ttl")
        TBox = self.knowledge_graph_creator.serialize_tbox(format="turtle", destination="./KnowledgeGraph/tbox.ttl")
        FullGraph = self.knowledge_graph_creator.serialize_knowledge_graph(format="turtle", destination="./KnowledgeGraph/kg.ttl")
        state["knowledge_graph"] = FullGraph
        print("Knowledge graph created.")
        return state


    def run(self, query: str = None, num_articles: int = None, article_ids: list = None) -> Dict[str, Any]:
        """
        Run the full analysis pipeline.
        
        Args:
            query: The search query
            num_articles: Number of articles to analyze
            article_ids: Optional list of specific article IDs to analyze
            
        Returns:
            The results of the analysis
        """
        if not self.chain:
            self._build_workflow()
        
        initial_state = {
            "query": query,
            "num_articles": num_articles,
            "article_ids": article_ids,
            "articles": [],
            "extracted_claims": [],
            "agents_info": [],  
            "summary": {},      
            "knowledge_graph": ""
        }
        
        try:
            final_state = self.chain.invoke(initial_state)
            return final_state
        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            # Return partial state for debugging
            return {
                "error": str(e),
                "partial_state": initial_state
            }
    
    def visualize(self):
        """Visualize the workflow graph"""
        if not self.chain:
            self._build_workflow()
        
        return display(Image(self.chain.get_graph().draw_mermaid_png()))
