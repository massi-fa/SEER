import os
import yaml
import json
import re
import time
import logging
import ast
from typing import TypedDict, List, Dict, Any, Optional
import dirtyjson
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from agent_utilities import get_openrouter_llm

# --- Initial Configuration ---
DEBUG = False
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
PROMPT_FILE_PATH = os.path.join(os.path.dirname(__file__), "Prompts", "claim_extraction_prompts.yaml")
MAX_RETRIES = 3

# Configuration of the 3 LLMs for parallel extraction
LLM_MODELS = [
    "z-ai/glm-4.5",
    "moonshotai/kimi-k2",
    "deepseek/deepseek-chat-v3.1",
]

def debug_print(message: str, separator: bool = False):
    """Print debug message if DEBUG flag is enabled."""
    if DEBUG:
        if separator:
            print("=" * 80)
        print(f"[DEBUG] {message}")
        if separator:
            print("=" * 80)

# --- Data Structures Definition ---

class ExtractedClaim(TypedDict):
    """Structure of claims extracted by each LLM."""
    agent_name: str
    agent_description: str
    agent_type: str
    utterance_type: str
    utterance_text: str
    source_context: str

class FinalClaim(TypedDict):
    """Structure of final claims after aggregation."""
    agent_name: str
    agent_description: str
    agent_type: str
    utterance_type: str
    utterance_text: str
    source_context: str

class GraphState(TypedDict):
    """State of the 2-phase workflow."""
    # Input
    article_text: str
    # Phase 1: Parallel extraction from 3 LLMs
    claims_A: Optional[List[ExtractedClaim]]
    claims_B: Optional[List[ExtractedClaim]]
    claims_C: Optional[List[ExtractedClaim]]
    # Phase 2: Aggregation
    final_claims: Optional[List[FinalClaim]]
    # Metadata
    processing_log: List[str]
    error_message: Optional[str]


# --- Workflow Implementation ---

class ClaimExtractorWorkflow:
    """
    2-phase Workflow:
    1. Parallel extraction with 3 independent LLMs
    2. Aggregation of results using a consensus protocol
    """
    def __init__(self, llm_models: List[str] = None):
        if llm_models is None:
            self.llm_models = LLM_MODELS
        else:
            self.llm_models = llm_models
        
        # Initialize the 3 LLMs
        print(self.llm_models)
        self.llm_A = get_openrouter_llm(model_name=self.llm_models[0], temperature=0)
        self.llm_B = get_openrouter_llm(model_name=self.llm_models[1], temperature=0)
        self.llm_C = get_openrouter_llm(model_name=self.llm_models[2], temperature=0)
        
        # LLM for the aggregator
        self.llm_aggregator = get_openrouter_llm(model_name=self.llm_models[2], temperature=0)
        self.fix_aggregator = get_openrouter_llm(model_name=self.llm_models[2], temperature=0.1)        
        self.prompts = self._load_prompts()
        self.chain = self._build_workflow()
        
        debug_print(f"Initialized with LLMs: {llm_models}")


    def normalizeModelName(self,model_id):
        return model_id.replace('.', '_').replace('/', '-')

    def get_exp_name(self):
        model_name_a = self.normalizeModelName(self.llm_models[0])
        model_name_b = self.normalizeModelName(self.llm_models[1])
        model_name_c = self.normalizeModelName(self.llm_models[2])
        aggregator_model_name = model_name_a
        return f"A_{aggregator_model_name}_LLMS{model_name_a}_{model_name_b}_{model_name_c}"

    def _load_prompts(self) -> Dict[str, Dict[str, str]]:
        """Load prompts from the YAML file."""
        try:
            with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            prompt_dict = {}
            for item in data:
                prompt_dict[item['name']] = {
                    'system': item['content'],
                    'user': item.get('user_prompt', '')
                }
            
            required_prompts = ["claim_extraction", "Aggregator", "Aggregator_JSON_Fixer"]
            missing_prompts = [p for p in required_prompts if p not in prompt_dict]
            if missing_prompts:
                raise KeyError(f"Missing essential prompts: {missing_prompts}")
            
            return prompt_dict
        except Exception as e:
            logging.error(f"Error loading prompts: {e}")
            raise

    def _build_workflow(self):
        """Builds the 2-phase graph."""
        from langgraph.graph import StateGraph, START, END
        
        workflow = StateGraph(GraphState)
        
        # Two nodes: parallel extraction and aggregation
        workflow.add_node("parallel_extraction", self._run_parallel_extraction)
        workflow.add_node("aggregation", self._run_aggregation)
        
        # Linear flow
        workflow.add_edge(START, "parallel_extraction")
        workflow.add_conditional_edges(
            "parallel_extraction",
            self._check_extraction_results,
            {"continue": "aggregation", "terminate": END}
        )
        workflow.add_edge("aggregation", END)
        
        return workflow.compile()

    # --- Graph Nodes ---

    def _run_parallel_extraction(self, state: GraphState) -> GraphState:
        """
        Phase 1: Parallel extraction with 3 LLMs.
        Each LLM executes the 'claim_extraction' prompt independently.
        """
        debug_print("ENTERING PHASE 1: Parallel Extraction with 3 LLMs", separator=True)
        state["processing_log"].append("Phase 1: Parallel extraction with 3 LLMs...")
        
        prompt_data = self.prompts["claim_extraction"]
        
        # Prepare the 3 chains
        chains = [
            (ChatPromptTemplate.from_messages([
                ("system", prompt_data['system']),
                ("user", prompt_data['user'])
            ]) | llm, name)
            for llm, name in [
                (self.llm_A, "LLM_A"),
                (self.llm_B, "LLM_B"),
                (self.llm_C, "LLM_C")
            ]
        ]
        
        # Execute in parallel
        results = {"claims_A": [], "claims_B": [], "claims_C": []}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._invoke_chain_with_retry,
                    chain,
                    {"input_article_text": state["article_text"]},
                    expected_key="results",
                    validation_func=self._validate_extracted_claim
                ): (idx, name)
                for idx, (chain, name) in enumerate(chains)
            }
            
            for future in as_completed(futures):
                idx, name = futures[future]
                try:
                    result = future.result()
                    if result:
                        key = f"claims_{chr(65 + idx)}"  # A, B, C
                        results[key] = result
                        state["processing_log"].append(f"{name}: Found {len(result)} claims.")
                        debug_print(f"{name} found {len(result)} claims")
                    else:
                        debug_print(f"{name} failed to extract claims")
                except Exception as e:
                    logging.error(f"Error in {name}: {e}")
                    state["processing_log"].append(f"ERROR in {name}: {e}")
        
        # Update state
        state["claims_A"] = results["claims_A"]
        state["claims_B"] = results["claims_B"]
        state["claims_C"] = results["claims_C"]
        
        total_claims = len(results["claims_A"]) + len(results["claims_B"]) + len(results["claims_C"])
        debug_print(f"Total claims extracted: {total_claims} (A:{len(results['claims_A'])}, B:{len(results['claims_B'])}, C:{len(results['claims_C'])})")
        
        return state

    def _fix_aggregator_json(self, raw_content: str) -> Optional[List[FinalClaim]]:
        """
        Uses the Aggregator_JSON_Fixer prompt to repair malformed JSON.
        
        Args:
            raw_content: The raw content to repair
            
        Returns:
            List of final claims if repair is successful, None otherwise
        """
        debug_print("Attempting JSON repair with Aggregator_JSON_Fixer", separator=True)
        
        prompt_data = self.prompts["Aggregator_JSON_Fixer"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_data['system']),
            ("user", prompt_data['user'])
        ])
        chain = prompt | self.llm_aggregator
        
        try:
            result = chain.invoke({"aggregator_output_text": raw_content})
            fixed_content = result.content
            
            # Attempt parsing of repaired JSON
            parsed = self._parse_json_with_validation(
                fixed_content,
                expected_key="final_claims",
                validation_func=self._validate_extracted_claim
            )
            
            if parsed:
                debug_print(f"JSON repair successful! Recovered {len(parsed)} claims")
                return parsed
            else:
                debug_print("JSON repair failed: parsing returned None")
                return None
                
        except Exception as e:
            logging.error(f"Error during JSON repair: {e}")
            debug_print(f"JSON repair exception: {e}")
            return None

    def _run_aggregation(self, state: GraphState) -> GraphState:
        """
        Phase 2: Aggregation with consensus protocol.
        Uses the 'Aggregator' prompt to consolidate the 3 outputs.
        """
        debug_print("ENTERING PHASE 2: Aggregation with Consensus Protocol", separator=True)
        state["processing_log"].append("Phase 2: Aggregation with consensus protocol...")
        
        prompt_data = self.prompts["Aggregator"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_data['system']),
            ("user", prompt_data['user'])
        ])
        chain = prompt | self.llm_aggregator
        
        # Prepare input for aggregator
        input_vars = {
            "claims_A": json.dumps(state.get("claims_A", []), ensure_ascii=False, indent=2),
            "claims_B": json.dumps(state.get("claims_B", []), ensure_ascii=False, indent=2),
            "claims_C": json.dumps(state.get("claims_C", []), ensure_ascii=False, indent=2)
        }
        
        # Standard parsing attempt
        parsed_data = None
        raw_content = None
        
        try:
            result = chain.invoke(input_vars)
            raw_content = result.content
            parsed_data = self._parse_json_with_validation(
                raw_content,
                expected_key="final_claims",
                validation_func=self._validate_extracted_claim
            )
        except Exception as e:
            logging.warning(f"Error during standard aggregation: {e}")
            state["processing_log"].append(f"Error in standard aggregation: {e}")
        
        # If standard parsing fails, attempt repair
        if parsed_data is None and raw_content:
            state["processing_log"].append("Standard parsing failed. Attempting JSON repair...")
            debug_print("Standard parsing failed, attempting JSON repair...")
            parsed_data = self._fix_aggregator_json(raw_content)
            
            if parsed_data:
                state["processing_log"].append(f"JSON repair successful! Recovered {len(parsed_data)} final claims.")
            else:
                state["processing_log"].append("JSON repair failed.")
        
        if parsed_data:
            state["final_claims"] = parsed_data
            state["processing_log"].append(f"Aggregator: Produced {len(parsed_data)} final claims.")
            debug_print(f"Aggregator produced {len(parsed_data)} final claims")
        else:
            state["final_claims"] = []
            state["processing_log"].append("Aggregator: No final claims produced.")
            debug_print("Aggregator produced no final claims")

        return state

    # --- Conditional Logic ---

    def _check_extraction_results(self, state: GraphState) -> str:
        """Checks if at least one LLM extracted claims."""
        total = len(state.get("claims_A", [])) + len(state.get("claims_B", [])) + len(state.get("claims_C", []))
        if total > 0:
            return "continue"
        else:
            state["processing_log"].append("No claims extracted by any LLM. Terminating workflow.")
            return "terminate"

    # --- Validation Functions ---

    def _validate_extracted_claim(self, claim: dict) -> bool:
        """Validates the structure of an extracted claim."""
        required_fields = ["agent_name", "agent_description", "agent_type", 
                          "utterance_type", "utterance_text", "source_context"]
        
        if not all(field in claim for field in required_fields):
            return False
        
        # Validate minimum content
        if not claim["agent_name"].strip() or not claim["utterance_text"].strip():
            return False
        
        # Validate agent_type
        if claim["agent_type"] not in ["person", "organization"]:
            return False
        
        # Validate utterance_type
        if claim["utterance_type"] not in ["direct", "partially-direct", "indirect"]:
            return False
        
        return True

    # --- Utility Functions ---

    def _invoke_chain_with_retry(self, chain: Runnable, input_vars: Dict, 
                                 attempt_limit=MAX_RETRIES, 
                                 expected_key=None, validation_func=None) -> Optional[Any]:
        """Invokes an LLM chain with retry logic, parsing, and validation."""
        debug_print(f"Invoking LLM chain with {attempt_limit} max attempts")
        for attempt in range(attempt_limit):
            try:
                debug_print(f"LLM attempt {attempt + 1}/{attempt_limit}")
                result = chain.invoke(input_vars)
                content = result.content
                debug_print(f"Raw content length: {len(content)} chars")
                parsed = self._parse_json_with_validation(content, expected_key, validation_func)
                if parsed is not None:
                    debug_print(f"LLM call and parsing successful on attempt {attempt + 1}")
                    return parsed
                else:
                    debug_print(f"Parsing/validation failed on attempt {attempt + 1}")
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/{attempt_limit} failed: {e}")
            time.sleep(1.5)
        logging.error(f"Invocation failed after {attempt_limit} attempts.")
        debug_print(f"All LLM attempts failed after {attempt_limit} tries")
        return None

    def _parse_json_with_validation(self, content: str, expected_key: Optional[str], validation_func=None) -> Optional[Any]:
        """JSON parsing with extraction, repair, and validation."""
        json_str = None
        # 1. Look for markdown block
        match = re.search(r'```json\s*([\s\S]+?)\s*```', content)
        if match:
            json_str = match.group(1)
        else:
            # 2. Fallback: look for raw object/array
            json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
            if json_match:
                json_str = json_match.group(0)

        if not json_str:
            logging.error(f"No JSON block found. Content: {content[:200]}...")
            return None

        # 3. Parsing with dirtyjson fallback
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logging.warning("Standard JSON parsing failed. Attempting with dirtyjson...")
            try:
                data = dirtyjson.loads(json_str)
            except Exception as e:
                logging.error(f"Parsing with dirtyjson failed: {e}")
                return None

        # 4. Extract expected key
        if expected_key is None:
            result = data
        elif expected_key and isinstance(data, dict):
            result = data.get(expected_key)
        else:
            result = None

        # 5. Validation
        if result is not None and isinstance(result, list) and validation_func:
            valid_items = [item for item in result if validation_func(item)]
            return valid_items
        
        if result is not None and isinstance(result, dict) and validation_func:
            return result if validation_func(result) else None
        
        return results

    def visualize(self):
        """
        Visualize the workflow graph.
        """
        # Add imports for visualization
        from IPython.display import display, Image
        return display(Image(self.chain.get_graph().draw_mermaid_png()))

    def run_claim_extraction(self, article_text: str) -> Dict[str, Any]:
        """
        Executes the entire 2-phase claim extraction workflow.
        
        Returns:
            Dict containing:
            - claims: List of final claims
            - processing_log: Operation log
            - status: "success" or "error"
            - pipeline_stats: Pipeline statistics
        """
        if DEBUG:
            print("\n" + "=" * 80)
            print("INIZIO ESTRAZIONE CLAIMS - 2 PHASE WORKFLOW")
            print("Phase 1: Parallel Extraction (3 LLMs) → Phase 2: Aggregation")
            print("=" * 80)
        
        initial_state = GraphState(
            article_text=article_text,
            claims_A=None,
            claims_B=None,
            claims_C=None,
            final_claims=[],
            error_message=None,
            processing_log=[]
        )
        
        debug_print("Invoking workflow chain...")
        final_state = self.chain.invoke(initial_state)
        
        # Raccogli statistiche
        claims_A_count = len(final_state.get('claims_A', []))
        claims_B_count = len(final_state.get('claims_B', []))
        claims_C_count = len(final_state.get('claims_C', []))
        final_count = len(final_state.get('final_claims', []))
        
        total_extracted = claims_A_count + claims_B_count + claims_C_count
        avg_per_llm = total_extracted / 3 if total_extracted > 0 else 0
        
        if DEBUG:
            print("\n" + "=" * 80)
            print("RIEPILOGO FINALE")
            print("=" * 80)
            print(f"Status: {'✓ Successo' if not final_state.get('error_message') else '✗ Errore'}")
            print(f"\nEstrazione Parallela:")
            print(f"  LLM A: {claims_A_count} claims")
            print(f"  LLM B: {claims_B_count} claims")
            print(f"  LLM C: {claims_C_count} claims")
            print(f"  Totale estratto: {total_extracted} claims (media: {avg_per_llm:.1f} per LLM)")
            print(f"\nAggregazione:")
            print(f"  Claims finali: {final_count}")
            if total_extracted > 0:
                retention_rate = (final_count / total_extracted) * 100
                print(f"  Tasso di retention: {retention_rate:.1f}%")
            print("=" * 80 + "\n")

        return {
            "claims": final_state.get("final_claims", []),
            "processing_log": final_state.get("processing_log", []),
            "status": "success" if not final_state.get("error_message") else "error",
            "error_message": final_state.get("error_message"),
            "pipeline_stats": {
                "llm_A_claims": claims_A_count,
                "llm_B_claims": claims_B_count,
                "llm_C_claims": claims_C_count,
                "total_extracted": total_extracted,
                "average_per_llm": avg_per_llm,
                "final_claims": final_count,
                "retention_rate": (final_count / total_extracted * 100) if total_extracted > 0 else 0
            }
        }


# --- Execution Example ---

if __name__ == '__main__':
    # Create the workflow with the 3 configured LLMs
    extractor = ClaimExtractorWorkflow()

    # Sample article
    sample_article = """
    During a press conference in Brussels, European Commission President Ursula von der Leyen announced a new digital strategy.
    "We will invest heavily in secure infrastructure," she stated. The president later explained that this plan aims to
    bolster the EU's digital sovereignty. In his response, industry analyst John Doe said that the minister's promise to
    "solve everything" was overly ambitious. According to Reuters, the plan has been in development for six months.
    """
    
    # Run extraction
    results = extractor.run_claim_extraction(sample_article)
    
    # Print results
    print("\n--- FINAL RESULTS ---")
    print(json.dumps(results["claims"], indent=2, ensure_ascii=False))
    
    print("\n--- PIPELINE STATISTICS ---")
    stats = results["pipeline_stats"]
    print(f"LLM A: {stats['llm_A_claims']} claims")
    print(f"LLM B: {stats['llm_B_claims']} claims")
    print(f"LLM C: {stats['llm_C_claims']} claims")
    print(f"Totale estratto: {stats['total_extracted']} (media: {stats['average_per_llm']:.1f})")
    print(f"Claims finali: {stats['final_claims']} (retention: {stats['retention_rate']:.1f}%)")
    
    print("\n--- PROCESSING LOG ---")
    for log_entry in results["processing_log"]:
        print(log_entry)