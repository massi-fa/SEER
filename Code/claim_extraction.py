from typing import TypedDict, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import nltk
import json
import re
import time
import os
import yaml

from agent_utilities import get_openrouter_llm, load_prompt_by_name

ATTEMPT_LIMIT = 2

# Define the state type
class ClaimExtractionState(TypedDict):
    """State for the claim extraction process."""
    news_article: str  # The original text to extract claims from
    extracted_claims: List[Dict[str, Any]]  # List of claims extracted from the input text
    validated_claims: List[Dict[str, Any]]  # List of validated claims


class ClaimExtractor:
    """
    A class for extracting and validating claims from news articles.
    """
    
    def __init__(self, model_name = "openai/gpt-oss-20b", temperature = 0): # , model_name=None, temperature=0):
        """
        Initialize the claim extractor.
        
        Args:
            model_name: Optional name of the LLM model to use
            temperature: Temperature setting for the LLM
        """
        # Pass parameters to the get_openrouter_llm function
        self.temperature = temperature
        self.llm = get_openrouter_llm( model_name= model_name, temperature= self.temperature)
        
        # Initialize workflow variables before building it
        self.workflow = None
        self.chain = None
        
        # Load prompts from YAML file
        self.prompts = self._load_prompts()

        self.build_workflow()
        
        # Download the punkt tokenizer only once during initialization
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        
    
    def _load_prompts(self) -> Dict[str, Dict[str, str]]:
        """Load prompts from the YAML file."""
        prompt_file_path = os.path.join(os.path.dirname(__file__), "Prompts", "claim_extraction_prompts.yaml")
        try:
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            prompt_dict = {}
            for item in data:
                prompt_dict[item['name']] = {
                    'system': item['content'],
                    'user': item.get('user_prompt', '')
                }
            
            return prompt_dict
        except Exception as e:
            print(f"Error loading prompts: {e}")
            raise

    def build_workflow(self):
        """
        Build a workflow for the claim extraction process.
        
        Returns:
            The compiled workflow chain
        """
        # Import here to avoid circular import issues
        from langgraph.graph import StateGraph, START, END
        
        # Create a new state graph
        workflow = StateGraph(ClaimExtractionState)
        
        # Add nodes for each step in the process
        workflow.add_node("extract_claims", self.extract_claims)
        
        # Add edges to connect the nodes
        workflow.add_edge(START, "extract_claims")
        workflow.add_edge("extract_claims", END)
        
        # Compile the workflow
        self.workflow = workflow
        self.chain = workflow.compile()
        
        return self.chain

    def extract_claims(self, state: ClaimExtractionState) -> ClaimExtractionState:
        """
        Extract claims from a news article.
        
        Args:
            state: The current state containing the news article
            
        Returns:
            Updated state with extracted claims
        """
        # Load prompt from YAML file
        prompt_data = self.prompts["claim_extraction"]
        
        # Create a ChatPromptTemplate using the documented approach
        prompt_template = ChatPromptTemplate([
            ("system", prompt_data['system']),
            ("user", prompt_data['user'])
        ])
        
        attempt_count = 0
        last_error = None
        content = None
        
        while attempt_count < ATTEMPT_LIMIT:
            try:
                attempt_count += 1
                #print(f"Attempt {attempt_count} of {ATTEMPT_LIMIT} to extract claims...")
                
                # Invoke the template with the news article
                formatted_messages = prompt_template.invoke({"input_article_text": state["news_article"]})
                
                # Invoke the model with formatted messages
                result = self.llm.invoke(formatted_messages)
            
                # Extract the content from the result
                if hasattr(result, "content"):
                    content = result.content
                else:
                    content = str(result)
                
                # Parse the JSON response to extract structured claims
                claims = self._parse_claims_from_response(content)
                
                # If we obtained claims, break the loop
                if claims:
                    #print(f"Extraction successful on attempt {attempt_count}.")
                    break
                else:
                    print(f"No claims found on attempt {attempt_count}.")
                    if attempt_count == ATTEMPT_LIMIT:
                        print("Maximum number of attempts reached. No claims extracted.")
                
            except Exception as e:
                last_error = e
                print(f"Error during attempt {attempt_count}: {str(e)}")
                if attempt_count == ATTEMPT_LIMIT:
                    print(f"Maximum number of attempts reached ({ATTEMPT_LIMIT}). Last error: {str(e)}")
                time.sleep(1)
        
        # Create a copy of the state and update it
        updated_state = state.copy()
        updated_state["extracted_claims"] = claims if 'claims' in locals() and claims else []
        
        # Add attempt information to the state
        updated_state["extraction_attempts"] = attempt_count
        if last_error:
            updated_state["extraction_error"] = str(last_error)
        
        return updated_state

    
    def _parse_claims_from_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse claims from the LLM response.
        
        Args:
            content: The response content from the LLM
            
        Returns:
            List of extracted claims
        """
        claims = []
        try:
            # First try to parse the entire response as JSON
            json_objects = json.loads(content)
            if isinstance(json_objects, dict) and "results" in json_objects:
                # Extract claims from the results array
                for claim in json_objects["results"]:
                    # Convert to the format expected by the rest of the application
                    formatted_claim = {
                        "agent_name": claim["agent_name"],
                        "utterance_type": claim["utterance_type"],
                        "utterance_text": claim["utterance_text"],
                        "source_context": claim["source_context"]
                    }

                    if "agent_type" in claim:
                        formatted_claim["agent_type"] = claim["agent_type"]

                    if "agent_description" in claim:
                        formatted_claim["agent_description"] = claim["agent_description"]
                    
                    claims.append(formatted_claim)
            elif isinstance(json_objects, list):
                claims = json_objects
        except json.JSONDecodeError:
            # If that fails, try to extract JSON objects using regex
            # First try to find the entire results array
            results_match = re.search(r'{\s*"results"\s*:\s*(\[.*?\])\s*}', content, re.DOTALL)
            if results_match:
                try:
                    results_array = json.loads(results_match.group(1))
                    for claim in results_array:
                        formatted_claim = {
                            "agent_name": claim["agent_name"],
                            "utterance_type": claim["utterance_type"],
                            "utterance_text": claim["utterance_text"],
                            "source_context": claim["source_context"]
                        }
                        if "agent_type" in claim:
                            formatted_claim["agent_type"] = claim["agent_type"]
                        
                        if "agent_description" in claim:
                            formatted_claim["agent_description"] = claim["agent_description"]
                        
                        claims.append(formatted_claim)
                except json.JSONDecodeError:
                    pass
            
            # If that fails, try to extract individual JSON objects
            if not claims:
                json_strings = re.findall(r'{\s*"agent_name".*?}', content, re.DOTALL)
                for json_str in json_strings:
                    try:
                        claim = json.loads(json_str)
                        claims.append(claim)
                    except json.JSONDecodeError:
                        continue
        
        return claims


    def run_claim_extraction(self, news_article: str) -> Dict[str, Any]:
        """
        Run the complete claim extraction workflow on a news article.
        
        Args:
            news_article: The news article text to process
            
        Returns:
            Dictionary containing the final state of the workflow
        """
        # Build the workflow if not already built
        if self.chain is None:
            self.build_workflow()
            
        # Initialize the state
        initial_state = ClaimExtractionState(
            news_article=news_article,
            extracted_claims=[],
            validated_claims=[],
            #final_claims=[]
        )
        
        # Run the workflow
        final_state = self.chain.invoke(initial_state)
        
        return final_state

    def visualize(self):
        """
        Visualize the workflow graph.
        """
        # Add imports for visualization
        from IPython.display import display, Image
        return display(Image(self.chain.get_graph().draw_mermaid_png()))

# Add utility functions for direct use
def extract_claims_from_article(article_text: str, model_name=None, temperature=0) -> Dict[str, Any]:
    """
    Utility function to extract claims from an article in a single call.
    
    Args:
        article_text: The news article text
        model_name: Optional model name to use
        temperature: Temperature setting for the model
        
    Returns:
        Dictionary with extracted, validated, and final claims
    """
    extractor = ClaimExtractor(model_name, temperature)
    return extractor.run(article_text)
