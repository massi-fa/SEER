import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent_utilities import get_openrouter_llm, load_prompt_by_name
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import MermaidDrawMethod
import spacy
import json
import time
import re
import threading

from PersonInfoExtractor import PersonInfoExtractor 
from OrganizationInfoExtractor import OrganizationInfoExtractor

ATTEMPT_LIMIT = 2
DEBUG = False

# Global lock and cached model for thread-safe spaCy loading
_spacy_lock = threading.Lock()
_spacy_model = None

def get_spacy_model():
    """
    Thread-safe spaCy model loader with singleton pattern.
    Downloads the model if not available.
    
    Returns:
        spaCy language model or None if loading fails
    """
    global _spacy_model
    
    # Fast path: model already loaded
    if _spacy_model is not None:
        return _spacy_model
    
    # Slow path: need to load model (thread-safe)
    with _spacy_lock:
        # Double-check after acquiring lock
        if _spacy_model is not None:
            if DEBUG: print("spaCy model already loaded by another thread")
            return _spacy_model
        
        model_name = "en_core_web_lg"
        
        try:
            # Try to load the model
            _spacy_model = spacy.load(model_name)
            if DEBUG: print(f"spaCy model '{model_name}' loaded successfully")
            return _spacy_model
            
        except OSError:
            # Model not found, try to download it
            print(f"Model '{model_name}' not found. Downloading...")
            try:
                import subprocess
                import sys
                
                # Download using subprocess to avoid import issues
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT
                )
                
                # Try loading again after download
                _spacy_model = spacy.load(model_name)
                print(f"Model '{model_name}' downloaded and loaded successfully")
                return _spacy_model
                
            except Exception as e_download:
                print(f"Failed to download spaCy model '{model_name}': {e_download}")
                print(f"Please install manually: python -m spacy download {model_name}")
                _spacy_model = None
                return None
                
        except Exception as e:
            print(f"Error loading spaCy model '{model_name}': {e}")
            _spacy_model = None
            return None

# Definition of the base state structure
class EntityExtractionState(TypedDict):
    """Base state for the entity extraction agent."""
    article_text: str    # Text of the article
    raw_name_text: str      # Input text to analyze
    type_of_agent: str   # Type of agent (person or organization)
    agent_description: str    # Description of the agent
    processed_name_text: Optional[str]    # Processed entity name text
    search_successful: bool    # Indicates if the search was successful
    entities: List[Dict[str, Any]]   # Entities extracted from Wikidata
    ner_entities: List[Dict[str, Any]] # Entities extracted by NER
    match_evaluation: str    # Result of the match evaluation (YES/NO)
    extraction_history: List[Dict[str, Any]]   # History of searches/processing steps
    wikidata_info: Dict[str, Any]    # Information extracted about the entity from Wikidata
    taxonomy_info: Dict[str, Any]    # Information about the entity's taxonomy
    entity_info: Dict[str, Any]    # Final result of the extraction

class EntityInfoExtractor:
    """Agent for entity information extraction."""

    organizational_speaker_type_taxonomy = {
        "Airline": "Transports passengers or cargo by air.",
        "Consortium": "Group of entities collaborating on a common goal or project.",
        "Cooperative": "Business owned and run by its members for their benefit.",
        "Corporation": "For-profit business, legally separate from its owners/shareholders.",
        "EducationalOrganization": "Institution providing education (e.g., school, university).",
        "FundingScheme": "Organization providing financial resources (grants, investments).",
        "GovernmentOrganization": "Part of a national, regional, or local government.",
        "LibrarySystem": "Manages collections of books and resources for public/private use.",
        "LocalBusiness": "Small business serving a local community.",
        "MedicalOrganization": "Provides healthcare services (e.g., hospital, clinic).",
        "NGO": "Non-profit, independent group advocating for social/political change.",
        "NewsMediaOrganization": "Gathers, produces, and distributes news.",
        "OnlineBusiness": "Company primarily operating via the internet.",
        "PerformingGroup": "Group of artists who perform (e.g., theater, orchestra).",
        "PoliticalParty": "Organization competing in elections.",
        "Project": "Temporary effort to create a unique product/service, often within a larger org.",
        "ResearchOrganization": "Institution focused on conducting research.",
        "SearchRescueOrganization": "Finds and assists people in distress or danger.",
        "SportsOrganization": "Organizes, promotes, or regulates sports.",
        "WorkersUnion": "Organization representing workers' interests.",
        "Other": "Organization not fitting other categories."
    }

    news_speaker_type_taxonomy = {
        "Politician": "Elected or campaigning political figures.",
        "PublicOfficial": "Non-elected government/state officials.",
        "Expert": "Recognized specialists or professionals.",
        "BusinessRepresentative": "Private sector or economic group representatives.",
        "UnionRepresentative": "Labor union or workers' group representatives.",
        "Journalist": "Media professionals quoted or referenced.",
        "Activist": "Individuals involved in organized activism.",
        "Celebrity": "Entertainment, sports, or public media figures.",
        "OrdinaryCitizen": "Quoted people in a personal capacity.",
        "AnonymousSource": "Speakers not identified by name.",
        "Spokesperson": "Official communicators (named or unnamed).",
        "Other": "If no other label applies or context is insufficient."
    }


    def __init__(self, model_name = "openai/gpt-oss-20b", temperature: float = 0):
        """
        Initialize the EntityInfoExtractor.
        Uses thread-safe spaCy model loading.
        """
        self.llm = get_openrouter_llm(model_name=model_name, temperature=temperature)
        self.person_info_extractor = PersonInfoExtractor()
        self.organization_info_extractor = OrganizationInfoExtractor()
        self.workflow = None
        self.chain = None
        
        # Use thread-safe loader
        self.nlp = get_spacy_model()
        
        if self.nlp is None:
            print("WARNING: spaCy model not available. NER search will be disabled.")
        
        self.build_workflow()

    def build_workflow(self):
        """
        Build the workflow for entity information extraction.
        """
        workflow = StateGraph(EntityExtractionState)
        workflow.add_node("direct_search", self.direct_search)
        workflow.add_node("ner_search", self.ner_search)
        workflow.add_node("one_entities_info_extractor", self.one_entities_info_extractor)
        workflow.add_node("multiple_entities_info_extractor", self.multiple_entities_info_extractor)
        workflow.add_node("no_entities_info_extractor", self.no_entities_info_extractor)
        
        workflow.add_edge(START, "direct_search")
        workflow.add_conditional_edges("direct_search", self.route_after_direct_search,
            {"multiple_entities": "multiple_entities_info_extractor", "one_entity": "one_entities_info_extractor", "no_entities": "ner_search"})
        workflow.add_conditional_edges("ner_search", self.route_after_ner_search,
            {"multiple_entities": "multiple_entities_info_extractor", "one_entity": "one_entities_info_extractor", "no_entities": "no_entities_info_extractor"})
        workflow.add_edge("one_entities_info_extractor", "no_entities_info_extractor")
        workflow.add_edge("multiple_entities_info_extractor", "no_entities_info_extractor")
        workflow.add_edge("no_entities_info_extractor", END)
        
        self.workflow = workflow
        self.chain = workflow.compile()
        return self.chain

    def _add_history_event(self, state: EntityExtractionState, event_name: str, details: Dict[str, Any]):
        """Helper to add a structured event to the history."""
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time())) # Human-readable UTC
        state['extraction_history'].append({
            #"timestamp": current_time,
            "event": event_name,
            "details": details
        })

    def direct_search(self, state: EntityExtractionState) -> EntityExtractionState:
        if DEBUG: print('Direct Search started')
        search_name = state["raw_name_text"]
        
        entities = self._search_entities(search_name, state["type_of_agent"])
        found_ids = [e.get('wikidata_id') for e in entities if e.get('wikidata_id')]

        if entities and len(entities) > 0:
            state["entities"] = entities
            state["search_successful"] = True
        else:
            state["entities"] = []
            state["search_successful"] = False
        
        self._add_history_event(state, "DirectSearch_Result", {
            "search_term_used": search_name,
            "entities_found_count": len(entities),
            "found_wikidata_ids": found_ids,
            "search_successful_flag": state["search_successful"]
        })
        if DEBUG: print(f"Direct Search found {len(entities)} entities. Success: {state['search_successful']}")
        return state

    def route_after_direct_search(self, state: EntityExtractionState) -> str:
        if DEBUG: print('Route After Direct Search started')
        if state.get("search_successful", False) and state.get("entities"):
            count = len(state["entities"])
            if count > 1: return "multiple_entities"
            elif count == 1: return "one_entity"
        return "no_entities"

    def ner_search(self, state: EntityExtractionState) -> EntityExtractionState:
        if DEBUG: print('NER Search started')

        if not self.nlp:
            if DEBUG: print("spaCy NLP model not available. Skipping NER search.")
            state["search_successful"] = False
            state["entities"] = [] 
            state["ner_entities"] = []
            self._add_history_event(state, "NERSearch_SpaCyExtraction_Result", {
                "status": "skipped_nlp_unavailable", "ner_entities_extracted_count": 0, "ner_entities_summary": []
            })
            return state
            
        doc = self.nlp(state["raw_name_text"])  # This call should no longer cause E139
        ner_entities_extracted = [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} for ent in doc.ents]
        state["ner_entities"] = ner_entities_extracted

        person_entities = [e for e in ner_entities_extracted if e["label"] == "PERSON"]
        org_entities = [e for e in ner_entities_extracted if e["label"] == "ORG"]
        
        selected_entity_text = None
        if state["type_of_agent"].lower() == "person" and person_entities:
            selected_entity_text = person_entities[0]["text"]
        elif state["type_of_agent"].lower() == "organization" and org_entities:
            selected_entity_text = org_entities[0]["text"]
        
        if selected_entity_text:
            state["processed_name_text"] = selected_entity_text
            if DEBUG: print('NER Search identified potential entity: ', selected_entity_text)
            self._add_history_event(state, "NERSearch_WikidataLookup_Attempt", {
                "selected_ner_entity_text": selected_entity_text, "agent_type": state["type_of_agent"]
            })
            entities_from_ner = self._search_entities(selected_entity_text, state["type_of_agent"])
            found_ids_from_ner = [e.get('wikidata_id') for e in entities_from_ner if e.get('wikidata_id')]

            if entities_from_ner and len(entities_from_ner) > 0:
                state["entities"] = entities_from_ner
                state["search_successful"] = True
            else:
                state["entities"] = [] 
                state["search_successful"] = False
            
            self._add_history_event(state, "NERSearch_WikidataLookup_Result", {
                "search_term_used": selected_entity_text,
                "entities_found_count": len(entities_from_ner),
                "found_wikidata_ids": found_ids_from_ner,
                "search_successful_flag": state["search_successful"]
            })
        else:
            if DEBUG: print('NER Search did not find a suitable entity for Wikidata lookup.')
            state["entities"] = [] 
            state["search_successful"] = False
            self._add_history_event(state, "NERSearch_WikidataLookup_Skipped", {
                "reason": "No suitable NER entity found matching agent type."
            })
        return state

    def one_entities_info_extractor(self, state: EntityExtractionState) -> EntityExtractionState:
        if DEBUG: print('One Entities Info Extractor started')
        if not state.get("entities") or len(state["entities"]) == 0:
            if DEBUG: print("No entities for LLM validation in one_entities_info_extractor.")
            state['match_evaluation'] = "NO" 
            self._add_history_event(state, "LLMEntityValidation_Skipped", {"reason": "No entity to process", "context": "one_entity_flow"})
            return state

        entity = state["entities"][0]
        self._add_history_event(state, "LLMEntityValidation_Start", {
            "context": "one_entity_flow",
            "input_details": {"name": state["raw_name_text"], "type": state["type_of_agent"], "description": state["agent_description"]},
            "candidate_entity_summary": {"wikidata_id": entity.get("wikidata_id"), "label": entity.get("label")}
        })
        
        prompt_template = load_prompt_by_name("single_entity_verification", prompt_file="entity_linking_prompts")
        entity_info_text = f"""Entity Name: **"{state["raw_name_text"]}"**\nEntity Type: **{state["type_of_agent"]}**\nEntity Description: **"{state["agent_description"]}"**\n\nExtracted Entity Data:\n```json\n{json.dumps(entity, indent=2)}\n```\n\nArticle Text:\n"{state["article_text"]}" """
        messages = [SystemMessage(content=prompt_template), HumanMessage(content=entity_info_text)]
        
        max_attempts = ATTEMPT_LIMIT
        attempt = 0
        llm_response_successful = False
        raw_llm_output_content = "N/A"
        
        while attempt < max_attempts and not llm_response_successful:
            attempt += 1
            if DEBUG: print(f"LLM Validation (one entity) Attempt {attempt}/{max_attempts}.")
            try:
                response = self.llm.invoke(messages)
                raw_llm_output_content = response.content 
                json_response = self._parse_json_answer(raw_llm_output_content)
                
                if not isinstance(json_response, dict): 
                    raise ValueError(f"Parsed JSON is not a dictionary. Parsed: {json_response}")
                if "answer" not in json_response or "reasoning" not in json_response:
                    raise ValueError(f"LLM response missing 'answer' or 'reasoning' field. Parsed: {json_response}")

                llm_answer = json_response.get("answer", "") 
                
                self._add_history_event(state, "LLMEntityValidation_Attempt", {
                    "attempt_number": attempt, "raw_llm_response": raw_llm_output_content, 
                    "parsed_llm_response": json_response, "status": "success"
                })

                if str(llm_answer).strip().upper() == "YES":
                    state['match_evaluation'] = "YES"
                    state["wikidata_info"] = entity 
                else:
                    state['match_evaluation'] = "NO"
                llm_response_successful = True 
            except Exception as e:
                error_message = f'LLM Validation Error (attempt {attempt}): {str(e)}'
                if DEBUG: print(error_message)
                self._add_history_event(state, "LLMEntityValidation_Attempt", {
                    "attempt_number": attempt, "raw_llm_response_on_error": raw_llm_output_content, 
                    "error": error_message, "status": "failure"
                })
                if attempt == max_attempts: state['match_evaluation'] = "NO"; time.sleep(1) 
        
        if not llm_response_successful: state['match_evaluation'] = "NO"
        self._add_history_event(state, "LLMEntityValidation_End", {
            "context": "one_entity_flow", "final_match_evaluation": state['match_evaluation'],
            "selected_entity_details": {"wikidata_id": state["wikidata_info"].get("wikidata_id"), "label": state["wikidata_info"].get("label")} if state['match_evaluation'] == "YES" else None
        })
        if DEBUG: print(f'One Entities Info Extractor finished. Match: {state["match_evaluation"]}')
        return state

    def multiple_entities_info_extractor(self, state: EntityExtractionState) -> EntityExtractionState:
        if DEBUG: print('Multiple Entities Info Extractor started')
        if not state.get("entities") or len(state["entities"]) == 0:
            if DEBUG: print("No entities for LLM validation in multiple_entities_info_extractor.")
            state['match_evaluation'] = "NO"; state["wikidata_info"] = {}
            self._add_history_event(state, "LLMEntityValidation_Skipped", {"reason": "No entities to process", "context": "multiple_entities_flow"})
            return state
            
        self._add_history_event(state, "LLMEntityValidation_Start", {
            "context": "multiple_entities_flow",
            "input_details": {"name": state["raw_name_text"], "type": state["type_of_agent"], "description": state["agent_description"]},
        })

        prompt_template = load_prompt_by_name("multiple_entities_verification", prompt_file="entity_linking_prompts")
        entity_candidates_text = f"""Entity Name: **"{state["raw_name_text"]}"**\nEntity Type: **{state["type_of_agent"]}**\nEntity Description: **"{state["agent_description"]}"**\n\nCandidate Entities:\n```json\n{json.dumps(state["entities"], indent=2)}\n```\n\nArticle Text:\n"{state["article_text"]}" """
        messages = [SystemMessage(content=prompt_template), HumanMessage(content=entity_candidates_text)]

        max_attempts = ATTEMPT_LIMIT
        attempt = 0
        llm_response_successful = False
        raw_llm_output_content = "N/A"
        
        state['match_evaluation'] = "NO"; state["wikidata_info"] = {} # Default
        
        while attempt < max_attempts and not llm_response_successful:
            attempt += 1
            if DEBUG: print(f"LLM Validation (multiple entities) Attempt {attempt}/{max_attempts}.")
            try:
                response = self.llm.invoke(messages)
                raw_llm_output_content = response.content
                json_response = self._parse_json_answer(raw_llm_output_content)

                if not isinstance(json_response, dict) or "answer" not in json_response or "reasoning" not in json_response: 
                    raise ValueError(f"LLM response missing 'answer' or 'reasoning' field or not a dict. Parsed: {json_response}")

                llm_answer_id = json_response.get("answer")
                self._add_history_event(state, "LLMEntityValidation_Attempt", {
                    "attempt_number": attempt, 
                    "parsed_llm_response": json_response, "status": "success"
                })
                
                chosen_wikidata_id = str(llm_answer_id).strip() if llm_answer_id is not None else "NONE"
                if chosen_wikidata_id != "NONE" and chosen_wikidata_id != "":
                    for entity_candidate in state.get("entities", []):
                        if entity_candidate.get("wikidata_id") == chosen_wikidata_id:
                            state['match_evaluation'] = "YES"; state["wikidata_info"] = entity_candidate
                            if DEBUG: print(f"LLM selected matching entity: {entity_candidate.get('label')}")
                            break
                    if state['match_evaluation'] == "NO" and DEBUG: print(f"LLM returned ID '{chosen_wikidata_id}' but not found in candidates.")
                llm_response_successful = True
            except Exception as e:
                error_message = f'LLM Validation Error (attempt {attempt}): {str(e)}'
                if DEBUG: print(error_message)
                self._add_history_event(state, "LLMEntityValidation_Attempt", {
                     "attempt_number": attempt, "raw_llm_response_on_error": raw_llm_output_content, 
                     "error": error_message, "status": "failure"
                })
                if attempt == max_attempts: time.sleep(1)
        
        self._add_history_event(state, "LLMEntityValidation_End", {
            "context": "multiple_entities_flow", "final_match_evaluation": state['match_evaluation'],
            "selected_entity_details": {"wikidata_id": state["wikidata_info"].get("wikidata_id"), "label": state["wikidata_info"].get("name")} if state['match_evaluation'] == "YES" else None
        })
        if DEBUG: print(f'Multiple Entities Info Extractor finished. Match: {state["match_evaluation"]}')
        return state

    def _invoke_taxonomy_with_retry(self, state: EntityExtractionState, is_person: bool) -> dict:
        max_attempts = ATTEMPT_LIMIT
        attempt = 0
        
        # Initial final_result in case all attempts (including retries for invalid terms) fail
        final_result = {"classification": "Unknown", "confidence": "NONE", "explanation": "Taxonomy classification failed after all attempts."}

        while attempt < max_attempts:
            attempt += 1
            raw_response_content = "N/A" # Initialize for this attempt
            parsed_response_from_llm = {} # Initialize for this attempt
            llm_originally_provided_valid_term = False # Initialize for this attempt

            try:
                taxonomy_info_wrapper = {}
                result_key = ""
                if is_person:
                    taxonomy_info_wrapper = self._get_taxonomy_classification_person(state) 
                    result_key = "speaker_type"
                else:
                    taxonomy_info_wrapper = self._get_taxonomy_classification_organization(state)
                    result_key = "organization_type"
                
                raw_response_content = taxonomy_info_wrapper.get("raw_response", "N/A")
                parsed_response_from_llm = taxonomy_info_wrapper.get("parsed_llm_json", {})
                llm_originally_provided_valid_term = taxonomy_info_wrapper.get("is_term_valid", False) # Get the flag

                if "error" in taxonomy_info_wrapper: # Handles hard errors from _get_taxonomy_classification...
                    raise ValueError(f"Taxonomy LLM call failed: {taxonomy_info_wrapper.get('error', 'Unknown error')}")

                # classification_payload_from_wrapper already contains 'Other' if the original term was invalid
                classification_payload_from_wrapper = taxonomy_info_wrapper.get(result_key, {}) 
                current_classification_details = { # This is what this attempt *would* yield
                    "classification": classification_payload_from_wrapper.get("classification", "Other"),
                    "confidence": classification_payload_from_wrapper.get("confidence", "LOW"),
                    "explanation": classification_payload_from_wrapper.get("explanation", "Explanation missing.")
                }

                if not llm_originally_provided_valid_term:
                    # LLM returned an invalid term.
                    self._add_history_event(state, "TaxonomyClassification_Attempt", {
                        "attempt_number": attempt,  
                        "parsed_llm_json": parsed_response_from_llm,
                        "classification_outcome_this_attempt (defaulted if invalid)": current_classification_details,
                        "llm_originally_provided_valid_term": llm_originally_provided_valid_term, # Will be False
                        "status": "llm_returned_invalid_term"
                    })
                    if attempt < max_attempts:
                        if DEBUG: print(f"Taxonomy Attempt {attempt}: LLM returned invalid term '{parsed_response_from_llm.get(result_key, {}).get('classification', 'N/A')}'. Retrying...")
                        time.sleep(1) # Wait before retrying
                        continue # Force a new attempt
                    else:
                        # Last attempt, and LLM still gave an invalid term.
                        # The current_classification_details already has the "Other" default from the inner function.
                        if DEBUG: print(f"Taxonomy Attempt {attempt} (Last): LLM returned invalid term. Accepting default: {current_classification_details}")
                        final_result = current_classification_details # Accept the defaulted value
                        # Log this final state of the last attempt
                        self._add_history_event(state, "TaxonomyClassification_End_LastAttemptInvalid", {
                             "final_classification_details": final_result,
                             "reason": "LLM provided invalid term on final attempt, defaulted to 'Other'."
                        })
                        return final_result # Exit loop, this is the final decision for invalid term on last try
                
                # If we reach here, llm_originally_provided_valid_term was True
                final_result = current_classification_details # Store the valid result
                self._add_history_event(state, "TaxonomyClassification_Attempt", {
                    "attempt_number": attempt,  
                    "parsed_llm_json": parsed_response_from_llm,
                    "classification_outcome": final_result,
                    "llm_originally_provided_valid_term": llm_originally_provided_valid_term, # Will be True
                    "status": "success"
                })
                if DEBUG: print(f"Taxonomy Attempt {attempt}: Success with valid term. {final_result}")
                return final_result # Return on first success with a valid term
                
            except Exception as e: # Catches hard errors like API issues, or the ValueError for "error" key
                error_msg = f'Hard error during taxonomy classification (attempt {attempt}): {str(e)}'
                if DEBUG: print(error_msg)
                
                # Log this specific attempt's hard failure
                self._add_history_event(state, "TaxonomyClassification_Attempt", {
                    "attempt_number": attempt, 
                    "raw_llm_response_on_error": raw_response_content,
                    "parsed_llm_json_on_error": parsed_response_from_llm,
                    "error": error_msg, 
                    "classification_outcome": {"classification": "Error", "confidence": "NONE", "explanation": error_msg},
                    "llm_originally_provided_valid_term": llm_originally_provided_valid_term, # Could be anything if error happened before check
                    "status": "failure_this_attempt"
                })
                if attempt < max_attempts: 
                    time.sleep(1)
                # If it's the last attempt and it's a hard failure, final_result remains the initial "Unknown"
        
        # This event logs the overall failure if the loop completes without returning (e.g. all attempts had hard errors)
        self._add_history_event(state, "TaxonomyClassification_End_AllAttemptsFailed", {
            "final_classification_details": final_result, # This will be the initial "Unknown"
            "status": "failed_all_attempts_due_to_hard_errors"
        })
        return final_result # Returns the initial "Unknown" if all attempts had hard errors


    def no_entities_info_extractor(self, state: EntityExtractionState) -> EntityExtractionState:
        if DEBUG: print('Taxonomy Classification step started')
        is_person = state["type_of_agent"].lower() == "person"
        taxonomy_result = self._invoke_taxonomy_with_retry(state, is_person) 
        state["taxonomy_info"] = taxonomy_result 
        if DEBUG: print(f"Taxonomy Info set: {state['taxonomy_info']}\nTaxonomy Classification step finished")
        return state

    def _get_taxonomy_classification_person(self, state: EntityExtractionState) -> Dict[str, Any]:
        agent_name = state.get("processed_name_text") or state["raw_name_text"]
        system_prompt = load_prompt_by_name("get_taxonomy_classification_person", prompt_file="entity_linking_prompts")
        human_prompt = f"Context:\n- Name: {agent_name}\n- Description: {state['agent_description']}\n- Article Text: {state['article_text']}"
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        raw_response_content = "N/A"
        llm_parsed_json = {}
        is_valid_taxonomy_term = False # Default
        try:
            response = self.llm.invoke(messages)
            raw_response_content = response.content
            llm_parsed_json = self._parse_json_answer(raw_response_content)
            
            if not isinstance(llm_parsed_json, dict):
                 raise json.JSONDecodeError("Parsed content from LLM is not a dictionary.", raw_response_content,0)

            speaker_type_info_from_llm = llm_parsed_json.get("speaker_type", {})
            if not isinstance(speaker_type_info_from_llm, dict): speaker_type_info_from_llm = {}

            original_classification = str(speaker_type_info_from_llm.get("classification", "Other")).replace("**", "").strip()
            validated_classification = original_classification
            current_explanation = speaker_type_info_from_llm.get("explanation", "No explanation provided by LLM.")

            if original_classification in self.news_speaker_type_taxonomy:
                is_valid_taxonomy_term = True
            else:
                is_valid_taxonomy_term = False
                if DEBUG: print(f"Person taxonomy: LLM classification '{original_classification}' invalid. Will be defaulted to 'Other'.")
                validated_classification = "Other" # Default to other if invalid
                current_explanation = f"Original LLM classification '{original_classification}' was invalid; defaulted to 'Other'. LLM Explanation: {current_explanation}".strip()
            
            final_speaker_type_payload = {
                "classification": validated_classification,
                "confidence": speaker_type_info_from_llm.get("confidence", "LOW"),
                "explanation": current_explanation
            }
            return {"speaker_type": final_speaker_type_payload, 
                    "raw_response": raw_response_content, 
                    "parsed_llm_json": llm_parsed_json,
                    "is_term_valid": is_valid_taxonomy_term} # ADDED FLAG

        except Exception as e:
            error_explanation = f"Error processing person taxonomy: {e}"
            if DEBUG: print(f"{error_explanation}. Raw response: {raw_response_content}")
            return {"error": str(e), "raw_response": raw_response_content, "parsed_llm_json_on_error": llm_parsed_json,
                    "speaker_type": {"classification": "Other", "confidence": "LOW", "explanation": error_explanation},
                    "is_term_valid": False} # ADDED FLAG (False on error)

    def _get_taxonomy_classification_organization(self, state: EntityExtractionState) -> Dict[str, Any]:
        agent_name = state.get("processed_name_text") or state["raw_name_text"]
        system_prompt = load_prompt_by_name("get_taxonomy_classification_organization", prompt_file="entity_linking_prompts")
        human_prompt = f"Context:\n- Name: {agent_name}\n- agent Type: {state['type_of_agent']}\n- Description: {state['agent_description']}\n- Article Text: {state['article_text']}"
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        
        raw_response_content = "N/A"
        llm_parsed_json = {}
        is_valid_taxonomy_term = False # Default
        try:
            response = self.llm.invoke(messages)
            raw_response_content = response.content
            llm_parsed_json = self._parse_json_answer(raw_response_content)

            if not isinstance(llm_parsed_json, dict): 
                 raise json.JSONDecodeError("Parsed content from LLM is not a dictionary.", raw_response_content, 0)

            org_type_info_from_llm = llm_parsed_json.get("organization_type", {})
            if not isinstance(org_type_info_from_llm, dict): org_type_info_from_llm = {}

            original_classification = str(org_type_info_from_llm.get("classification", "Other")).replace("**", "").strip()
            validated_classification = original_classification
            current_explanation = org_type_info_from_llm.get("explanation", "No explanation provided by LLM.")

            if original_classification in self.organizational_speaker_type_taxonomy:
                is_valid_taxonomy_term = True
            else:
                is_valid_taxonomy_term = False
                if DEBUG: print(f"Organization taxonomy: LLM classification '{original_classification}' invalid. Will be defaulted to 'Other'.")
                validated_classification = "Other" # Default to other if invalid
                current_explanation = f"Original LLM classification '{original_classification}' was invalid; defaulted to 'Other'. LLM Explanation: {current_explanation}".strip()
            
            final_org_type_payload = {
                "classification": validated_classification,
                "confidence": org_type_info_from_llm.get("confidence", "LOW"),
                "explanation": current_explanation
            }
            return {"organization_type": final_org_type_payload, 
                    "raw_response": raw_response_content, 
                    "parsed_llm_json": llm_parsed_json,
                    "is_term_valid": is_valid_taxonomy_term} # ADDED FLAG

        except Exception as e:
            error_explanation = f"Error processing organization taxonomy: {e}"
            if DEBUG: print(f"{error_explanation}. Raw response: {raw_response_content}")
            return {"error": str(e), "raw_response": raw_response_content, "parsed_llm_json_on_error": llm_parsed_json,
                    "organization_type": {"classification": "Other", "confidence": "LOW", "explanation": error_explanation},
                    "is_term_valid": False} # ADDED FLAG (False on error)

    def visualize(self):
        if self.chain is None: print("Workflow not built."); return
        try:
            from IPython.display import Image, display
            img_bytes = self.chain.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
            display(Image(img_bytes))
        except ImportError: print("IPython.display not available. Mermaid text output:\n", self.chain.get_graph().draw_mermaid())
        except Exception as e: print(f"Graph visualization error: {e}\nMermaid text output:\n", self.chain.get_graph().draw_mermaid())

    def run(self, article_text: str, raw_name_text: str, type_of_agent: str, agent_description: str) -> Dict[str, Any]:
        initial_state = EntityExtractionState(
            article_text=article_text, raw_name_text=raw_name_text, type_of_agent=type_of_agent,
            agent_description=agent_description, processed_name_text=None, search_successful=False, 
            entities=[], ner_entities=[], match_evaluation="NO", extraction_history=[], 
            wikidata_info={}, taxonomy_info={}, entity_info={}
        )
        final_state = self.chain.invoke(initial_state)
        
        name_to_use = final_state.get("processed_name_text") or final_state["raw_name_text"]
        final_wikidata_info = final_state.get("wikidata_info", {})
        final_taxonomy_info = final_state.get("taxonomy_info", {})

        entity_info = {
            "name": name_to_use, "type_of_agent": final_state["type_of_agent"],
            "agent_description": final_state["agent_description"], "wikidata_info": final_wikidata_info,
            "taxonomy_info": final_taxonomy_info, "match_evaluation": final_state.get("match_evaluation", "NO"), 
            "extraction_history": final_state["extraction_history"]
        }
        final_state["entity_info"] = entity_info
        # Add a final event to mark the end of the whole process
        return final_state

    def _search_entities(self, search_name: str, type_of_agent: str) -> List[Dict[str, Any]]:
        if not search_name or not search_name.strip(): return []
        entities = []
        try:
            if type_of_agent.lower() == "person": entities = self.person_info_extractor._get_wikidata_person_info(search_name)
            elif type_of_agent.lower() == "organization": entities = self.organization_info_extractor._get_wikidata_organization_info(search_name)
            else: 
                if DEBUG: print(f"Unknown type_of_agent: {type_of_agent} in _search_entities")
        except Exception as e:
            if DEBUG: print(f"Error during _search_entities for '{search_name}' ({type_of_agent}): {e}")
        return entities if isinstance(entities, list) else [] 

    def route_after_ner_search(self, state: EntityExtractionState) -> str:
        if DEBUG: print('Route After NER Search started')
        if state.get("search_successful", False) and state.get("entities"):
            count = len(state["entities"])
            if count > 1: return "multiple_entities"
            elif count == 1: return "one_entity"
        return "no_entities"
        
    def _parse_json_answer(self, json_string: str) -> Dict[str, Any]:
        if not json_string or not isinstance(json_string, str):
            if DEBUG: print("_parse_json_answer: Input is not a valid string.")
            return {}
        cleaned_json = json_string.strip()
        # Remove markdown code blocks
        cleaned_json = re.sub(r"^```json\s*([\s\S]*?)\s*```$", r"\1", cleaned_json, flags=re.MULTILINE)
        cleaned_json = re.sub(r"^```\s*([\s\S]*?)\s*```$", r"\1", cleaned_json, flags=re.MULTILINE)
        cleaned_json = cleaned_json.strip()

        try: 
            data = json.loads(cleaned_json)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError: pass 
        try:
            start_idx = cleaned_json.find("{"); end_idx = cleaned_json.rfind("}")
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                data = json.loads(cleaned_json[start_idx : end_idx + 1])
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError: pass
        try: # Try to fix common quoting issues and trailing commas
            # Fix trailing commas
            temp_json = re.sub(r",\s*([}\]])", r"\1", cleaned_json)
            # Fix unquoted keys and single quotes for keys/values (simplistic)
            temp_json = re.sub(r"([{,]\s*)(\w+)(\s*:)", r'\1"\2"\3', temp_json) # unquoted keys
            temp_json = temp_json.replace("'", "\"") # all single to double (can be risky)
            data = json.loads(temp_json)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError, re.error): pass
        
        if DEBUG: print(f"Final JSON parsing error. Original (snippet): {json_string[:100]}... Cleaned (snippet): {cleaned_json[:100]}...")
        return {}