import requests
import json
import time
from typing import Dict, Any, Optional, List

class OrganizationInfoExtractor:
    """
    A class to extract organization information from Wikidata and other sources.
    """
    
    def __init__(self):
        self.wikidata_api_url = "https://www.wikidata.org/w/api.php"
        self.wikidata_entity_url = "https://www.wikidata.org/wiki/Special:EntityData/"
        # Set a proper user agent as required by Wikidata
        self.headers = {
            'User-Agent': 'ClaimExtractionAgent/1.0 (https://example.com/contact; your-email@example.com) requests/2.31.0'
        }

    def _make_request(self, params: Dict[str, Any]) -> Optional[requests.Response]:
        """
        Make a request to Wikidata API with rate limit handling (exponential backoff).
        """
        max_retries = 5
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.wikidata_api_url, params=params, headers=self.headers)
                
                # Handle Rate Limit (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                    print(f"Rate limited (429). Retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                # Check for 5xx server errors
                if 500 <= response.status_code < 600:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Server error {response.status_code}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                return response
                
            except requests.exceptions.RequestException as e:
                wait_time = base_delay * (2 ** attempt)
                print(f"Request error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        print("Max retries exceeded.")
        return None
    
    def search_organization(self, name: str) -> List[tuple[str, str]]:
        """
        Search for organizations on Wikidata by name.
        
        Args:
            name: The name of the organization to search for.
            
        Returns:
            A list of tuples (entity_id, entity_label) if found, empty list otherwise.
        """
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": name,
            "type": "item"
        }
        
        try:
            response = self._make_request(params)
            
            # Check if the response is successful and has content
            if response is None or response.status_code != 200:
                print(f"Wikidata API error: Status {response.status_code if response else 'None'}")
                return []
                
            if not response.text:
                print("Empty response from Wikidata API")
                return []
                
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error in search_organization: {e}")
            print(f"Response content: {response.text[:500]}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Request error in search_organization: {e}")
            return []
        
        results = []
        if "search" in data and len(data["search"]) > 0:
            # Filter results to include only organization entities
            for result in data["search"]:
                entity_id = result["id"]
                entity_label = result.get("label", name)  # Usa l'etichetta da Wikidata o il nome di ricerca come fallback
                # Check if this is an organization
                if self._is_organization(entity_id):
                    results.append((entity_id, entity_label))
        
        return results
    
    def _is_organization(self, entity_id: str) -> bool:
        """
        Check if an entity is an organization of any type (company, government body, NGO, etc.).
        
        Args:
            entity_id: The Wikidata entity ID.
            
        Returns:
            True if the entity is an organization, False otherwise.
        """
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "claims"
        }
        
        try:
            response = self._make_request(params)
            
            if response is None or response.status_code != 200:
                print(f"Wikidata API error in _is_organization: Status {response.status_code if response else 'None'}")
                return False
                
            if not response.text:
                print("Empty response from Wikidata API in _is_organization")
                return False
                
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error in _is_organization: {e}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Request error in _is_organization: {e}")
            return False
        
        if "entities" in data and entity_id in data["entities"]:
            claims = data["entities"][entity_id].get("claims", {})
            # P31 is the 'instance of' property
            if "P31" in claims:
                for claim in claims["P31"]:
                    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                        value = claim["mainsnak"]["datavalue"].get("value", {})
                        # Extended list of organization types
                        org_types = [
                            "Q43229",    # organization
                            "Q4830453",  # business
                            "Q783794",   # company
                            "Q6881511",  # enterprise
                            "Q891723",   # joint-stock company
                            "Q7210356",  # political organization
                            "Q41487",    # non-governmental organization
                            "Q327333",   # government agency
                            "Q20857065", # human rights organization
                            "Q15911314", # public institution
                            "Q5341295",  # council of ministers
                            "Q847017",   # government cabinet
                            "Q212198",   # parliament
                            "Q7188",     # government
                            "Q262166",   # political party
                            "Q31855",    # research institute
                            "Q3918",     # university
                            "Q4287745",  # association
                            "Q163740",   # non-profit organization
                            "Q1660397",  # media company
                            "Q1110684",  # news agency
                            "Q1114515",  # intergovernmental organization
                            "Q15343039", # technology company
                            "Q1664720",  # institution
                            "Q748019",   # scientific organization
                            "Q3152824",  # religious organization
                            "Q2659904",  # international organization
                            "Q5633421",  # scientific society
                            "Q1194951",  # international court
                            "Q1752939",  # judiciary
                            "Q178790",   # coalition
                            "Q2385804",  # educational institution
                            "Q4671277",  # academic institution
                            "Q1298668",  # labor union
                            "Q622659",   # charity
                            "Q2178147",  # foundation
                            "Q2061186",  # publisher
                            "Q1616075",  # television network
                            "Q1137809",  # healthcare organization
                            # Added countries and states
                            "Q6256",     # country (sovereign state)
                            "Q3624078",  # federated state
                        ]
                        if value.get("id") in org_types:
                            return True
        
        return False
    
    def get_organization_info(self, name: str) -> List[Dict[str, Any]]:
        """
        Get organization information from Wikidata.
        
        Args:
            name: The organization name.
            
        Returns:
            A list of dictionaries containing information about found organizations.
        """
        
        # Use only Wikidata
        return self._get_wikidata_organization_info(name)
    
    def _get_wikidata_organization_info(self, name: str) -> List[Dict[str, Any]]:
        """
        Get organization information from Wikidata.
        
        Args:
            name: The organization name.
            
        Returns:
            A list of dictionaries containing information from Wikidata.
        """
        entity_infos = self.search_organization(name)
        if not entity_infos or len(entity_infos) == 0:
            return []
        
        results = []
        
        for entity_id, entity_label in entity_infos:
            result = {
                "name": entity_label,
                #"alias": name,
                "found": True,
                "wikidata_id": None,
                "type": None,
                "inception_date": None,
                "dissolution_date": None,
                "headquarters": None,
                "country": None,
                "official_website": None,
                "parent_organization": None,
                "subsidiaries": [],
                "industry": [],
                "key_people": [],
                "description": None,
                "image_url": None,
            }
            
            result["wikidata_id"] = entity_id
            result["name"] = entity_label
            
            # Retrieve detailed information about the entity
            params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": entity_id,
                "props": "claims|descriptions|labels|aliases",
                "languages": "en|it"
            }
            
            response = self._make_request(params)
            if response is None or response.status_code != 200:
                continue

            data = response.json()
            
            if "entities" in data and entity_id in data["entities"]:
                entity = data["entities"][entity_id]
                
                # Get the description
                if "descriptions" in entity:
                    if "en" in entity["descriptions"]:
                        result["description"] = entity["descriptions"]["en"]["value"]
                    elif "it" in entity["descriptions"]:
                        result["description"] = entity["descriptions"]["it"]["value"]
                
                # Get aliases
                if "aliases" in entity:
                    aliases = []
                    for lang in ["en", "it"]:
                        if lang in entity["aliases"]:
                            for alias in entity["aliases"][lang]:
                                aliases.append(alias["value"])
                    if aliases:
                        result["alias"] = aliases[0]  # Prendi il primo alias
                
                # Extract information from claims
                if "claims" in entity:
                    claims = entity["claims"]
                    
                    # Organization type (P31 - instance of)
                    result["type"] = self._extract_organization_type(claims)
                    
                    # Inception date (P571)
                    if "P571" in claims:
                        result["inception_date"] = self._extract_time_value(claims["P571"][0])
                    
                    # Dissolution date (P576 - dissolved, abolished or demolished)
                    if "P576" in claims:
                        result["dissolution_date"] = self._extract_time_value(claims["P576"][0])
                    
                    # Headquarters location (P159)
                    if "P159" in claims:
                        result["headquarters"] = self._extract_location(claims["P159"][0])
                    
                    # Country (P17)
                    if "P17" in claims:
                        result["country"] = self._extract_entity_label(claims["P17"][0])
                    
                    # Official website (P856)
                    if "P856" in claims:
                        result["official_website"] = self._extract_url(claims["P856"][0])
                    
                    # Parent organization (P749)
                    if "P749" in claims:
                        result["parent_organization"] = self._extract_entity_label(claims["P749"][0])
                    
                    # Subsidiaries (P355)
                    if "P355" in claims:
                        for claim in claims["P355"]:
                            subsidiary = self._extract_entity_label(claim)
                            if subsidiary:
                                result["subsidiaries"].append(subsidiary)
                    
                    # Industry (P452)
                    if "P452" in claims:
                        for claim in claims["P452"]:
                            industry = self._extract_entity_label(claim)
                            if industry:
                                result["industry"].append(industry)
                    
                    # Key people (P169 - CEO, P488 - chairperson, etc.)
                    key_people_properties = ["P169", "P488", "P3320", "P1037", "P1075"]
                    for prop in key_people_properties:
                        if prop in claims:
                            for claim in claims[prop]:
                                person = self._extract_entity_label(claim)
                                if person:
                                    result["key_people"].append(person)
                    
                    # Image (P18)
                    if "P18" in claims:
                        result["image_url"] = self._get_commons_image_url(self._extract_string_value(claims["P18"][0]))
            
            results.append(result)
        
        return results
    
    def _extract_organization_type(self, claims: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """
        Extract the organization type from Wikidata claims.
        
        Args:
            claims: The Wikidata claims for the entity.
            
        Returns:
            The organization type as a string, or None if not found.
        """
        if "P31" in claims:  # P31 è "istanza di"
            for claim in claims["P31"]:
                entity_id = self._extract_entity_id(claim)
                if entity_id:
                    entity_label = self._get_entity_label(entity_id)
                    if entity_label:
                        return entity_label
        return None
    
    def _extract_entity_id(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract the entity ID from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            The entity ID as a string, or None if not found.
        """
        if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
            value = claim["mainsnak"]["datavalue"].get("value", {})
            return value.get("id")
        return None
    
    def _extract_entity_label(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract the entity label from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            The entity label as a string, or None if not found.
        """
        entity_id = self._extract_entity_id(claim)
        if entity_id:
            return self._get_entity_label(entity_id)
        return None
    
    def _get_entity_label(self, entity_id: str) -> Optional[str]:
        """
        Get the Wikidata entity label by its ID.
        
        Args:
            entity_id: The Wikidata entity ID.
            
        Returns:
            The entity label as a string, or None if not found.
        """
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "labels",
            "languages": "en|it"
        }
        
        try:
            response = self._make_request(params)
            
            if response is None or response.status_code != 200:
                print(f"Wikidata API error in _get_entity_label: Status {response.status_code if response else 'None'}")
                return None
                
            if not response.text:
                print("Empty response from Wikidata API in _get_entity_label")
                return None
                
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error in _get_entity_label: {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error in _get_entity_label: {e}")
            return None
        
        if "entities" in data and entity_id in data["entities"]:
            labels = data["entities"][entity_id].get("labels", {})
            # Prova prima l'inglese, poi l'italiano
            if "en" in labels:
                return labels["en"]["value"]
            elif "it" in labels:
                return labels["it"]["value"]
        
        return None
    
    def _extract_time_value(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract a time value from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            The time value as a string, or None if not found.
        """
        if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
            value = claim["mainsnak"]["datavalue"].get("value", {})
            if "time" in value:
                # Formatta la data in un formato leggibile
                time_str = value["time"]
                # Rimuovi il prefisso "+" e la precisione alla fine
                if time_str.startswith("+"):
                    time_str = time_str[1:]
                # Estrai solo l'anno, il mese e il giorno
                date_parts = time_str.split("T")[0].split("-")
                if len(date_parts) >= 3:
                    year, month, day = date_parts[:3]
                    # Rimuovi gli zeri iniziali
                    if year.startswith("0"):
                        year = year.lstrip("0")
                    if month.startswith("0"):
                        month = month.lstrip("0")
                    if day.startswith("0"):
                        day = day.lstrip("0")
                    # Formatta la data
                    return f"{day}/{month}/{year}"
        return None
    
    def _extract_location(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract a location information from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            The location as a string, or None if not found.
        """
        return self._extract_entity_label(claim)
    
    def _extract_string_value(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract a string value from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            The string value, or None if not found.
        """
        if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
            return claim["mainsnak"]["datavalue"].get("value")
        return None
    
    def _extract_url(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract a URL from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            The URL as a string, or None if not found.
        """
        return self._extract_string_value(claim)
    
    def _get_commons_image_url(self, filename: Optional[str]) -> Optional[str]:
        """
        Convert a Wikimedia Commons filename into a direct image URL.
        
        Args:
            filename: The Commons file name.
            
        Returns:
            The direct image URL, or None if the filename is invalid.
        """
        if not filename:
            return None
        
        # Sostituisci gli spazi con underscore
        filename = filename.replace(" ", "_")
        
        # Calcola l'hash MD5 del nome del file
        import hashlib
        md5_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
        
        # Costruisci l'URL
        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{md5_hash[0]}/{md5_hash[0:2]}/{filename}/800px-{filename}"