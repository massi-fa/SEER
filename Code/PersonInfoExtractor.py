import requests
import json
import time
from typing import Dict, Any, Optional, List

class PersonInfoExtractor:
    """
    A class to extract information about persons from Wikidata and other sources.
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
                
                # Check for 5xx server errors which are also worth retrying
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

    def search_person(self, name: str) -> List[tuple[str, str]]:
        """
        Search for persons in Wikidata by name.
        
        Args:
            name: The name of the person to search for.
            
        Returns:
            Returns a list of tuples (entity_id, entity_label), or an empty list if none found.
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
            print(f"JSON decode error in search_person: {e}")
            print(f"Response content: {response.text[:500]}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Request error in search_person: {e}")
            return []

        persons = []
        if "search" in data and len(data["search"]) > 0:
            # Filter results to find person-type entities
            for result in data["search"]:
                entity_id = result["id"]
                entity_label = result.get("label", name)  # Usa l'etichetta da Wikidata o il nome di ricerca come fallback
                # Check if this is a person
                if self._is_human(entity_id):
                    persons.append((entity_id, entity_label))
        
        return persons
    
    def _is_human(self, entity_id: str) -> bool:
        """
        Check if an entity is a human (instance of human, Q5).
        
        Args:
            entity_id: The Wikidata entity ID.
            
        Returns:
            Returns True if the entity is a human, False otherwise.
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
                print(f"Wikidata API error in _is_human: Status {response.status_code if response else 'None'}")
                return False
                
            if not response.text:
                print("Empty response from Wikidata API in _is_human")
                return False
                
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"JSON decode error in _is_human: {e}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Request error in _is_human: {e}")
            return False
        
        if "entities" in data and entity_id in data["entities"]:
            claims = data["entities"][entity_id].get("claims", {})
            # P31 is the "instance of" property
            if "P31" in claims:
                for claim in claims["P31"]:
                    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                        value = claim["mainsnak"]["datavalue"].get("value", {})
                        # Q5 is "human"
                        if value.get("id") == "Q5":
                            return True
        
        return False
    
    def get_person_info(self, name: str) -> List[Dict[str, Any]]:
        """
        Retrieve information about persons from Wikidata.
        
        Args:
            name: The name of the person to search for.
            
        Returns:
            Returns a list of dicts with the found persons' information.
        """
        # Use only Wikidata
        return self._get_wikidata_person_info(name)
    
    def _extract_birth_date(self, claims: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """
        Extract the birth date from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns the birth date as a string, or None if not found.
        """
        if "P569" in claims:
            birth_date = self._extract_time_value(claims["P569"][0])
            return birth_date
        return None
            
    def _extract_death_date(self, claims: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """
        Extract the death date from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns the death date as a string, or None if not found.
        """
        if "P570" in claims:
            death_date = self._extract_time_value(claims["P570"][0])
            return death_date
        return None
    
    def _extract_birth_place(self, claims: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """
        Extract the birthplace from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns the birthplace as a string, or None if not found.
        """
        if "P19" in claims:
            birth_place_id = self._extract_entity_id(claims["P19"][0])
            if birth_place_id:
                return self._get_entity_label(birth_place_id)
        return None
    
    def _extract_gender(self, claims: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """
        Extract the gender from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns the gender as a string, or None if not found.
        """
        if "P21" in claims:
            gender_id = self._extract_entity_id(claims["P21"][0])
            if gender_id:
                return self._get_entity_label(gender_id)
        return None

    def _extract_political_party(self, claims: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract political party information from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns a list of dicts with party info, sorted chronologically.
        """
        parties = []
        if "P102" in claims:
            for claim in claims["P102"]:
                party_id = self._extract_entity_id(claim)
                if party_id:
                    party_name = self._get_entity_label(party_id)
                    if party_name:
                        # Check qualifiers for start/end dates
                        start_date = None
                        end_date = None
                        if "qualifiers" in claim:
                            # P580 is "start date"
                            if "P580" in claim["qualifiers"]:
                                start_date = self._extract_time_value(claim["qualifiers"]["P580"][0])
                            # P582 is "end date"
                            if "P582" in claim["qualifiers"]:
                                end_date = self._extract_time_value(claim["qualifiers"]["P582"][0])
                        
                        parties.append({
                            "name": party_name,
                            "start_date": start_date,
                            "end_date": end_date
                        })
        
        # Sort parties by end date (most recent first), then by start date
        # This will put current parties (without end date) at the top
        return self._sort_items_chronologically(parties)
    
    def _extract_religion(self, claims: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Extract religion or worldview information from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns a list of religions or worldviews.
        """
        religions = []
        if "P140" in claims:
            for claim in claims["P140"]:
                religion_id = self._extract_entity_id(claim)
                if religion_id:
                    religion_name = self._get_entity_label(religion_id)
                    if religion_name:
                        religions.append(religion_name)
        return religions
    
    def _extract_political_ideology(self, claims: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Extract political ideology information from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns a list of political ideologies.
        """
        ideologies = []
        if "P1142" in claims:
            for claim in claims["P1142"]:
                ideology_id = self._extract_entity_id(claim)
                if ideology_id:
                    ideology_name = self._get_entity_label(ideology_id)
                    if ideology_name:
                        ideologies.append(ideology_name)
        return ideologies

    def _extract_occupation(self, claims: Dict[str, List[Dict[str, Any]]]) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract occupation information from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns a tuple: 
            - A list of occupation names (most recent first)
            - A list of dicts with detailed occupation info including dates
        """
        occupations_with_dates = []
        if "P106" in claims:
            for claim in claims["P106"]:
                occupation_id = self._extract_entity_id(claim)
                if occupation_id:
                    occupation_name = self._get_entity_label(occupation_id)
                    if occupation_name:
                        # Check qualifiers for start/end dates
                        start_date = None
                        end_date = None
                        if "qualifiers" in claim:
                            # P580 is "start date"
                            if "P580" in claim["qualifiers"]:
                                start_date = self._extract_time_value(claim["qualifiers"]["P580"][0])
                            # P582 is "end date"
                            if "P582" in claim["qualifiers"]:
                                end_date = self._extract_time_value(claim["qualifiers"]["P582"][0])
                        
                        occupations_with_dates.append({
                            "name": occupation_name,
                            "start_date": start_date,
                            "end_date": end_date
                        })
        
        # Sort occupations by end date (most recent first), then by start date
        sorted_occupations = self._sort_items_chronologically(occupations_with_dates)
        
        # Extract only names for the simple list
        occupation_names = [occ["name"] for occ in sorted_occupations]
        
        return occupation_names, sorted_occupations

    def _extract_position_held(self, claims: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract positions held information from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns a list of dicts with positions held info, sorted chronologically.
        """
        positions_with_dates = []
        if "P39" in claims:
            for claim in claims["P39"]:
                position_id = self._extract_entity_id(claim)
                if position_id:
                    position_name = self._get_entity_label(position_id)
                    if position_name:
                        # Check qualifiers for start/end dates
                        start_date = None
                        end_date = None
                        organization = None
                        if "qualifiers" in claim:
                            # P580 is "start date"
                            if "P580" in claim["qualifiers"]:
                                start_date = self._extract_time_value(claim["qualifiers"]["P580"][0])
                            # P582 is "end date"
                            if "P582" in claim["qualifiers"]:
                                end_date = self._extract_time_value(claim["qualifiers"]["P582"][0])
                            # P642 is "of" (organization)
                            if "P642" in claim["qualifiers"]:
                                org_id = self._extract_entity_id(claim["qualifiers"]["P642"][0])
                                if org_id:
                                    organization = self._get_entity_label(org_id)
                        
                        positions_with_dates.append({
                            "name": position_name,
                            "start_date": start_date,
                            "end_date": end_date,
                            "organization": organization
                        })
        
        # Sort positions by end date (most recent first), then by start date
        return self._sort_items_chronologically(positions_with_dates)
    
    def _extract_field_of_work(self, claims: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Extract field of work information from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns a list of fields of work.
        """
        fields = []
        if "P101" in claims:
            for claim in claims["P101"]:
                field_id = self._extract_entity_id(claim)
                if field_id:
                    field_name = self._get_entity_label(field_id)
                    if field_name:
                        fields.append(field_name)
        return fields

    def _extract_image(self, claims: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """
        Extract the image URL from Wikidata claims.
        
        Args:
            claims: The dictionary of Wikidata claims.
            
        Returns:
            Returns the image URL as a string, or None if not found.
        """
        if "P18" in claims:
            image_filename = self._extract_string_value(claims["P18"][0])
            if image_filename:
                return self._get_commons_image_url(image_filename)
        return None

    def _get_wikidata_person_info(self, name: str) -> List[Dict[str, Any]]:
        """
        Get detailed person information from Wikidata.
        
        Args:
            name: The name of the person to search for.
            
        Returns:
            Returns a list of dicts with Wikidata info for each found person.
        """
        
        entity_infos = self.search_person(name)
        if not entity_infos or len(entity_infos) == 0:
            return []
        
        results = []
        
        # Iterate over all found entities
        for entity_id, entity_label in entity_infos:
            result = {
                "name": entity_label,
                #"alias": name,
                "wikidata_id": entity_id,
                "birth_date": None,
                "death_date": None,
                "birth_place": None,
                "gender": None,
                "political_party": [],
                "religion": [],
                "political_ideology": [],
                "occupation": [],
                "field_of_work": [],
                "position_held": [],
                "description": None,
                "image_url": None,
            }
            
            # Get entity details
            params = {
                "action": "wbgetentities",
                "format": "json",
                "ids": entity_id,
                "props": "claims|descriptions"
            }
            
            response = self._make_request(params)
            if response is None or response.status_code != 200:
                continue
                
            data = response.json()
            
            if "entities" in data and entity_id in data["entities"]:
                entity = data["entities"][entity_id]
                
                # Extract description
                if "descriptions" in entity and "en" in entity["descriptions"]:
                    result["description"] = entity["descriptions"]["en"]["value"]
                
                # Extract claims
                if "claims" in entity:
                    claims = entity["claims"]
                    
                    # Extract basic information
                    result["birth_date"] = self._extract_birth_date(claims)
                    result["death_date"] = self._extract_death_date(claims)
                    result["birth_place"] = self._extract_birth_place(claims)
                    result["gender"] = self._extract_gender(claims)
                    
                    # Extract political information
                    result["political_party"] = self._extract_political_party(claims)
                    result["religion"] = self._extract_religion(claims)
                    result["political_ideology"] = self._extract_political_ideology(claims)
                    
                    # Extract professional information
                    occupation_names, occupation_details = self._extract_occupation(claims)
                    result["occupation"] = occupation_names
                    result["occupation_details"] = occupation_details
                    result["field_of_work"] = self._extract_field_of_work(claims)
                    result["position_held"] = self._extract_position_held(claims)
                    
                    # Extract image
                    result["image_url"] = self._extract_image(claims)
            
            results.append(result)
        
        return results
    
    def _extract_entity_id(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract the entity ID from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            Returns the entity ID as a string, or None if not found.
        """
        if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
            value = claim["mainsnak"]["datavalue"].get("value", {})
            return value.get("id")
        return None
    
    def _extract_time_value(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract a time value from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            Returns the time value as a string, or None if not found.
        """
        if "datavalue" in claim and "value" in claim["datavalue"]:
            value = claim["datavalue"]["value"]
            if "time" in value:
                # Format the time into a readable date
                time_str = value["time"]
                # Remove the "+" prefix and precision suffix
                if time_str.startswith("+"):
                    time_str = time_str[1:]
                # Extract only the date part (YYYY-MM-DD)
                date_part = time_str.split("T")[0]
                return date_part
        return None
    
    def _extract_string_value(self, claim: Dict[str, Any]) -> Optional[str]:
        """
        Extract a string value from a Wikidata claim.
        
        Args:
            claim: The Wikidata claim.
            
        Returns:
            Returns the string value, or None if not found.
        """
        if "datavalue" in claim and "value" in claim["datavalue"]:
            return claim["datavalue"]["value"]
        return None
    
    def _get_entity_label(self, entity_id: str) -> Optional[str]:
        """
        Get the label of a Wikidata entity.
        
        Args:
            entity_id: The Wikidata entity ID.
            
        Returns:
            Returns the entity label as a string, or None if not found.
        """
        params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "labels"
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
            entity = data["entities"][entity_id]
            if "labels" in entity and "en" in entity["labels"]:
                return entity["labels"]["en"]["value"]
        return None
    
    def _get_commons_image_url(self, filename: str) -> str:
        """
        Convert a Wikimedia Commons filename into a direct image URL.
        
        Args:
            filename: The Wikimedia Commons filename.
            
        Returns:
            Returns the direct image URL.
        """
        # Replace spaces with underscores
        filename = filename.replace(" ", "_")
        
        # Calculate MD5 hash of the filename
        import hashlib
        md5_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
        
        # Build the URL
        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{md5_hash[0]}/{md5_hash[0:2]}/{filename}/300px-{filename}"
    
    def _sort_items_chronologically(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort items chronologically, with the most recent first.
        
        Args:
            items: A list of dicts containing 'start_date' and 'end_date'.
            
        Returns:
            The sorted list.
        """
        def sort_key(item):
            end_date = item.get("end_date")
            start_date = item.get("start_date")
            
            # Handle case where end_date is None (current position)
            # Current positions (without end date) should come first
            if end_date is None:
                end_date = "9999-99-99"  # A future value to put items without end in top
            
            # Handle case where start_date is None
            if start_date is None:
                start_date = "0000-00-00"  # A past value for items without start
                
            return (end_date, start_date)
        
        # Sort items
        return sorted(items, key=sort_key, reverse=True)