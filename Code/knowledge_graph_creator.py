from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD, SKOS
import os
# The following imports were in the original code; they can be removed if unused.
import re  # Useful for JSON parsing and string manipulation
import hashlib  # Useful for creating consistent unique IDs
from urllib.parse import quote # Useful for creating safe URIs if necessary

class KnowledgeGraphCreator:
    def __init__(self, data=None, base_uri: str = "http://example.org/newstestontology/",
                 base_tbox_file: str | None = None,
                 extension_tbox_file: str | None = None):
        self.data = data
        self.base_uri_str = base_uri if base_uri.endswith(('/', '#')) else base_uri + '/'
        self.BASE = Namespace(self.base_uri_str)
        self.tbox_graph = Graph()
        self.abox_graph = Graph()

        # Resolve default paths relative to this script's location if not provided
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if base_tbox_file is None:
            base_tbox_file = os.path.join(current_dir, "Ontology", "nco2_2.ttl")
        if extension_tbox_file is None:
            extension_tbox_file = os.path.join(current_dir, "Ontology", "nco2_2_schemaorgv2_0.ttl")

        # Namespace map
        self.ns_map = {
            "rdf": RDF, "rdfs": RDFS, "owl": OWL, "xsd": XSD, "skos": SKOS, "base": self.BASE,
            "dcterms": Namespace("http://purl.org/dc/terms/")
            # Add other common prefixes here if necessary, e.g. "nco", "time"
        }
        self.NCO = None # Will be set in _load_tbox_and_namespaces
        self.TIME = None # Will be set in _load_tbox_and_namespaces
        self.DCTERMS = self.ns_map["dcterms"]

        # Dictionaries to avoid URI duplicates
        self.publisher_uris = {}
        self.news_source_uris = {}
        self.topic_uris = {}
        self.claim_uris = {}
        self.role_uris = {}
        self.agent_component_uris = {}

        self.SCHEMA = Namespace("https://schema.org/")
        self.ns_map["schema"] = self.SCHEMA
        self.ORG_TYPE_MAP = {
            "airline": self.SCHEMA.Airline, "consortium": self.SCHEMA.Organization,
            "cooperative": self.SCHEMA.Organization, "corporation": self.SCHEMA.Corporation,
            "educationalorganization": self.SCHEMA.EducationalOrganization, "fundingscheme": self.SCHEMA.FundingScheme,
            "governmentorganization": self.SCHEMA.GovernmentOrganization, "librarysystem": self.SCHEMA.LibrarySystem,
            "localbusiness": self.SCHEMA.LocalBusiness, "medicalorganization": self.SCHEMA.MedicalOrganization,
            "ngo": self.SCHEMA.NGO, "newsmediaorganization": self.SCHEMA.NewsMediaOrganization,
            "onlinebusiness": self.SCHEMA.Organization, "performinggroup": self.SCHEMA.PerformingGroup,
            "politicalparty": self.SCHEMA.PoliticalParty, "project": self.SCHEMA.Organization,
            "researchorganization": self.SCHEMA.ResearchOrganization,
            "searchrescueorganization": self.SCHEMA.Organization, "sportsorganization": self.SCHEMA.SportsOrganization,
            "workersunion": self.SCHEMA.Organization,
            "businessrepresentative": self.SCHEMA.Corporation,
            "other": self.SCHEMA.Organization
        }

        self._bind_namespaces_to_graphs()  # Initial namespace bindings
        self._load_tbox_and_namespaces(base_tbox_file, extension_tbox_file)  # Load TBox and configure namespaces
        self.process_data()

    def _load_tbox_and_namespaces(self, base_tbox_file: str | None, extension_tbox_file: str | None):
        """Load base and extension TBox files and set NCO/TIME namespaces."""
        tbox_files_loaded = False

        if base_tbox_file:
            try:
                print(f"Loading base TBox ontology from: {base_tbox_file}")
                self.tbox_graph.parse(base_tbox_file, format="turtle") # It is good practice to specify the format
                print(f"Base TBox loaded. Graph now has {len(self.tbox_graph)} triples.")
                tbox_files_loaded = True
            except FileNotFoundError:
                print(f"Error: Base TBox ontology file not found at {base_tbox_file}")
            except Exception as e:
                print(f"Error parsing base TBox ontology file '{base_tbox_file}': {e}")

        if extension_tbox_file:
            try:
                print(f"Loading extension TBox ontology from: {extension_tbox_file}")
                self.tbox_graph.parse(extension_tbox_file, format="turtle") # Adds to existing graph
                print(f"Extension TBox loaded. Graph now has {len(self.tbox_graph)} triples.")
                tbox_files_loaded = True
            except FileNotFoundError:
                print(f"Error: Extension TBox ontology file not found at {extension_tbox_file}")
            except Exception as e:
                print(f"Error parsing extension TBox ontology file '{extension_tbox_file}': {e}")

        # Namespace discovery happens after all TBox files have been loaded (or attempted)
        if tbox_files_loaded:
            print("Discovering namespaces from loaded TBox graph(s)...")
            for prefix, namespace_uri_obj in self.tbox_graph.namespaces():
                namespace_uri = str(namespace_uri_obj)
                if prefix and (prefix not in self.ns_map or str(self.ns_map[prefix]) != namespace_uri):
                    # Add or update only if the prefix is not empty and different
                    print(f"  Binding/Updating prefix '{prefix}' to <{namespace_uri}>")
                    self.ns_map[prefix] = Namespace(namespace_uri)

                # Detect NCO and TIME specifically
                # NOTE: Ensure the NCO URI is correct for your files
                # From your files it seems to be "http://data.open.ac.uk/ontology/newsclassification/v2#"
                if namespace_uri == "http://data.open.ac.uk/ontology/newsclassification/v2#":
                    print(f"  Found NCO namespace: <{namespace_uri}> (associated with prefix: {prefix or 'nco'})")
                    self.NCO = Namespace(namespace_uri)
                    self.ns_map[prefix or "nco"] = self.NCO # Ensure 'nco' or the original prefix is mapped
                elif namespace_uri == "http://www.w3.org/2006/time#":
                    print(f"  Found TIME namespace: <{namespace_uri}> (associated with prefix: {prefix or 'time'})")
                    self.TIME = Namespace(namespace_uri)
                    self.ns_map[prefix or "time"] = self.TIME

        # Fallback if NCO or TIME were not found in TBox files
        if not self.NCO:
            nco_fallback_uri = "http://data.open.ac.uk/ontology/newsclassification/v2#" # Updated to v2
            print(f"NCO namespace not found in TBox files, using fallback: <{nco_fallback_uri}>")
            self.NCO = Namespace(nco_fallback_uri)
            self.ns_map["nco"] = self.NCO # Ensure "nco" is in the map
        if not self.TIME:
            time_fallback_uri = "http://www.w3.org/2006/time#"
            print(f"TIME namespace not found in TBox files, using fallback: <{time_fallback_uri}>")
            self.TIME = Namespace(time_fallback_uri)
            self.ns_map["time"] = self.TIME

        self._bind_namespaces_to_graphs() # Re-apply all bindings, including discovered ones

        if not tbox_files_loaded:
            print("No TBox ontology files were successfully loaded. TBox may be empty or incomplete.")

    def _bind_namespaces_to_graphs(self):
        """Bind the namespaces in ns_map to both tbox_graph and abox_graph."""
        for prefix, namespace_obj in self.ns_map.items():
            if namespace_obj: # Ensure namespace object is not None
                self.tbox_graph.bind(prefix, namespace_obj)
                self.abox_graph.bind(prefix, namespace_obj)

    def get_tbox_graph(self) -> Graph:
        return self.tbox_graph

    def get_abox_graph(self) -> Graph:
        return self.abox_graph

    def get_full_graph(self) -> Graph:
        full_graph = Graph()
        # Bind all known namespaces
        for prefix, namespace_obj in self.ns_map.items():
            if namespace_obj:
                 full_graph.bind(prefix, namespace_obj)

        full_graph += self.tbox_graph
        full_graph += self.abox_graph
        return full_graph
    
    def serialize_tbox(self, format: str = "turtle", destination: str = "tbox.ttl") -> None:
        self.tbox_graph.serialize(destination=destination, format=format)
        print(f"TBox serialized to {destination} in {format} format.")

    def serialize_abox(self, format: str = "turtle", destination: str = "abox.ttl") -> None:
        self.abox_graph.serialize(destination=destination, format=format)
        print(f"ABox serialized to {destination} in {format} format.")

    def serialize_knowledge_graph(self, format: str = "turtle", destination: str = "knowledge_graph.ttl") -> None:
        self.get_full_graph().serialize(destination=destination, format=format)
        print(f"Full Knowledge Graph serialized to {destination} in {format} format.")

    # --- Methods to populate the ABox ---

    @staticmethod
    def make_instance_uri(namespace, instance_type: str, instance_id: str) -> URIRef:
        """
        Create a URI for an instance, using only '/' as a separator.
        If the namespace URI ends with '#', it is replaced with '/'.
        Example: make_instance_uri(NCO, "NewsItem", "123") -> NCO["NewsItem/123"]
        """
        ns_str = str(namespace)
        if ns_str.endswith('#'):
            ns_str = ns_str[:-1] + '/'
        # Remove any trailing slashes from type and id to avoid double slashes
        instance_type = instance_type.strip('/').replace(' ', '_')
        instance_id = instance_id.strip('/').replace(' ', '_')
        return URIRef(f"{ns_str}{instance_type}/{instance_id}")

    def process_data(self) -> None:
        if not self.data:
            print("No data to process.")
            return
        
        if not self.NCO or not self.TIME:
            print("Warning: NCO or TIME namespace not initialized. ABox population might be incomplete or incorrect.")
            # Try fallback initialization if not already done
            if not self.NCO: self.NCO = Namespace("http://data.open.ac.uk/ontology/newsclassification#"); self.ns_map["nco"] = self.NCO
            if not self.TIME: self.TIME = Namespace("http://www.w3.org/2006/time#"); self.ns_map["time"] = self.TIME
            self._bind_namespaces_to_graphs()


        for article_data in self.data:
            if isinstance(article_data, dict):
                self._process_article(article_data)
            else:
                print(f"Warning: Invalid article data format. Expected dict, got {type(article_data)}.")

    def _topic_from_article(self, article_id) -> URIRef | None:
        topic_map = {
            'climate_change': ('ClimateChange', 'Climate Change'),
            'uk_immigration': ('UKImmigration', 'UK Immigration'),
            'ukraine_war': ('UkraineWar', 'Ukraine War'),
            'us_politics': ('USPolitics', 'US Politics')
        }
        for key, (topic_name, topic_label) in topic_map.items():
            if key in article_id:
                # Use only / for instances, NCO namespace
                topic_uri = self.make_instance_uri(self.NCO, "Topic", topic_name)
                if (topic_uri, RDF.type, self.NCO.Topic) not in self.abox_graph:
                    self.abox_graph.add((topic_uri, RDF.type, self.NCO.Topic))
                    self.abox_graph.add((topic_uri, SKOS.prefLabel, Literal(topic_label)))
                return topic_uri
        return None

    def _process_agent(self, agent_name: str, agent_data: dict) -> URIRef:
        """
        Create an agent individual (person or organization) and add it to the ABox.
        Returns the agent's URI.
        """
        agent_type = agent_data.get("agent_type", "person")
        taxonomy_info = agent_data.get("taxonomy_info", {})
        classification = taxonomy_info.get("classification", "Role")
        #agent_id = hashlib.md5(f"{agent_name}|{classification}".encode("utf-8")).hexdigest()
        agent_id = re.sub(r'\s+', '_', agent_name)  # Removes spaces and replaces with underscore
        agent_description = agent_data.get("agent_description")
        wikidata_info = agent_data.get("wikidata_info", {})
        wikidata_id = wikidata_info.get("wikidata_id")

        # Create the agent (person or organization)
        if agent_type == "person":
            agent_uri = self.make_instance_uri(self.NCO, "Person", agent_id)
            self.abox_graph.add((agent_uri, RDF.type, self.NCO.Person))
        else:
            org_taxonomy = classification.lower().replace(" ", "")
            org_class = self.ORG_TYPE_MAP.get(org_taxonomy, self.SCHEMA.Organization)
            agent_uri = self.make_instance_uri(self.NCO, "Organization", agent_id)
            self.abox_graph.add((agent_uri, RDF.type, self.NCO.Organization))
            self.abox_graph.add((agent_uri, RDF.type, org_class))
        self.abox_graph.add((agent_uri, RDF.type, self.NCO.Agent))
        self.abox_graph.add((agent_uri, RDFS.label, Literal(agent_name)))
        
        if wikidata_id:
            wikidata_uri = URIRef(f"https://www.wikidata.org/entity/{wikidata_id}")
            self.abox_graph.add((agent_uri, OWL.sameAs, wikidata_uri))
            
        return agent_uri
    
    def _add_publisher_and_source(self, publisher_name: str, source_domain: str):
        """
        Create and add RDF triples for the publisher and news source.
        Returns (publisher_uri, news_source_uri).
        """
        # Check existing dictionaries first to avoid duplicates
        if publisher_name in self.publisher_uris and source_domain in self.news_source_uris:
            publisher_uri = self.publisher_uris[publisher_name]
            news_source_uri = self.news_source_uris[source_domain]
        else:
            #publisher_id_hash = hashlib.md5(publisher_name.encode('utf-8')).hexdigest()
            publisher_id_hash = re.sub(r'\s+', '_', publisher_name)  # Removes spaces and replaces with underscore
            publisher_uri = self.make_instance_uri(self.NCO, "Publisher", publisher_id_hash)
            self.abox_graph.add((publisher_uri, RDF.type, self.NCO.Publisher))
            # A publisher can be an Organization or a Person
            # Here we assume Organization, but logic for Person can be added if needed
            self.abox_graph.add((publisher_uri, RDF.type, self.NCO.Organization))
            self.abox_graph.add((publisher_uri, RDFS.label, Literal(publisher_name)))

            #news_source_id = hashlib.md5(source_domain.encode('utf-8')).hexdigest()
            news_source_id = re.sub(r'\s+', '_', source_domain)  # Removes spaces and replaces with underscore
            news_source_uri = self.make_instance_uri(self.NCO, "NewsSource", news_source_id)
            self.abox_graph.add((news_source_uri, RDF.type, self.NCO.NewsSource))
            self.abox_graph.add((news_source_uri, RDFS.label, Literal(f"News Source: {source_domain}")))
            # In NCO v2.2, NewsSource has exactly one Publisher
            self.abox_graph.add((news_source_uri, self.NCO.hasPublisher, publisher_uri))

            # Add to dictionaries to avoid future duplicates
            self.publisher_uris[publisher_name] = publisher_uri
            self.news_source_uris[source_domain] = news_source_uri

        return publisher_uri, news_source_uri

    def _process_article(self, article_data: dict) -> None:
        article_id = article_data.get("article_id")
        if not article_id:
            print("Warning: Article data missing 'article_id'. Skipping.")
            return

        article_uri = self._add_news_item(article_id, article_data)
        news_source_uri = self._add_news_source(article_uri, article_data)
        topic_uri = self._add_topic(article_uri, article_id)
        self._add_article_metadata(article_uri, article_data)
        agent_uris = self._add_agents(article_data.get("agents_info", {}))
        self._add_claims(article_uri, article_data.get("claims", []), agent_uris, topic_uri, article_id, article_data.get("agents_info", {}))

    def _add_news_item(self, article_id, article_data):
        safe_article_id = re.sub(r'\s+', '_', article_id)
        safe_article_id = quote(safe_article_id, safe='_')
        article_uri = self.make_instance_uri(self.NCO, "NewsItem", safe_article_id)
        self.abox_graph.add((article_uri, RDF.type, self.NCO.NewsItem))
        return article_uri

    def _add_news_source(self, article_uri, article_data):
        source_info = article_data.get("source_info")
        if source_info and isinstance(source_info, dict):
            publisher_name = source_info.get("name")
            source_domain = source_info.get("domain")
            if publisher_name and source_domain:
                publisher_uri, news_source_uri = self._add_publisher_and_source(publisher_name, source_domain)
                self.abox_graph.add((article_uri, self.NCO.hasNewsSource, news_source_uri))
                return news_source_uri
        else:
            print(f"Warning: Article {article_uri} missing source_info. NewsItem must have a NewsSource.")
        return None

    def _add_topic(self, article_uri, article_id):
        topic_uri = self._topic_from_article(article_id)
        if topic_uri:
            self.abox_graph.add((article_uri, self.NCO.hasTopic, topic_uri))
            self.abox_graph.add((topic_uri, self.NCO.topicInNewsItem, article_uri))
        return topic_uri

    def _add_article_metadata(self, article_uri, article_data):
        title = article_data.get("title")
        if title:
            self.abox_graph.add((article_uri, self.NCO.hasTitle, Literal(title)))
            self.abox_graph.add((article_uri, RDFS.label, Literal(title)))
        body = article_data.get("body")
        if body:
            self.abox_graph.add((article_uri, self.NCO.hasText, Literal(body)))
        source_link = article_data.get("source_link")
        if source_link:
            self.abox_graph.add((article_uri, self.NCO.hasURL, Literal(source_link, datatype=XSD.anyURI)))
        date_str = article_data.get("date")
        if date_str:
            try:
                safe_article_id = re.sub(r'\s+', '_', article_data.get("article_id"))
                safe_article_id = quote(safe_article_id, safe='_')
                instant_id = f"{safe_article_id}"
                instant_uri = self.make_instance_uri(self.NCO, "Instant", instant_id)
                self.abox_graph.add((instant_uri, RDF.type, self.TIME.Instant))
                formatted_date_str = date_str
                if ' ' in date_str and 'T' not in date_str.split(' ')[0]:
                    parts = date_str.split(' ', 1)
                    if len(parts) == 2:
                        formatted_date_str = f"{parts[0]}T{parts[1]}"
                self.abox_graph.add((instant_uri, self.TIME.inXSDDateTimeStamp, Literal(formatted_date_str, datatype=XSD.dateTimeStamp)))
                self.abox_graph.add((article_uri, self.TIME.hasTime, instant_uri))
            except Exception as e:
                print(f"Warning: Could not process date '{date_str}' for article {article_uri}: {e}")

    def _add_agents(self, agents_info):
        agent_uris = {}
        for agent_name, agent_data in agents_info.items():
            agent_uris[agent_name] = self._process_agent(agent_name, agent_data)
        return agent_uris

    def _add_claims(self, article_uri, claims, agent_uris, topic_uri, article_id, agents_info):
        # --- Sequential claim counter per article ---
        if not hasattr(self, '_claim_counter_per_article'):
            self._claim_counter_per_article = {}
        claim_counter = self._claim_counter_per_article.setdefault(str(article_uri), 0)
        for claim_data in claims:
            utterance_text = claim_data.get("utterance_text", "")
            agent_name = claim_data.get("agent_name", "")
            claim_key = (utterance_text, str(article_uri), agent_name)
            if claim_key in self.claim_uris:
                claim_uri = self.claim_uris[claim_key]
            else:
                claim_id = hashlib.md5((utterance_text + str(article_uri) + agent_name).encode("utf-8")).hexdigest()
                claim_uri = self.make_instance_uri(self.NCO, "Claim", claim_id)
                self.claim_uris[claim_key] = claim_uri
                self.abox_graph.add((claim_uri, RDF.type, self.NCO.Claim))
                # --- Progressive numerical LABEL for claim in article ---
                claim_counter += 1
                label = f"Claim {claim_counter} in {article_id}"
                self.abox_graph.add((claim_uri, RDFS.label, Literal(label)))
            self._claim_counter_per_article[str(article_uri)] = claim_counter
            self.abox_graph.add((claim_uri, self.NCO.claimInNewsItem, article_uri))
            self.abox_graph.add((article_uri, self.NCO.hasClaim, claim_uri))
            if topic_uri:
                self.abox_graph.add((claim_uri, self.NCO.concernsTopic, topic_uri))
            self._add_agent_component(claim_uri, agent_name, agent_uris, article_id, agents_info)
            self._add_utterance(claim_uri, claim_data)

    def _add_agent_component(self, claim_uri, agent_name, agent_uris, article_id, agents_info):
        agent_uri = agent_uris.get(agent_name)
        if agent_uri:
            agent_data = agents_info.get(agent_name, {})
            taxonomy_info = agent_data.get("taxonomy_info", {})
            role_label = taxonomy_info.get("classification", "AgentRole")
            # Unique key for AgentComponent based solely on agent and role
            ac_key = (agent_name, role_label)
            if ac_key not in self.agent_component_uris:
                #agent_component_id = hashlib.md5(f"{agent_name}|{role_label}".encode("utf-8")).hexdigest()
                agent_component_id = re.sub(r'\s+', '_', f"{agent_name}_{role_label}")  # Removes spaces and replaces with underscore
                agent_component_uri = self.make_instance_uri(self.NCO, "AgentComponent", agent_component_id)
                self.agent_component_uris[ac_key] = agent_component_uri
                self.abox_graph.add((agent_component_uri, RDF.type, self.NCO.AgentComponent))
                self.abox_graph.add((agent_component_uri, self.NCO.hasAgent, agent_uri))
                component_label = f"{agent_name} as {role_label}"
                self.abox_graph.add((agent_component_uri, RDFS.label, Literal(component_label)))
                # --- CREATE OR RETRIEVE A SINGLE ENTITY FOR EACH UNIQUE ROLE ---
                if role_label not in self.role_uris:
                    #role_id = hashlib.md5(role_label.encode("utf-8")).hexdigest()
                    role_id = re.sub(r'\s+', '_', role_label)  # Removes spaces and replaces with underscore
                    role_uri = self.make_instance_uri(self.NCO, "AgentRole", role_id)
                    self.abox_graph.add((role_uri, RDF.type, self.NCO.AgentRole))
                    self.abox_graph.add((role_uri, RDFS.label, Literal(role_label)))
                    self.role_uris[role_label] = role_uri
                else:
                    role_uri = self.role_uris[role_label]
                self.abox_graph.add((agent_component_uri, self.NCO.hasAgentRole, role_uri))
                self.abox_graph.add((agent_uri, self.NCO.hasAgentRole, role_uri))
            else:
                agent_component_uri = self.agent_component_uris[ac_key]
        # Always link AgentComponent to claim
        self.abox_graph.add((claim_uri, self.NCO.hasAgentComponent, agent_component_uri))

    def _add_utterance(self, claim_uri, claim_data):
        existing_utterances = list(self.abox_graph.objects(claim_uri, self.NCO.hasUtterance))
        if existing_utterances:
            print(f"[DEBUG] Claim {claim_uri} already has an utterance linked: {existing_utterances[0]}. Not adding another.")
            return
        utterance_text = claim_data.get("utterance_text", "")
        utterance_id = hashlib.md5((claim_data.get("source_context", "") + str(claim_uri)).encode("utf-8")).hexdigest()
        utterance_uri = self.make_instance_uri(self.NCO, "Utterance", utterance_id)
        self.abox_graph.add((utterance_uri, RDF.type, self.NCO.Utterance))
        if utterance_text:
            self.abox_graph.add((utterance_uri, self.NCO.hasText, Literal(utterance_text)))
        if claim_data.get("source_context"):
            self.abox_graph.add((utterance_uri, self.NCO.hasTextContext, Literal(claim_data["source_context"])))
        if claim_data.get("utterance_type"):
            utterance_type_map = {
                "direct": self.NCO.directUtterance,
                "indirect": self.NCO.indirectUtterance,
                "partially_direct": self.NCO.partiallyDirectUtterance
            }
            stype = claim_data["utterance_type"].lower().replace("-", "_")
            utterance_type_uri = utterance_type_map.get(stype)
            if utterance_type_uri:
                self.abox_graph.add((utterance_uri, self.NCO.hasUtteranceType, utterance_type_uri))
        self.abox_graph.add((claim_uri, self.NCO.hasUtterance, utterance_uri))

    # --- Utility methods to inspect the TBox (unchanged) ---
    def list_classes(self, from_namespace: str | None = None) -> list[URIRef]:
        classes = set()
        query_types = [OWL.Class, RDFS.Class]
        for class_type in query_types:
            for s, p, o in self.tbox_graph.triples((None, RDF.type, class_type)):
                if isinstance(s, URIRef):
                    if from_namespace is None or str(s).startswith(from_namespace):
                        classes.add(s)
        return sorted(list(classes))

    def list_object_properties(self, from_namespace: str | None = None) -> list[URIRef]:
        properties = set()
        for s, p, o in self.tbox_graph.triples((None, RDF.type, OWL.ObjectProperty)):
            if isinstance(s, URIRef):
                if from_namespace is None or str(s).startswith(from_namespace):
                    properties.add(s)
        return sorted(list(properties))

    def list_datatype_properties(self, from_namespace: str | None = None) -> list[URIRef]:
        properties = set()
        for s, p, o in self.tbox_graph.triples((None, RDF.type, OWL.DatatypeProperty)):
            if isinstance(s, URIRef):
                if from_namespace is None or str(s).startswith(from_namespace):
                    properties.add(s)
        return sorted(list(properties))

    def get_property_details(self, property_uri: URIRef) -> dict:
        details = {"domain": [], "range": []}
        for s, p, o in self.tbox_graph.triples((property_uri, RDFS.domain, None)):
            details["domain"].append(o)
        for s, p, o in self.tbox_graph.triples((property_uri, RDFS.range, None)):
            details["range"].append(o)
        return details

    def get_class_details(self, class_uri: URIRef) -> dict:
        details = {"superclasses": [], "labels": [], "comments": []}
        for s, p, o in self.tbox_graph.triples((class_uri, RDFS.subClassOf, None)):
            if isinstance(o, URIRef):
                details["superclasses"].append(o)
        for s, p, o in self.tbox_graph.triples((class_uri, RDFS.label, None)):
            details["labels"].append(str(o))
        for s, p, o in self.tbox_graph.triples((class_uri, RDFS.comment, None)):
            details["comments"].append(str(o))
        return details


