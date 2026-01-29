import os
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path
import ftfy  
import unicodedata


# Try to import embedding models - with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class SemanticNewsDB:
    """
    A semantic database for news articles that supports vector search.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2", 
                 data_dir: str = "NewsArticles",
                 db_path: Optional[str] = None):
        """
        Initialize the semantic database.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            data_dir: Directory containing news articles
            db_path: Path to save/load the database
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        
        # Calculate absolute path dynamically
        repo_root = Path(__file__).resolve().parent.parent
        self.db_path = Path(db_path) if db_path else repo_root / "SemanticArticlesDB/news_semantic_db.json"
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            # Check if CUDA is available for GPU acceleration
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Loading model on {device}")
            self.model = SentenceTransformer(model_name, device=device)
        else:
            print("Warning: sentence-transformers not available. Please install with: pip install sentence-transformers")
            self.model = None
            
        # Initialize database
        self.articles = []
        self.embeddings = None
        self.index = None

        # Inizalize database
        self.initialize_db()
        
    def load_articles(self) -> None:
        """Load articles from the data directory."""
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} does not exist.")
            return
            
        # Load all JSON files in the directory and subdirectories
        for folder_path in self.data_dir.glob("*"):
            if folder_path.is_dir():
                folder_name = folder_path.name
                
                for file_path in folder_path.glob("*.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # Create a structured article dictionary
                            article = {}
                            
                            # Extract file name without extension and combine with folder name
                            file_name = file_path.name.replace('.json', '')
                            article['article_id'] = f"{file_name}_{folder_name}"
                            
                            article['title'] = self.preprocess_text(data.get('title', 'No Title'))
                            article['body'] = self.preprocess_text(data.get('body', ''))
                            
                            # Handle nested fields with safe access
                            if 'summary' in data and 'sentences' in data['summary']:
                                # Clean summary sentences
                                article['summary'] = [ftfy.fix_text(s) for s in data['summary']['sentences']]
                            else:
                                article['summary'] = []
                                
                            if 'source' in data and 'name' in data['source']:
                                article['source'] = data['source']['name']
                            else:
                                article['source'] = 'Unknown'
                                
                            if 'author' in data and 'name' in data['author']:
                                article['author'] = data['author']['name']
                            else:
                                article['author'] = 'Unknown'
                                
                            article['date'] = data.get('published_at', '')
                            
                            if 'links' in data and 'permalink' in data['links']:
                                article['source_link'] = data['links']['permalink']
                            else:
                                article['source_link'] = ''

                            if 'source' in data:
                                source_data = {
                                    'name': data['source'].get('name', ''),
                                    'domain': data['source'].get('domain', ''),
                                    'description': data['source'].get('description', ''),
                                    'home_page_url': data['source'].get('home_page_url', ''),
                                    'locations': data['source'].get('locations', []),
                                }

                            article['source_info'] = source_data    

                            
                            # Add original file path as metadata
                            article['file_path'] = str(file_path)
                            
                            # Add the article to our collection
                            self.articles.append(article)
                            
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                
        print(f"Loaded {len(self.articles)} articles from {self.data_dir}")
    
    def create_embeddings(self) -> None:
        """Create embeddings for all loaded articles."""
        if not self.model:
            print("Embedding model not available.")
            return
            
        if not self.articles:
            print("No articles loaded. Call load_articles() first.")
            return
            
        # Extract text from articles - separate title and body
        title_texts = []
        body_texts = []
        combined_texts = []
        
        for article in self.articles:
            # Extract title
            title = article.get('title', '')
            title_texts.append(title)
            
            # Extract body
            if 'body' in article and article['body']:
                body = article['body']
            elif 'content' in article and article['content']:
                body = article['content']
            else:
                # Fallback to other fields if body/content not available
                body = " ".join(str(v) for k, v in article.items() 
                              if k not in ['title', 'file_path', 'article_id'] and not isinstance(v, (dict, list)))
            
            body_texts.append(body)
            
            # Create combined text for overall embedding
            combined_texts.append(f"{title}. {body}")
        
        # Create embeddings for titles, bodies, and combined texts
        print("Creating title embeddings...")
        title_embeddings = self.model.encode(title_texts, show_progress_bar=True)
        
        print("Creating body embeddings...")
        body_embeddings = self.model.encode(body_texts, show_progress_bar=True)
        
        print("Creating combined embeddings...")
        self.embeddings = self.model.encode(combined_texts, show_progress_bar=True)
        
        # Store the specialized embeddings
        self.title_embeddings = title_embeddings
        self.body_embeddings = body_embeddings
        
        # Create FAISS index if available
        if FAISS_AVAILABLE and len(self.embeddings) > 0:
            # Main index for combined embeddings
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(self.embeddings).astype('float32'))
            
            # Create separate indices for title and body
            self.title_index = faiss.IndexFlatL2(dimension)
            self.title_index.add(np.array(title_embeddings).astype('float32'))
            
            self.body_index = faiss.IndexFlatL2(dimension)
            self.body_index.add(np.array(body_embeddings).astype('float32'))
            
            print(f"Created FAISS indices with {len(self.embeddings)} embeddings for each type")
        else:
            print("FAISS not available or no embeddings created.")

    
    def save_database(self, path: Optional[str] = None) -> None:
        """Save the database to disk."""
        save_path = Path(path) if path else self.db_path
        
        # Prepare data for saving
        data = {
            'model_name': self.model_name,
            'articles': self.articles
        }
        
        # Save combined embeddings if available
        if self.embeddings is not None:
            data['embeddings'] = self.embeddings.tolist()
        
        # Save title embeddings if available
        if hasattr(self, 'title_embeddings') and self.title_embeddings is not None:
            data['title_embeddings'] = self.title_embeddings.tolist()
        
        # Save body embeddings if available
        if hasattr(self, 'body_embeddings') and self.body_embeddings is not None:
            data['body_embeddings'] = self.body_embeddings.tolist()
        
        # Save to file
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
        print(f"Database saved to {save_path}")
    
    def load_database(self, path: Optional[str] = None) -> None:
        """Load the database from disk."""
        load_path = Path(path) if path else self.db_path
        
        if not load_path.exists():
            print(f"Database file {load_path} does not exist.")
            return
            
        # Load from file
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Restore data
        self.model_name = data.get('model_name', self.model_name)
        self.articles = data.get('articles', [])
        
        # Restore combined embeddings if available
        if 'embeddings' in data and data['embeddings']:
            self.embeddings = np.array(data['embeddings'])
            
        # Restore title embeddings if available
        if 'title_embeddings' in data and data['title_embeddings']:
            self.title_embeddings = np.array(data['title_embeddings'])
            
        # Restore body embeddings if available
        if 'body_embeddings' in data and data['body_embeddings']:
            self.body_embeddings = np.array(data['body_embeddings'])
            
        # Recreate FAISS indices if available
        if FAISS_AVAILABLE:
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                dimension = self.embeddings.shape[1]
                
                # Main index for combined embeddings
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(np.array(self.embeddings).astype('float32'))
                
                # Create title index if title embeddings are available
                if hasattr(self, 'title_embeddings') and self.title_embeddings is not None:
                    self.title_index = faiss.IndexFlatL2(dimension)
                    self.title_index.add(np.array(self.title_embeddings).astype('float32'))
                
                # Create body index if body embeddings are available
                if hasattr(self, 'body_embeddings') and self.body_embeddings is not None:
                    self.body_index = faiss.IndexFlatL2(dimension)
                    self.body_index.add(np.array(self.body_embeddings).astype('float32'))
                
        print(f"Loaded database from {load_path} with {len(self.articles)} articles")


    def get_articles_by_ids(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve articles by their IDs.    
        Args:
            article_ids: List of article IDs
    
        Returns:
            List of articles        
        """
        if not self.articles:
            print("No articles loaded. Call load_articles() first.")
            return []
        
        # Create a dictionary for faster lookup
        articles_dict = {article.get('article_id', ''): article for article in self.articles}
        
        # Retrieve articles by ID
        found_articles = []
        missing_ids = []
        
        for article_id in article_ids:
            if article_id in articles_dict:
                found_articles.append(articles_dict[article_id])
            else:
                missing_ids.append(article_id)
                
        if missing_ids:
            print(f"Warning: Could not find articles with these IDs: {', '.join(missing_ids)}")
            
        return found_articles


    def search(self, query: str, top_k: int = 5, search_type: str = "combined") -> List[Dict[str, Any]]:
        """
        Search for articles semantically similar to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            search_type: Type of search to perform ("combined", "title", or "body")
            
        Returns:
            List of articles with similarity scores
        """
        if not self.model:
            print("Embedding model not available.")
            return []
        
        # Clean the query text
        query = ftfy.fix_text(query)
        
        # Validate search_type
        valid_types = ["combined", "title", "body"]
        if search_type not in valid_types:
            print(f"Invalid search_type: {search_type}. Using 'combined' instead.")
            search_type = "combined"
            
        # Select the appropriate embeddings and index based on search_type
        if search_type == "title" and hasattr(self, 'title_embeddings') and self.title_embeddings is not None:
            embeddings = self.title_embeddings
            index = self.title_index if hasattr(self, 'title_index') else None
        elif search_type == "body" and hasattr(self, 'body_embeddings') and self.body_embeddings is not None:
            embeddings = self.body_embeddings
            index = self.body_index if hasattr(self, 'body_index') else None
        else:
            # Default to combined embeddings
            embeddings = self.embeddings
            index = self.index
            
        if embeddings is None or len(embeddings) == 0:
            print(f"No {search_type} embeddings available. Call create_embeddings() first.")
            return []
            
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search using FAISS if available
        if FAISS_AVAILABLE and index is not None:
            distances, indices = index.search(
                np.array([query_embedding]).astype('float32'), 
                min(top_k, len(self.articles))
            )
            results = []
            for i, idx in enumerate(indices[0]):
                article = self.articles[idx].copy()
                article['similarity_score'] = float(1.0 - distances[0][i] / 100.0)  # Convert distance to similarity
                article['search_type'] = search_type  # Add search type to results
                results.append(article)
            return results
        else:
            # Fallback to manual search
            similarities = []
            for i, embedding in enumerate(embeddings):
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            results = []
            for idx, score in similarities[:top_k]:
                article = self.articles[idx].copy()
                article['similarity_score'] = float(score)
                article['search_type'] = search_type  # Add search type to results
                results.append(article)
            return results

    def initialize_db(self) -> None:
        """
        Initialize the database by either loading an existing one or creating a new one.
        This function checks if the database file exists and loads it if it does,
        otherwise it creates a new database by loading articles, creating embeddings,
        and saving the database.
        """
        # Check if database file exists
        if os.path.exists(self.db_path):
            # Load existing database
            print(f"Loading existing database from {self.db_path}")
            self.load_database()
        else:
            # Create new database
            print(f"Database not found at {self.db_path}. Creating new database...")
            self.load_articles()
            self.create_embeddings()
            self.save_database()
            print("Database initialization complete.")
    
    def integrate_with_langgraph(self, state: Dict[str, Any], query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Integrate with LangGraph by updating the state with relevant articles.
        
        Args:
            state: The current LangGraph state
            query: The search query
            top_k: Number of results to include
            
        Returns:
            Updated state with relevant articles
        """
        # Search for relevant articles
        relevant_articles = self.search(query, top_k=top_k)
        
        # Create a copy of the state
        updated_state = state.copy()
        
        # Add relevant articles to the state
        updated_state['relevant_articles'] = relevant_articles
        
        # If there's a current article being processed, add it to the state
        if relevant_articles:
            updated_state['current_article'] = relevant_articles[0]['text'] if 'text' in relevant_articles[0] else \
                relevant_articles[0].get('content', '')
        
        return updated_state

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text using ftfy with enhanced options.
        
        Args:
            text: The text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Fix encoding issues
        cleaned_text = ftfy.fix_text(text)
        #cleaned_text = unicodedata.normalize('NFC', cleaned_text)
        
        # Additional preprocessing steps could be added here
        # For example, normalizing whitespace
        #cleaned_text = " ".join(cleaned_text.split())
        
        return cleaned_text

# Example usage
if __name__ == "__main__":
    # Initialize the database
    db = SemanticNewsDB()
    
    # Check if database file exists
    if os.path.exists(db.db_path):
        # Load existing database
        db.load_database()
    else:
        # Create new database
        db.load_articles()
        db.create_embeddings()
        db.save_database()
    
    # Example search
    results = db.search("climate change policy", top_k=3)
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result['similarity_score']:.4f}):")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"File: {result.get('file_path', 'N/A')}")
        print("-" * 50)