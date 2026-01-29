from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# LLM provider functions
def get_google_llm(model_name="gemma-3-27b-it", temperature=0):
    """Get a Google Generative AI LLM instance."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
    )

def get_openai_llm(model_name="gpt-4o-mini", temperature=0, api_key=None):
    """Get an OpenAI LLM instance."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

def get_openrouter_llm(model_name="deepseek/deepseek-chat-v3-0324:free", temperature=0, api_key=None):
    """Get an OpenRouter LLM instance."""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key or os.getenv("OPEN_ROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )

# Default LLM getter that can be modified based on preference
def get_llm(model_name=None, temperature=0):
    """Get the default LLM instance based on initialized API keys."""
    kwargs = {"temperature": temperature}
    if model_name:
        kwargs["model_name"] = model_name

    if os.getenv("OPEN_ROUTER_API_KEY"):
        return get_openrouter_llm(**kwargs)
    elif os.getenv("OPENAI_API_KEY"):
        return get_openai_llm(**kwargs)
    elif os.getenv("GOOGLE_API_KEY"):
        return get_google_llm(**kwargs)
    
    return get_openrouter_llm(**kwargs)

# Load prompt using a YAML file
def load_prompt_by_name(target_name, prompt_file="prompts"):
    """
    Load a prompt by name from a YAML file.
    
    Args:
        target_name: Name of the prompt to load
        prompt_file: Name of the YAML file (without .yaml extension) in the Prompts directory
                    Default is "prompts" for backward compatibility
    
    Returns:
        The content of the prompt if found, None otherwise
    """
    base_dir = os.path.dirname(__file__)  # directory of this utility file
    file_path = os.path.join(base_dir, "Prompts", f"{prompt_file}.yaml")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Expecting a list of dicts with 'name' and 'content' keys
        for item in data:
            if item.get('name') == target_name:
                return item.get('content')
        
        print(f"Warning: Prompt '{target_name}' not found in {file_path}")
        return None
    except FileNotFoundError:
        print(f"Error: Prompt file '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error loading prompt from {file_path}: {e}")
        return None