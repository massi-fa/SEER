import argparse
import sys
import os
import json
import subprocess
import importlib.util
from datetime import datetime

# --- Environment Management Functions ---

def parse_requirements():
    """Parse requirements.txt to get a list of required packages and their likely import names."""
    req_file_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(req_file_path):
        # Fallback to hardcoded core list if requirements.txt is missing
        return {
            'langgraph': 'langgraph',
            'spacy': 'spacy',
            'sentence-transformers': 'sentence_transformers',
            'faiss-cpu': 'faiss',
            'rdflib': 'rdflib',
            'tqdm': 'tqdm'
        }
        
    requirements = {}
    with open(req_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove version specifiers (e.g., package==1.0.0 -> package)
            # Also handle <, >, >=, etc.
            import re
            app_name = re.split(r'[=<>!]', line)[0].strip()
            
            # Map PyPI package names to Import names
            # Most packages use underscores instead of hyphens (e.g. 'sentence-transformers' -> 'sentence_transformers')
            if 'faiss' in app_name: import_name = 'faiss'
            elif app_name == 'python-dotenv': import_name = 'dotenv'
            elif app_name == 'pyyaml': import_name = 'yaml'
            else: import_name = app_name.replace('-', '_')
            
            requirements[app_name] = import_name
            
    return requirements

def check_imports():
    """Check if dependencies from requirements.txt are installed."""
    required_packages = parse_requirements()
    missing = []
    
    for pkg_name, import_name in required_packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pkg_name)
            
    return missing

def manage_environment_setup():
    """
    Ensures the standard environment ('venv_claim_agent') exists and is used.
    Returns the path to the venv python if a switch is needed, or None if current is correct.
    """
    venv_dir = "venv_claim_agent"
    cwd = os.getcwd()
    
    # Expected path for the venv python
    if os.name == 'nt':
        venv_python = os.path.join(cwd, venv_dir, 'Scripts', 'python.exe')
    else:
        venv_python = os.path.join(cwd, venv_dir, 'bin', 'python')
        
    # 1. Check if we are already running inside the target venv
    try:
        # Compare absolute paths to determine if we are active
        current_exe = os.path.abspath(sys.executable)
        target_exe = os.path.abspath(venv_python)
        
        # On Windows, paths are case-insensitive
        if os.name == 'nt':
            current_exe = current_exe.lower()
            target_exe = target_exe.lower()
            
        if current_exe == target_exe:
            # We are in the correct env. Verify dependencies are actually installed.
            missing = check_imports()
            if missing:
                print(f"⚠️  Missing packages in {venv_dir}: {', '.join(missing)}")
                print("📦 Auto-installing missing dependencies...")
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
                except subprocess.CalledProcessError:
                    print("❌ Failed to install dependencies.")
                    sys.exit(1)
            return None # No switch needed, stay here.
    except Exception:
        pass # Comparison failed, assume not in venv, proceed to switch.

    # 2. If we are here, we are NOT in the venv. 
    # We must switch to venv_claim_agent (creating it if missing).
    
    if not os.path.exists(venv_python):
        print(f"🚀 Initializing standardized environment: '{venv_dir}'")
        try:
            # Create venv using current python
            print("   • Creating virtual environment...")
            subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])
            
            # Upgrade pip and install deps in the NEW env
            print("   • Installing dependencies from requirements.txt...")
            subprocess.check_call([venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'])
            subprocess.check_call([venv_python, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Environment setup complete.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to setup environment: {e}")
            sys.exit(1)
    
    # Return the path to switch to
    return venv_python

# --- Main Pipeline Logic ---

def check_env_setup():
    """Checks for .env file and valid keys. Prompts user to add key if missing."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    valid_keys = ['OPEN_ROUTER_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY']
    has_valid_key = False

    # Check if file exists and has at least one valid key
    if os.path.exists(env_path):
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        if k.strip() in valid_keys and v.strip():
                            has_valid_key = True
                            return # Found a valid key, proceed
        except Exception:
            pass # Error reading, proceed to prompt

    print("\n" + "="*60)
    print("🔑 API KEY CONFIGURATION")
    print("="*60)
    print("No valid API key found in .env.")
    print("To use the SEER Framework, you need to configure at least one LLM provider.")
    print("\nAvailable Providers:")
    print("  1. OpenRouter API (DeepSeek, Llama, etc.) [Recommended]")
    print("  2. Google API (Gemini models)")
    print("  3. OpenAI API (GPT models)")
    print("="*60)
    
    mapping = {'1': 'OPEN_ROUTER_API_KEY', '2': 'GOOGLE_API_KEY', '3': 'OPENAI_API_KEY'}
    while True:
        sel = input("\n👉 Select a provider (1-3): ").strip()
        if sel in mapping:
            key_name = mapping[sel]
            break
        print("❌ Invalid selection. Please enter 1, 2, or 3.")
            
    api_key = input(f"👉 Please paste your {key_name} here: ").strip()
    if not api_key:
        print("\n❌ No API Key provided. Exiting...")
        sys.exit(1)
        
    try:
        # Append to existing file or create new
        mode = 'a' if os.path.exists(env_path) else 'w'
        with open(env_path, mode, encoding='utf-8') as f:
            if mode == 'a': 
                f.write("\n") # Ensure newline separator
            f.write(f"{key_name}={api_key}\n")
            
        print(f"\n✅ Configuration saved to: {env_path}")
        print("   You can edit this file manually later if needed.")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Failed to write .env file: {e}")
        sys.exit(1)

def run_pipeline_args():
    # Setup argument parser first - no heavy imports yet
    parser = argparse.ArgumentParser(
        description="SEER (Semantic Extraction & Enrichment Reasoning): Framework for Media Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core Pipeline Configuration
    parser.add_argument("--model-claims", type=str, nargs='+', default=["google/gemini-2.5-flash-lite"], 
                        help="LLM model(s) to use for claim extraction. Pass multiple for V2.")
    parser.add_argument("--model-entities", type=str, default="google/gemini-2.5-flash-lite", 
                        help="LLM model to use for entity analysis/linking")
    parser.add_argument("--temp", type=float, default=0.0, 
                        help="Temperature for LLM generation (0.0 = deterministic)")
    
    # Extraction Version & Performance
    parser.add_argument("--v2", action="store_true", 
                        help="Use Claim Extraction V2.0 (Multi-LLM Consensus) instead of V1.0")
    parser.add_argument("--parallel", action="store_true", 
                        help="Enable parallel processing of articles")
    parser.add_argument("--workers", type=int, default=None, 
                        help="Maximum number of parallel workers (default: auto)")

    # Execution Mode (Search vs Direct IDs)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, 
                       help="Semantic search query (e.g., 'climate change policies')")
    group.add_argument("--ids", nargs="+", 
                       help="List of specific article IDs to process")
    
    parser.add_argument("--num", type=int, default=5, 
                        help="Number of articles to retrieve/process (when using --query)")

    # Output
    parser.add_argument("--output", type=str, default="summary_pipeline.json",
                        help="Path to save the final JSON output summary")

    # Parse args
    args = parser.parse_args()

    # NOW we import the heavy stuff, after args are parsed and help is handled
    try:
        from FullPipeline import FullPipeline
    except ImportError as e:
        print(f"\n❌ Critical Error: Failed to import FullPipeline even after checks.")
        print(f"   Detail: {e}")
        print(f"   Current Python: {sys.executable}")
        sys.exit(1)

    print("\n" + "="*60)
    print("🚀 INITIALIZING SEER FRAMEWORK")
    print("="*60)
    print(f"Configurations:")
    print(f"  • Claim Model:    {args.model_claims}")
    print(f"  • Entity Model:   {args.model_entities}")
    print(f"  • Pipeline Ver:   {'v2.0 (Consensus)' if args.v2 else 'v1.0 (Standard)'}")
    print(f"  • Processing:     {'Parallel' if args.parallel else 'Sequential'}")
    if args.query:
        print(f"  • Mode:           Semantic Search (Query: '{args.query}', Limit: {args.num})")
    else:
        print(f"  • Mode:           Direct ID Processing ({len(args.ids)} articles)")
    print("="*60 + "\n")

    # Initialize Pipeline
    pipeline = FullPipeline(
        model_claim_extraction=args.model_claims,
        model_entity_analysis=args.model_entities,
        temperature=args.temp,
        claim_extraction_version=2 if args.v2 else 1,
        parallel=args.parallel,
        max_workers=args.workers
    )

    # Run Pipeline
    start_time = datetime.now()
    try:
        if args.ids:
            results = pipeline.run(article_ids=args.ids)
        else:
            results = pipeline.run(query=args.query, num_articles=args.num)
        
        # Save condensed results
        if results and 'summary' in results:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results['summary'], f, indent=2, ensure_ascii=False)
            print(f"\n✅ Pipeline completed successfully.")
            print(f"💾 Summary saved to: {args.output}")
            
            # Print basic stats
            summary = results['summary']
            print(f"\n📊 Statistics:")
            print(f"   - Articles Processed: {summary.get('num_articles_processed', 0)}")
            # Handle potential varying summary structures
            claims_count = summary.get('num_claims', summary.get('total_claims', 0))
            print(f"   - Claims Extracted:   {claims_count}")
            print(f"   - Knowledge Graph:    Generated in 'KnowledgeGraph/' folder")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        duration = datetime.now() - start_time
        print(f"\n⏱️  Total Execution Time: {duration}")

if __name__ == "__main__":
    # Special flag for internal validation logic used by validate_python_env()
    if "--check-env" in sys.argv:
        missing = check_imports()
        if missing:
            sys.exit(1) # Failure
        else:
            sys.exit(0) # Success

    # 1. Environment Check
    # Simple heuristic to skip env check if help is requested, so we show help fast
    if "--help" in sys.argv or "-h" in sys.argv:
        run_pipeline_args() # This will print help and exit
        sys.exit(0)

    # 2. Manage Environment
    # This checks for imports, and if missing, offers to switch/create env
    new_python_exe = manage_environment_setup()
    
    # 3. Handle Environment Switch
    if new_python_exe:
        # If a different python interpreter is returned, we need to restart the script with it
        # We pass exactly the same arguments, just with the new interpreter
        print(f"🔄 Restarting pipeline with configured environment...")
        print(f"   Interpreter: {new_python_exe}")
        
        try:
            # Using subprocess.call to replace current process effectively
            # ADDED '-s' to avoid importing packages from user site directory (AppData) 
            # which caused the binary incompatibility/DLL mix-up.
            cmd = [new_python_exe, '-s'] + sys.argv
            ret_code = subprocess.call(cmd)
            sys.exit(ret_code)
        except Exception as e:
            print(f"❌ Failed to restart with new environment: {e}")
            sys.exit(1)

    # 4. Run Main Application
    # If we get here, we are running in the correct environment (or user decliened to switch)
    check_env_setup()
    run_pipeline_args()

