# launch.py - Application Launcher with CLI Support
import os
import sys
import argparse
from pathlib import Path
import importlib
import json

# Corrected: Removed the import of EmpowerFinApplication as it does not exist
# from main_application import EmpowerFinApplication 

# Import necessary data models for demos
from data_models import EnhancedTransaction, ConversationEntry, FinancialHealthMetrics # Assuming these are needed for demos

# Import specific agents needed for demo functions
# These were missing or implicitly assumed but not explicitly imported
from health_analyzer_agent import LLMHealthAnalyzer
from categorization_agent import TransactionCategorizationAgent
from file_processor_agent import FileProcessorAgent
from conversational_agent import ConversationalAgent

class ApplicationLauncher:
    """Advanced launcher for financial intelligence platform"""
    
    def __init__(self):
        # Added 'description' to agent info for better help messages
        self.available_agents = {
            'health_analyzer': {
                'module': 'health_analyzer_agent',
                'class': 'LLMHealthAnalyzer',
                'description': 'Analyze overall financial health'
            },
            'transaction_categorizer': {
                'module': 'categorization_agent',
                'class': 'TransactionCategorizationAgent',
                'description': 'Categorize transactions into spending types'
            },
            'file_processor': {
                'module': 'file_processor_agent',
                'class': 'FileProcessorAgent',
                'description': 'Process and parse bank statement files'
            },
            'conversational_advisor': {
                'module': 'conversational_agent',
                'class': 'ConversationalAgent',
                'description': 'Engage with an AI financial advisor'
            }
        }
        
    def launch_full_app(self, demo=False):
        """Launch the full Streamlit application"""
        try:
            # Streamlit's internal CLI main function is used to launch the app
            from streamlit.web.cli import main as st_main
            sys.argv = ["streamlit", "run", "main_application.py"]
            if demo:
                # Set environment variable for demo mode, to be read by main_application.py
                os.environ["EMPOWERFIN_DEMO"] = "true"
            st_main()
        except ImportError:
            print("? Streamlit not installed. Install with: pip install streamlit")
            sys.exit(1)
        except Exception as e:
            print(f"? Error launching full application: {e}")
            sys.exit(1)
            
    def launch_agent_demo(self, agent_name):
        """Launch a specific agent in demo mode"""
        if agent_name not in self.available_agents:
            print(f"? Unknown agent: {agent_name}")
            print("Available agents:", list(self.available_agents.keys()))
            return
            
        agent_info = self.available_agents[agent_name]
        try:
            # Dynamically import the module and get the class
            module = importlib.import_module(agent_info['module'])
            agent_class = getattr(module, agent_info['class'])
            
            # Create agent instance
            agent = agent_class()
            
            print(f"\n?? Running {agent_name} in demo mode\n")
            
            # Run agent-specific demo function
            if agent_name == 'health_analyzer':
                self._demo_health_analysis(agent)
            elif agent_name == 'transaction_categorizer':
                self._demo_categorization(agent)
            elif agent_name == 'file_processor':
                self._demo_file_processing(agent)
            elif agent_name == 'conversational_advisor':
                self._demo_conversation(agent)
                
        except Exception as e:
            print(f"? Error launching agent demo: {e}")
            # Optional: print traceback for debugging
            # import traceback
            # print(traceback.format_exc())
    
    def _demo_health_analysis(self, agent: LLMHealthAnalyzer):
        """Run health analyzer demo"""
        print("?? Running Health Analyzer Demo...")
        sample_tx = [
            EnhancedTransaction(date="2024-01-15", description="Grocery Store", amount=-75.50),
            EnhancedTransaction(date="2024-01-16", description="Restaurant Meal", amount=-120.00),
            EnhancedTransaction(date="2024-01-17", description="Salary Deposit", amount=3000.00),
            EnhancedTransaction(date="2024-01-18", description="Electric Bill", amount=-120.00),
            EnhancedTransaction(date="2024-01-19", description="Amazon Purchase", amount=-80.25)
        ]
        
        # Analyze financial health and print the dictionary representation
        metrics = agent.analyze_financial_health(sample_tx)
        print("?? Financial Health Analysis:")
        print(json.dumps(metrics.to_dict(), indent=2))
    
    def _demo_categorization(self, agent: TransactionCategorizationAgent):
        """Run categorization agent demo"""
        print("?? Running Transaction Categorization Demo...")
        sample_tx = [
            EnhancedTransaction(date="2024-01-15", description="Starbucks Coffee", amount=-4.99),
            EnhancedTransaction(date="2024-01-16", description="Monthly Rent", amount=-1200.00),
            EnhancedTransaction(date="2024-01-17", description="Freelance Income", amount=800.00),
            EnhancedTransaction(date="2024-01-18", description="Amazon Prime Subscription", amount=-14.99),
            EnhancedTransaction(date="2024-01-19", description="Uber Ride", amount=-25.50)
        ]
        
        # Note: categorize_dataframe expects a DataFrame, not a list of EnhancedTransaction directly
        # For a simple demo, we'll create a dummy DataFrame.
        # In a real scenario, this would come from the FileProcessorAgent.
        # Ensure your agent has a method to prepare data if it expects a DataFrame
        # or mock the necessary input. Assuming a direct list of EnhancedTransaction for demo now.
        
        # This part assumes categorize_transactions takes a list of EnhancedTransaction
        categorized, result = agent.categorize_transactions(sample_tx)
        
        print("??? Categorized Transactions:")
        for tx in categorized:
            print(f"{tx.description}: {tx.category} -> {tx.subcategory} ({tx.spending_type})")
        print(f"Analysis Result: {result.reasoning}")
            
    def _demo_file_processing(self, agent: FileProcessorAgent):
        """Run file processor agent demo"""
        print("?? File Processing Demo")
        print("This would simulate reading files and extracting transaction data.")
        print("For a full test, you would need to provide a sample file.")
        print("Example: Create a dummy CSV file named 'transactions.csv' in the project root:")
        print("Date,Description,Amount")
        print("2024-01-01,Coffee Shop,-5.00")
        print("2024-01-02,Salary,2000.00")
        
        # To make this demo functional, you'd need to simulate an uploaded file.
        # This is complex in a pure CLI context without Streamlit's file_uploader.
        # For now, it remains a print statement, encouraging manual testing with Streamlit.
    
    def _demo_conversation(self, agent: ConversationalAgent):
        """Run conversational agent demo"""
        print("\n??? AI Financial Advisor Demo")
        print("Type 'exit' or 'quit' to end the conversation.\n")
        
        # Initialize context for the conversation agent
        context = {
            'transactions': [], # You can populate this with sample EnhancedTransaction objects
            'health_metrics': FinancialHealthMetrics(
                overall_score=75.0, risk_level="Moderate Risk",
                savings_ratio_score=60.0, emergency_fund_score=50.0,
                spending_stability_score=80.0, cashflow_score=70.0,
                llm_insights={"overall_assessment": "You have a good grasp of your finances but could improve savings."}
            )
        }
        
        # Initialize conversation history
        conversation_history: List[ConversationEntry] = []

        while True:
            try:
                user_input = input("?? You: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                # Pass a FinancialDataSummary object
                financial_data_summary = FinancialDataSummary(
                    transactions=context['transactions'],
                    metrics=context['health_metrics'],
                    temporal_patterns={}, # Placeholder
                    spending_breakdown={} # Placeholder
                )

                conversation_entry = agent.generate_response(
                    user_input, 
                    financial_data_summary, # Pass the structured summary
                    conversation_history # Pass the ongoing history
                )
                conversation_history.append(conversation_entry) # Add to history
                
                print(f"?? AI: {conversation_entry.ai_response}")
                # Optionally print more details from conversation_entry
                print(f"   Emotion: {conversation_entry.emotion}, Intent: {conversation_entry.intent}")
                
            except Exception as e:
                print(f"An error occurred during conversation: {e}")
                # Optional: print traceback for debugging
                # import traceback
                # print(traceback.format_exc())

    def show_agent_list(self):
        """Show available agents and descriptions"""
        print("\n?? Available Agents:")
        for name, info in self.available_agents.items():
            print(f"- {name:<20} : {info.get('description', 'No description provided')}")
        print("\nExample Usage:")
        print("python launch.py --agent health_analyzer")
        print("python launch.py --agent transaction_categorizer\n")
        
    def run_setup_wizard(self):
        """Run setup wizard to configure environment"""
        print("\n??? EmpowerFin Guardian 2.0 - Setup Wizard")
        print("=" * 50)
        
        # Step 1: Check dependencies
        self._check_dependencies()
        
        # Step 2: Setup API keys
        self._setup_llm_config()
        
        # Step 3: Create required directories
        self._setup_directories()
        
        print("\n? Setup complete!")
        print("You can now start the app with: python launch.py")
            
    def _check_dependencies(self):
        """Check and install required dependencies"""
        print("\n?? Checking dependencies...")
        # Added plotly and pandas due to their use in main_application.py and agents
        # Added speech_recognition, gtts, PyAudio for voice features
        required_packages = ['pandas', 'langchain', 'streamlit', 'plotly', 'python-dotenv', 
                             'openpyxl', 'speechrecognition', 'gtts', 'pyaudio'] 
        missing_packages = []
        
        # Ensure langchain specific modules are checked
        langchain_sub_packages = ['langchain_groq', 'langchain_openai', 'langchain_anthropic']
        
        for package in required_packages:
            try:
                # Special handling for PyAudio as its import name is different from package name
                if package.lower() == 'pyaudio':
                    importlib.import_module('PyAudio') # Check for the module name
                else:
                    # Replace hyphen with underscore for module names if necessary
                    module_name = package.replace('-', '_') 
                    importlib.import_module(module_name)
                print(f"? {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"? {package}")
            except Exception as e:
                print(f"? {package} (Error during import check: {e})")
                missing_packages.append(package)


        # Check for specific langchain integration packages
        for package in langchain_sub_packages:
            try:
                importlib.import_module(package)
                print(f"? {package} (LLM Integration)")
            except ImportError:
                print(f"?? {package} (LLM Integration) not found. Some LLM features may be limited.")


        if missing_packages:
            print(f"\n?? Missing required packages: {', '.join(missing_packages)}")
            install = input("Install missing packages? (y/n): ").lower().strip()
            if install == 'y':
                print("Attempting to install missing packages...")
                try:
                    # Install all missing packages at once. Note: PyAudio might need separate handling.
                    # Exclude PyAudio from batch install if it's in the missing list, guide user for it
                    if 'pyaudio' in missing_packages:
                        print("\n?PyAudio often requires system-level dependencies. Please try installing it separately:")
                        print("  On Ubuntu/Debian: sudo apt-get install portaudio19-dev")
                        print("  On macOS: brew install portaudio")
                        print(f"  Then: {sys.executable} -m pip install PyAudio")
                        missing_packages.remove('pyaudio') # Remove it so other packages can install
                    
                    if missing_packages: # Install remaining packages
                        os.system(f"{sys.executable} -m pip install {' '.join(missing_packages)}")
                    print("Installation attempt finished. Please re-run setup to verify.")
                except Exception as e:
                    print(f"Error during installation: {e}")
            else:
                print("Skipping installation. The application may not function correctly.")
        else:
            print("\nAll core dependencies are met.")

    def _setup_llm_config(self):
        """Interactive LLM configuration setup"""
        print("\n?? LLM Configuration")
        print("-" * 30)
        
        providers = {
            'groq': {
                'env_var': 'GROQ_API_KEY',
                'url': 'https://console.groq.com/', 
                'description': 'Fast inference with free tier'
            },
            'openai': {
                'env_var': 'OPENAI_API_KEY',
                'url': 'https://platform.openai.com/', # Corrected to general platform link
                'description': 'Powerful models like GPT-4'
            },
            'anthropic': {
                'env_var': 'ANTHROPIC_API_KEY',
                'url': 'https://console.anthropic.com/', 
                'description': 'Claude models for detailed analysis'
            }
        }
        
        for provider, info in providers.items():
            current_key = os.getenv(info['env_var'])
            status = "? Configured" if current_key else "? Not configured"
            print(f"{provider.title()}: {status}")
            print(f"  Get key: {info['url']}")
            
            if not current_key:
                setup = input(f"  Set up {provider}? (y/n): ").lower().strip()
                if setup == 'y':
                    api_key = input(f"Enter your {provider.upper()} API key: ")
                    self._save_api_key(info['env_var'], api_key)
                    print(f"Saved {provider} API key to .env")
                
        print("\n?? Tip: You can also set API keys manually in .env file:")
        print("GROQ_API_KEY='your_key'")
        print("OPENAI_API_KEY='your_key'")
        print("ANTHROPIC_API_KEY='your_key'")
    
    def _save_api_key(self, env_var, api_key):
        """Save API key to .env file"""
        env_path = Path('.env')
        env_vars = {}
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'): # Ignore comments
                        key, value = line.split('=', 1)
                        env_vars[key] = value.strip('\'"') # Remove existing quotes
                
        # Update with new key, ensuring it's quoted for .env format
        env_vars[env_var] = f'"{api_key}"'
        
        # Write back to .env
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
            
    def _setup_directories(self):
        """Create required directories if they don't exist"""
        dirs_to_create = ['data', 'exports', 'logs', 'financial_knowledge']
        print("\n?? Setting up directories...")
        
        for dir_name in dirs_to_create:
            path = Path(dir_name)
            if not path.exists():
                try:
                    path.mkdir(exist_ok=True)
                    print(f"? Created {dir_name}/")
                except OSError as e:
                    print(f"? Error creating directory {dir_name}/: {e}")
            else:
                print(f"? {dir_name}/ already exists")
                
    def _run_tests(self):
        """Run basic functionality tests (placeholder)"""
        print("\n?? Running basic functionality tests...")
        print("This is a placeholder for actual tests. You would typically add unit or integration tests here.")
        # Example of a simple test check
        try:
            # Check if Streamlit can be imported
            import streamlit
            print("? Streamlit can be imported.")
        except Exception as e:
            print(f"? Streamlit import test failed: {e}")
        
        # Add more specific tests for agents here if desired
        print("Tests complete (placeholder).")

    def show_help(self):
        """Show help message"""
        print("\n?? EmpowerFin Guardian 2.0 - Launch Script")
        print("Usage:")
        print("  python launch.py              # Launch full application")
        print("  python launch.py --agent <name> # Launch specific agent demo")
        print("  python launch.py --demo       # Launch full application with demo data")
        print("  python launch.py --setup      # Run setup wizard (install deps, configure API keys)")
        print("  python launch.py --agents     # List available agents for demo")
        print("  python launch.py --test       # Run basic functionality tests")
        print("  python launch.py --help       # Show this help message")
        print("\nAvailable Agents for --agent option:")
        for name, info in self.available_agents.items():
            print(f"  {name:<20} - {info.get('description', 'No description provided')}")
        print("\nExamples:")
        print("  python launch.py --agent health_analyzer")
        print("  python launch.py --demo")
        print("  python launch.py --setup")


def main():
    """Main entry point with CLI support"""
    parser = argparse.ArgumentParser(
        description="EmpowerFin Guardian 2.0 Launcher",
        formatter_class=argparse.RawTextHelpFormatter # For better formatting of help message
    )
    parser.add_argument('--agent', type=str, help='Launch a specific agent demo (e.g., health_analyzer)')
    parser.add_argument('--demo', action='store_true', help='Launch full application with demo data')
    parser.add_argument('--setup', action='store_true', help='Run initial setup wizard (checks dependencies, configures API keys)')
    parser.add_argument('--agents', action='store_true', help='List all available agents for demo')
    parser.add_argument('--test', action='store_true', help='Run basic functionality tests')
    
    args = parser.parse_args()
    launcher = ApplicationLauncher()
    
    # Handle different launch modes
    if args.setup:
        launcher.run_setup_wizard()
    elif args.agents:
        launcher.show_agent_list()
    elif args.agent:
        launcher.launch_agent_demo(args.agent)
    elif args.demo:
        launcher.launch_full_app(demo=True)
    elif args.test:
        launcher._run_tests() # Call the placeholder test method
    else:
        # If no arguments are provided, show help or launch full app
        if len(sys.argv) == 1: # Only 'python launch.py' was run
            launcher.launch_full_app()
        else:
            launcher.show_help() # Show help if unknown args are passed
            sys.exit(1)


if __name__ == "__main__":
    main()
