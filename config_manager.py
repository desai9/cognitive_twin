# config_manager.py - Core Configuration and LLM Management
import os
from dotenv import load_dotenv
import json
from typing import Dict, Optional, Any
import streamlit as st

# Load environment variables
load_dotenv()

# LLM Integration
try:
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain.schema import HumanMessage, SystemMessage
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class LLMConfigManager:
    """Centralized LLM configuration and management"""

    def __init__(self):
        self.available_models = {
            'groq': {
                'models': ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
                'api_key_env': 'GROQ_API_KEY',
                'default_model': 'llama3-70b-8192'
            },
            'openai': {
                'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                'api_key_env': 'OPENAI_API_KEY',
                'default_model': 'gpt-4o-mini'
            },
            'anthropic': {
                'models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
                'api_key_env': 'ANTHROPIC_API_KEY',
                'default_model': 'claude-3-haiku-20240307'
            }
        }

        self.selected_provider = None
        self.selected_model = None
        self.llm_client = None

        # Initialize with available provider
        self._initialize_llm()
        print(f"[LLM] Initialized {self.selected_provider} - {self.selected_model}")

    def _initialize_llm(self):
        """Initialize LLM with first available provider"""
        if not LLM_AVAILABLE:
            return False

        for provider, config in self.available_models.items():
            api_key = os.getenv(config['api_key_env'])
            if api_key:
                try:
                    self.selected_provider = provider
                    self.selected_model = config['default_model']
                    self.llm_client = self._create_llm_client(provider, self.selected_model, api_key)
                    return True
                except Exception as e:
                    # Log the error for debugging but continue trying other providers
                    print(f"Error initializing LLM for {provider}: {e}")
                    continue

        return False

    def _create_llm_client(self, provider: str, model: str, api_key: str):
        """Create LLM client based on provider"""
        if provider == 'groq':
            return ChatGroq(
                model=model,
                groq_api_key=api_key,
                temperature=0.7,
                max_tokens=1000
            )
        elif provider == 'openai':
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=1000
            )
        elif provider == 'anthropic':
            return ChatAnthropic(
                model=model,
                anthropic_api_key=api_key,
                temperature=0.7,
                max_tokens=1000
            )

    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self.llm_client is not None

    def get_model_info(self) -> str:
        """Get current model information"""
        if self.is_available():
            return f"{self.selected_provider.title()} - {self.selected_model}"
        return "No LLM Available"

    def get_client(self):
        """Get the LLM client"""
        return self.llm_client


class AppConfig:
    """Application configuration and settings"""

    def __init__(self):
        self.app_name = "EmpowerFin Guardian 2.0"
        self.version = "2.0.0"
        self.description = "LLM-Enhanced Financial Intelligence Platform"

        # Define standard column names and their possible aliases from various bank statements
        self.column_aliases = {
            'date': ['date', 'transaction date', 'posted date', 'value date', 'activity date'],
            'description': ['description', 'transaction description', 'details', 'payee', 'memo'],
            'amount': ['amount', 'debit', 'credit', 'transaction amount', 'value'],
            'balance': ['balance', 'running balance', 'current balance']
        }

        # Financial categories schema
        self.category_schema = {
            'income': {
                'subcategories': ['salary', 'freelance', 'investment_returns', 'business_income', 'other_income'],
                'description': 'Money coming in from various sources'
            },
            'fixed_expenses': {
                'subcategories': ['rent_mortgage', 'insurance', 'utilities', 'subscriptions', 'loan_payments'],
                'description': 'Regular recurring expenses that are difficult to change'
            },
            'food_dining': {
                'subcategories': ['groceries', 'restaurants', 'coffee_snacks', 'meal_delivery', 'alcohol'],
                'description': 'All food and dining related expenses'
            },
            'transportation': {
                'subcategories': ['fuel', 'public_transport', 'rideshare', 'parking', 'vehicle_maintenance'],
                'description': 'Transportation and vehicle related costs'
            },
            'shopping': {
                'subcategories': ['clothing', 'electronics', 'home_goods', 'online_shopping', 'gifts'],
                'description': 'Discretionary shopping and purchases'
            },
            'healthcare': {
                'subcategories': ['medical_visits', 'prescriptions', 'dental', 'vision', 'fitness'],
                'description': 'Health and wellness related expenses'
            },
            'entertainment': {
                'subcategories': ['streaming', 'movies', 'events', 'hobbies', 'travel'],
                'description': 'Entertainment and leisure activities'
            },
            'financial': {
                'subcategories': ['bank_fees', 'investment', 'savings_transfer', 'credit_payment', 'tax'],
                'description': 'Financial services and money management'
            },
            'other': {
                'subcategories': ['uncategorized', 'cash_withdrawal', 'miscellaneous'],
                'description': 'Transactions that don\'t fit other categories'
            }
        }

        # Rule-based categorization keywords
        self.category_keywords = {
            'food_dining': ['grocery', 'supermarket', 'food', 'restaurant', 'cafe', 'mcdonalds', 'starbucks'],
            'transportation': ['gas', 'fuel', 'uber', 'taxi', 'bus', 'metro', 'parking'],
            'fixed_expenses': ['rent', 'mortgage', 'insurance', 'utility', 'phone', 'internet', 'electric'],
            'shopping': ['amazon', 'walmart', 'target', 'shopping', 'store', 'mall'],
            'income': ['salary', 'wage', 'payroll', 'deposit', 'income', 'pay'],
            'entertainment': ['netflix', 'spotify', 'movie', 'game', 'entertainment'],
            'healthcare': ['pharmacy', 'doctor', 'medical', 'hospital', 'cvs', 'walgreens']
        }


class SessionManager:
    """Manages Streamlit session state"""

    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        # Note: These initializations are also handled in EmpowerFinApplication's __init__
        # for robustness against Streamlit's rerun behavior, but kept here for clarity.
        if 'transactions' not in st.session_state:
            st.session_state.transactions = []
        if 'health_metrics' not in st.session_state:
            st.session_state.health_metrics = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'llm_analysis' not in st.session_state:
            st.session_state.llm_analysis = {}
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {}
        if 'user_consent' not in st.session_state:
            st.session_state.user_consent = False
        if 'file_processed' not in st.session_state:
            st.session_state.file_processed = False
        if 'last_uploaded_file_name' not in st.session_state:
            st.session_state.last_uploaded_file_name = None
        if 'masking_preview_confirmed' not in st.session_state:
            st.session_state.masking_preview_confirmed = False
        if '_temp_raw_df' not in st.session_state: # To temporarily hold original df for consent review
            st.session_state._temp_raw_df = None
        if '_temp_masked_df' not in st.session_state: # To temporarily hold masked df for consent review
            st.session_state._temp_masked_df = None


    @staticmethod
    def clear_session():
        """Clear all session data, except user_consent to persist choice"""
        keys_to_clear = [
            'transactions', 
            'health_metrics', 
            'conversation_history', 
            'llm_analysis', 
            'file_processed', 
            'last_uploaded_file_name',
            'masking_preview_confirmed',
            '_temp_raw_df', # Clear temporary DFs
            '_temp_masked_df' # Clear temporary DFs
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Keep user_consent, or reset it if explicitly desired for a completely fresh start
        # For now, let's keep it (or let main_application set it based on checkbox)

    @staticmethod
    def save_analysis_result(key: str, data: Any):
        """Save analysis result to session"""
        if 'llm_analysis' not in st.session_state:
            st.session_state.llm_analysis = {}
        st.session_state.llm_analysis[key] = data


# Global instances
llm_manager = LLMConfigManager()
app_config = AppConfig()
session_manager = SessionManager()

# Perform session state initialization globally when modules are imported
session_manager.initialize_session_state()
