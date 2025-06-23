# categorization_agent.py - Intelligent Transaction Categorization Agent
import pandas as pd
import json
import re
from typing import List, Dict, Any, Tuple
# Removed direct streamlit import as agents should generally not interact with UI directly
# import streamlit as st 
from langchain.schema import HumanMessage, SystemMessage
from config_manager import llm_manager, app_config # Import app_config for category schema
from data_models import EnhancedTransaction, LLMAnalysisResult
from file_processor_agent import FileProcessorAgent # To use its convert_to_enhanced_transactions

class TransactionCategorizationAgent:
    """
    Intelligent transaction categorization using LLMs and rule-based fallbacks.
    Categorizes financial transactions into predefined categories and subcategories,
    and identifies spending types.
    """

    def __init__(self):
        self.llm = llm_manager.get_client()
        # The supported_formats here might be redundant if file processing is handled
        # by FileProcessorAgent, but kept for consistency if needed elsewhere.
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.txt', '.pdf'] 
        self.file_processor = FileProcessorAgent() # Instantiate to use its conversion method
        self.category_schema = app_config.category_schema
        self.category_keywords = app_config.category_keywords

    def categorize_dataframe(self, df: pd.DataFrame) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """
        Categorizes transactions present in a Pandas DataFrame.
        First converts DataFrame to EnhancedTransaction objects, then applies LLM or rule-based categorization.

        Args:
            df (pd.DataFrame): The input DataFrame containing transaction data.

        Returns:
            Tuple[List[EnhancedTransaction], LLMAnalysisResult]: A tuple containing
            a list of categorized EnhancedTransaction objects and an LLMAnalysisResult.
        """
        try:
            # Step 1: Convert DataFrame to EnhancedTransaction objects.
            # This ensures all necessary fields (like 'balance') are preserved
            # from the FileProcessorAgent's output.
            transactions_list = self.file_processor.convert_to_enhanced_transactions(df)

            if not transactions_list:
                return [], LLMAnalysisResult(
                    success=False,
                    error_message="No valid transactions to categorize after conversion from DataFrame."
                )

            # Step 2: Use LLM for advanced categorization if available, otherwise fall back to rules.
            if self.llm and llm_manager.is_available(): # Explicitly check if LLM is available
                # print("LLM available for categorization. Attempting LLM-enhanced categorization.") # For debugging
                categorized_transactions, analysis_result = self._llm_enhanced_categorization(transactions_list)
            else:
                # print("LLM not available for categorization. Falling back to rule-based categorization.") # For debugging
                categorized_transactions, analysis_result = self._rule_based_categorization(transactions_list)

            return categorized_transactions, analysis_result

        except Exception as e:
            # Catch any unexpected errors during the overall process within this agent.
            # Do not use st.error here; return a failure result.
            return [], LLMAnalysisResult(
                success=False,
                error_message=f"Categorization failed due to an unexpected error: {str(e)}"
            )

    def _llm_enhanced_categorization(self, transactions: List[EnhancedTransaction]) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """
        Uses the LLM to intelligently categorize transactions based on their descriptions.
        The LLM is expected to return a list of categorized transactions.
        """
        if not self.llm or not llm_manager.is_available():
            # Fallback to rule-based if LLM becomes unavailable during this specific step
            # print("LLM not available in _llm_enhanced_categorization. Falling back to rule-based.")
            return self._rule_based_categorization(transactions)

        try:
            # Prepare a sample of transactions for the LLM prompt.
            # Only send relevant fields to save tokens and focus LLM.
            sample_data_for_llm = self._prepare_sample_for_llm(transactions)

            # Create the LLM prompt for categorization.
            prompt = self._create_categorization_prompt(sample_data_for_llm)

            messages = [
                SystemMessage(content="You are a highly accurate financial data expert specializing in categorizing bank statement transactions. Categorize EACH transaction provided."),
                HumanMessage(content=prompt)
            ]

            # Invoke the LLM
            response = self.llm.invoke(messages)
            raw_response = response.content

            # Extract and parse the JSON response from the LLM.
            match = re.search(r'```json\n(\{.*?\})\n```', raw_response, re.DOTALL)
            if not match:
                # Fallback to general JSON extraction if code block not found
                match = re.search(r'\{.*\}', raw_response, re.DOTALL)

            if match:
                llm_categorized_data = json.loads(match.group(1)) # Use group(1) to get content inside curly braces
            else:
                raise ValueError(f"LLM did not return valid JSON for categorization. Raw response: {raw_response[:500]}...")

            # Validate the structure of the LLM response
            if 'transactions' not in llm_categorized_data or not isinstance(llm_categorized_data['transactions'], list):
                raise ValueError("LLM response missing 'transactions' list or it's not a list.")

            # Create a new list to store updated transactions
            updated_transactions = []
            
            # Apply LLM categorization to each transaction.
            # Assuming the order of transactions in the LLM response matches the input `transactions` list.
            for i, tx in enumerate(transactions):
                if i < len(llm_categorized_data['transactions']):
                    llm_tx_data = llm_categorized_data['transactions'][i]
                    
                    # Update the EnhancedTransaction object with LLM's categorization
                    tx.category = llm_tx_data.get('category', tx.category)
                    tx.subcategory = llm_tx_data.get('subcategory', tx.subcategory)
                    tx.spending_type = llm_tx_data.get('spending_type', tx.spending_type)
                    tx.confidence = llm_tx_data.get('confidence', 0.8) # Default confidence if LLM doesn't provide
                    tx.reasoning = llm_tx_data.get('reasoning', "LLM inferred category/subcategory.")
                updated_transactions.append(tx)

            # Prepare the LLMAnalysisResult
            analysis_data = {
                "categorized_count": len(updated_transactions),
                "confidence_avg": sum(t.confidence for t in updated_transactions) / len(updated_transactions) if updated_transactions else 0
            }

            return updated_transactions, LLMAnalysisResult(
                success=True,
                confidence='high',
                reasoning="LLM-assisted categorization applied successfully.",
                data=analysis_data
            )

        except Exception as e:
            # If LLM categorization fails, return a failure result and fall back to rule-based categorization
            # print(f"LLM categorization failed: {e}. Falling back to rule-based categorization.") # For debugging
            return self._rule_based_categorization(transactions) # This already returns LLMAnalysisResult

    def _rule_based_categorization(self, transactions: List[EnhancedTransaction]) -> Tuple[List[EnhancedTransaction], LLMAnalysisResult]:
        """
        Applies rule-based categorization to a list of EnhancedTransaction objects.
        This serves as a fallback or initial categorization method.
        """
        categorized_transactions = []
        for tx in transactions:
            description_lower = tx.description.lower()
            
            # Detect main category using keywords from app_config
            detected_category = "other"
            for category_name, keywords_list in self.category_keywords.items():
                if any(keyword in description_lower for keyword in keywords_list):
                    detected_category = category_name
                    break
            
            # Detect subcategory using app_config schema
            detected_subcategory = "uncategorized"
            if detected_category in self.category_schema:
                for subcat in self.category_schema[detected_category]['subcategories']:
                    # Check if the subcategory name (with spaces instead of underscores) is in the description
                    if subcat.replace('_', ' ') in description_lower: 
                        detected_subcategory = subcat
                        break
                # Fallback to custom subcategory detection for some common cases if specific subcat not found
                if detected_category == "food_dining":
                    if "fast food" in description_lower:
                        detected_subcategory = "fast_food"
                    elif "dining out" in description_lower or "restaurant" in description_lower:
                        detected_subcategory = "restaurants"
                    elif "grocery" in description_lower or "supermarket" in description_lower:
                        detected_subcategory = "groceries"
                    elif "coffee" in description_lower or "cafe" in description_lower: # Added coffee/cafe
                        detected_subcategory = "coffee_snacks"

                elif detected_category == "transportation":
                    if "uber" in description_lower or "lyft" in description_lower:
                        detected_subcategory = "rideshare"
                    elif "gas" in description_lower or "fuel" in description_lower:
                        detected_subcategory = "fuel"
                    elif "bus" in description_lower or "metro" in description_lower:
                        detected_subcategory = "public_transport"
                
                elif detected_category == "fixed_expenses": # Added for fixed expenses
                    if "rent" in description_lower or "mortgage" in description_lower:
                        detected_subcategory = "rent_mortgage"
                    elif "insurance" in description_lower:
                        detected_subcategory = "insurance"
                    elif "utility" in description_lower or "electricity" in description_lower or "water" in description_lower:
                        detected_subcategory = "utilities"
                    elif "subscription" in description_lower:
                        detected_subcategory = "subscriptions"


            # Basic spending type detection (can be expanded)
            spending_type = "regular"
            if "online" in description_lower or "amazon" in description_lower or "impulse" in description_lower:
                spending_type = "impulse"
            elif "monthly" in description_lower or "subscription" in description_lower:
                spending_type = "regular"
            elif "travel" in description_lower or "vacation" in description_lower:
                spending_type = "seasonal" # Could be seasonal/major purchase depending on amount
            elif "emi" in description_lower or "loan" in description_lower: # Added for loans
                spending_type = "fixed" # New spending type if applicable

            tx.category = detected_category
            tx.subcategory = detected_subcategory
            tx.spending_type = spending_type
            tx.confidence = 0.7 # Medium confidence for rule-based
            tx.reasoning = "Rule-based categorization."
            categorized_transactions.append(tx)

        analysis_data = {
            "categorized_count": len(categorized_transactions),
            "confidence_avg": 0.7 # Fixed confidence for rule-based
        }
        return categorized_transactions, LLMAnalysisResult(
            success=True,
            confidence='medium',
            reasoning="Rule-based categorization applied successfully.",
            data=analysis_data
        )

    def _create_categorization_prompt(self, sample_data: List[Dict]) -> str:
        """
        Creates a comprehensive prompt for the LLM to categorize transactions.
        Specifies the expected JSON output format for parsing.
        """
        # Get all available categories and subcategories from app_config for the prompt
        all_categories = list(self.category_schema.keys())
        all_subcategories = {cat: data['subcategories'] for cat, data in self.category_schema.items()}

        return f"""
        You are an expert financial data categorizer. Your task is to accurately categorize
        each financial transaction into a main 'category', a more specific 'subcategory',
        and a 'spending_type'. Provide a confidence score for each categorization and a brief reasoning.

        Here are the allowed categories and their subcategories:
        {json.dumps(all_subcategories, indent=2)}

        Allowed spending types are: "regular", "impulse", "seasonal", "major_purchase", "fixed".

        Analyze these transactions from a bank statement:
        
        TRANSACTIONS:
        {json.dumps(sample_data, indent=2)}
        
        Return your analysis in a structured JSON array, where each object corresponds
        to an input transaction in the *same order*.

        Example of desired JSON output for ONE transaction:
        ```json
        {{
            "transactions": [
                {{
                    "description": "Starbucks Coffee",
                    "category": "food_dining",
                    "subcategory": "coffee_snacks",
                    "spending_type": "regular",
                    "confidence": 0.95,
                    "reasoning": "Coffee shop indicates regular food_dining."
                }},
                {{
                    "description": "Monthly Rent",
                    "category": "fixed_expenses",
                    "subcategory": "rent_mortgage",
                    "spending_type": "regular",
                    "confidence": 0.99,
                    "reasoning": "Explicit 'Rent' payment, a fixed expense."
                }}
            ]
        }}
        ```
        """

    def _prepare_sample_for_llm(self, transactions: List[EnhancedTransaction]) -> List[Dict]:
        """
        Prepares a sample of transactions (first 5) to send to the LLM.
        Selects only relevant fields to optimize token usage.
        """
        sample_size = min(5, len(transactions))
        sample_transactions = transactions[:sample_size]
        
        # Create a light-weight dictionary representation for the LLM
        return [{
            'date': tx.date,
            'description': tx.description,
            'amount': tx.amount
        } for tx in sample_transactions]

