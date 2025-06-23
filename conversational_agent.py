# conversational_agent.py - LLM-Powered Conversational Financial Advisor
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import streamlit as st # Used for Streamlit-specific feedback

from langchain.schema import HumanMessage, SystemMessage
from config_manager import llm_manager, session_manager
from data_models import ConversationEntry, EnhancedTransaction, FinancialHealthMetrics, FinancialDataSummary, UserProfile


class ConversationalAgent:
    """
    Advanced LLM-powered conversational financial advisor.
    Manages conversation flow, context preparation, intent analysis,
    and generating empathetic and actionable financial advice.
    """

    def __init__(self):
        self.llm = llm_manager.get_client()
        self.conversation_memory: List[Dict] = [] # Stores history of ConversationEntry.to_dict()
        self.user_profile: UserProfile = UserProfile() # Default UserProfile
        self.max_memory_entries = 20 # Max entries to keep in conversation history

    def generate_response(self, query: str, context: Dict = None) -> ConversationEntry:
        """
        Generates an intelligent response to a user query, considering financial context
        (transactions, health metrics) and conversation history.

        Args:
            query (str): The user's input query.
            context (Dict): A dictionary containing 'transactions' (List[EnhancedTransaction])
                            and 'health_metrics' (FinancialHealthMetrics) if available.

        Returns:
            ConversationEntry: An object encapsulating the user's query, AI's response,
                                and various metadata about the interaction.
        """
        # Ensure LLM is available before proceeding with advanced features
        if not self.llm or not llm_manager.is_available():
            st.warning("LLM is not available. Providing a basic fallback response.")
            return self._fallback_response(query, context)

        try:
            # Step 1: Prepare comprehensive context for the LLM
            full_context = self._prepare_context(query, context)

            # Step 2: Analyze user intent to tailor the response
            intent_analysis = self._analyze_user_intent(query)
            
            # Step 3: Generate the core LLM response
            # Pass the full context to the LLM response generation
            ai_raw_response = self._generate_llm_response(query, full_context, intent_analysis)

            # Step 4: Validate and potentially enhance the AI's response
            validated_response = self._validate_response(ai_raw_response)

            # Step 5: Create a ConversationEntry and update memory
            conversation_entry = ConversationEntry(
                user_query=query,
                ai_response=validated_response,
                emotion=intent_analysis.get('emotion', 'neutral'),
                intent=intent_analysis.get('intent', 'general_inquiry'),
                topics=intent_analysis.get('topics', ['general']),
                confidence=intent_analysis.get('confidence', 'medium'),
                follow_up_suggestions=intent_analysis.get('follow_up_suggestions', [])
            )

            self._update_conversation_memory(conversation_entry)
            return conversation_entry

        except Exception as e:
            # Catch broader exceptions during conversation generation
            st.error("?? An unexpected error occurred during conversation generation.")
            st.exception(e) # Display full traceback in Streamlit
            return self._fallback_response(query, context)

    def _prepare_context(self, query: str, context: Dict = None) -> Dict:
        """
        Prepares a comprehensive context dictionary for the LLM, combining
        user profile, financial metrics, transactions, and conversation history.

        Args:
            query (str): The current user query.
            context (Dict): Incoming context, typically containing 'transactions' and 'health_metrics'.

        Returns:
            Dict: A consolidated dictionary of all relevant context.
        """
        context = context or {}
        
        # Initialize with default user profile if not set
        if not self.user_profile.user_id and 'user_profile' in context and context['user_profile']:
            if isinstance(context['user_profile'], UserProfile):
                 self.user_profile = context['user_profile']
            else: # If it's a dict, try to convert
                try:
                    self.set_user_profile(UserProfile.from_dict(context['user_profile']))
                except Exception as e:
                    st.warning(f"Could not load user profile from context: {e}")

        # Construct FinancialDataSummary if transactions and health metrics are available
        financial_summary_context = {}
        transactions_list = context.get('transactions', [])
        health_metrics_obj = context.get('health_metrics')

        if transactions_list and health_metrics_obj:
            # Ensure transactions are EnhancedTransaction objects
            if transactions_list and hasattr(transactions_list[0], 'to_dict'):
                # Already EnhancedTransaction objects, proceed
                pass
            elif transactions_list: # Assume list of dicts, convert if necessary
                try:
                    transactions_list = [EnhancedTransaction.from_dict(tx) for tx in transactions_list]
                except Exception as e:
                    st.warning(f"Could not convert transactions to EnhancedTransaction: {e}")
                    transactions_list = [] # Clear if conversion fails
            
            # Ensure health_metrics is a FinancialHealthMetrics object
            if health_metrics_obj and not isinstance(health_metrics_obj, FinancialHealthMetrics):
                try:
                    health_metrics_obj = FinancialHealthMetrics.from_dict(health_metrics_obj)
                except Exception as e:
                    st.warning(f"Could not convert health metrics to FinancialHealthMetrics: {e}")
                    health_metrics_obj = None # Clear if conversion fails

            if transactions_list and health_metrics_obj:
                try:
                    # Create a dummy temporal_patterns and spending_breakdown for FinancialDataSummary
                    # In a real app, these would be calculated by an analysis agent
                    dummy_temporal_patterns = {
                        "monthly_income": self._get_monthly_income_data(transactions_list),
                        "monthly_expenses": self._get_monthly_expenses_data(transactions_list),
                        "spending_trend_summary": self._get_spending_trend_summary(transactions_list)
                    }
                    dummy_spending_breakdown = {
                        "by_category": self._get_top_spending_categories(transactions_list),
                        "by_behavior": {} # Could be extended later
                    }

                    financial_summary = FinancialDataSummary(
                        transactions=transactions_list,
                        metrics=health_metrics_obj,
                        temporal_patterns=dummy_temporal_patterns,
                        spending_breakdown=dummy_spending_breakdown
                    )
                    financial_summary_context = financial_summary.to_llm_context()
                except Exception as e:
                    st.warning(f"Failed to create FinancialDataSummary for LLM context: {e}")
                    financial_summary_context = {}
        
        return {
            'current_query': query,
            'financial_summary_context': financial_summary_context, # Pass structured summary
            'conversation_history': self.conversation_memory[-5:], # Last 5 entries for brevity
            'user_profile': self.user_profile.to_dict() # Pass user profile as dict
        }

    def _analyze_user_intent(self, query: str) -> Dict:
        """
        Analyzes the user's query to determine emotional state, primary intent,
        financial topics, and suggested follow-up questions using the LLM.
        """
        if not self.llm or not llm_manager.is_available():
            return self._basic_intent_analysis(query)

        try:
            prompt = f"""
            Analyze the following user query for its emotional tone, primary intent,
            and relevant financial topics. Provide a confidence level and suggest
            1-3 concise follow-up questions.

            USER QUERY: "{query}"

            Expected JSON format (within a markdown code block):
            ```json
            {{
                "emotion": "neutral|happy|concerned|frustrated|curious",
                "intent": "information_seeking|advice_seeking|problem_solving|goal_setting|feedback|clarification",
                "topics": ["budgeting", "saving", "investing", "debt", "spending", "income", "retirement", "taxes", "general_health"],
                "urgency": "low|medium|high",
                "confidence": "low|medium|high",
                "follow_up_suggestions": ["Question 1?", "Question 2?"]
            }}
            ```
            """

            messages = [
                SystemMessage(content="You are an expert AI assistant specializing in natural language understanding for financial coaching. You precisely identify user intent and emotions."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            raw_response = response.content
            
            # Extract JSON from response, looking for markdown code block first
            match = re.search(r'```json\n(\{.*?\})\n```', raw_response, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', raw_response, re.DOTALL) # Fallback if not in code block

            if match:
                result = json.loads(match.group(1)) # Use group(1) for content inside curly braces
                result["success"] = True
                return result
            else:
                st.warning(f"LLM intent analysis failed to return valid JSON. Raw: {raw_response[:200]}...")
                return self._basic_intent_analysis(query)
        except Exception as e:
            st.warning(f"LLM intent analysis experienced an error: {e}. Falling back to basic analysis.")
            return self._basic_intent_analysis(query)

    def _basic_intent_analysis(self, query: str) -> Dict:
        """
        Provides a basic, rule-based intent analysis when LLM is unavailable or fails.
        """
        return {
            "emotion": "neutral",
            "intent": "general_inquiry",
            "topics": ["general"],
            "urgency": "medium",
            "confidence": "low", # Low confidence for rule-based fallback
            "follow_up_suggestions": [
                "Could you please provide more details?",
                "What specifically are you interested in regarding your finances?",
                "How can I help you achieve your financial goals?"
            ],
            "success": False
        }

    def _generate_llm_response(self, query: str, full_context: Dict, intent_analysis: Dict) -> str:
        """
        Generates the main conversational response from the LLM, leveraging
        the full prepared context and analyzed user intent.
        """
        prompt = self._create_financial_coaching_prompt(query, full_context, intent_analysis)
        
        messages = [
            SystemMessage(content="You are an empathetic and practical certified financial planner and behavioral finance coach. Your goal is to provide clear, actionable advice, simplify complex concepts, and guide the user towards better financial health. Maintain a supportive and encouraging tone."),
            HumanMessage(content=prompt)
        ]

        st.info("?? Sending prompt to LLM for response generation...")
        # st.code(prompt[:800] + "..." if len(prompt) > 800 else prompt) # Optional: for debugging LLM prompts

        try:
            response = self.llm.invoke(messages)
            st.success("? LLM responded successfully.")
            if hasattr(response, 'content'):
                # st.code(response.content, language="markdown") # Optional: for debugging LLM responses
                return response.content
            else:
                st.warning("?? LLM response object has no '.content' attribute.")
                return "I received an empty response from my core advisor system."
        except Exception as e:
            st.error("? LLM invocation failed during response generation.")
            st.exception(e) # Display full traceback in Streamlit
            return f"I'm sorry, I'm having trouble connecting to my financial advice system right now: {e}"

    def _create_financial_coaching_prompt(self, query: str, full_context: Dict, intent_analysis: Dict) -> str:
        """
        Constructs the detailed prompt for the LLM to generate a financial coaching response.
        Incorporates user profile, financial summary, conversation history, and user intent.
        """
        user_profile_str = json.dumps(full_context.get('user_profile', {}), indent=2)
        financial_summary_str = json.dumps(full_context.get('financial_summary_context', {}), indent=2)
        conversation_history_str = json.dumps(full_context.get('conversation_history', []), indent=2)
        
        # Prepare intent analysis for prompt, focusing on key aspects
        intent_summary = f"Emotion: {intent_analysis.get('emotion', 'neutral')}, Intent: {intent_analysis.get('intent', 'general_inquiry')}, Topics: {', '.join(intent_analysis.get('topics', []))}"


        return f"""
        As a certified financial coach, provide empathetic, practical, and actionable advice.
        Your response should be direct, easy to understand, and conclude by asking 1-2 relevant follow-up questions.
        If appropriate, simplify financial concepts.

        --- CONTEXT ---
        USER PROFILE:
        {user_profile_str}

        FINANCIAL DATA SUMMARY:
        {financial_summary_str}

        CONVERSATION HISTORY (recent interactions):
        {conversation_history_str}

        USER INTENT ANALYSIS:
        {intent_summary}

        --- USER QUERY ---
        {query}

        --- RESPONSE INSTRUCTIONS ---
        1. Start directly addressing the user's query based on the provided context.
        2. Provide clear and actionable financial advice.
        3. Break down complex financial terms if necessary.
        4. Maintain an empathetic and supportive tone.
        5. Conclude with 1-2 open-ended questions to continue the conversation or gather more information.
        6. Always include the general disclaimer at the end of your response.
        """

    def _validate_response(self, response: str) -> str:
        """
        Validates the LLM's response, ensuring it's not empty and appending a disclaimer if missing.
        """
        if not response or len(response.strip()) < 20: # Increased minimum length for meaningful response
            return "I'm having a little difficulty formulating a helpful response right now. Could you please provide more context or rephrase your question?"
        
        # Check if a disclaimer is already present (case-insensitive)
        disclaimer_keywords = ["general financial guidance", "not personalized investment advice", "consult a professional"]
        if not any(keyword in response.lower() for keyword in disclaimer_keywords):
            response += "\n\n**Disclaimer:** This is general financial guidance and not personalized investment advice. Always consult with a qualified financial professional for advice tailored to your specific situation."
            
        return response.strip()

    def _fallback_response(self, query: str, context: Dict = None) -> ConversationEntry:
        """
        Provides a basic fallback response when the LLM is unavailable or an unrecoverable
        error occurs during response generation.
        """
        return ConversationEntry(
            user_query=query,
            ai_response="I'm currently unable to provide enhanced financial advice due to a system issue. Please try again later or provide simpler queries.",
            emotion='neutral',
            intent='general_inquiry',
            topics=['system_issue'],
            confidence='low',
            follow_up_suggestions=["Can I help with something else?"]
        )

    def _update_conversation_memory(self, entry: ConversationEntry):
        """
        Adds a new conversation entry to the memory and prunes it to maintain a maximum size.
        """
        self.conversation_memory.append(entry.to_dict())
        if len(self.conversation_memory) > self.max_memory_entries:
            self.conversation_memory.pop(0) # Remove oldest entry

    def set_user_profile(self, profile: UserProfile):
        """Sets the user profile for the conversational agent."""
        self.user_profile = profile

    def get_conversation_summary(self) -> Dict:
        """
        Generates a summary of the conversation history, including common topics
        and emotional patterns.
        """
        if not self.conversation_memory:
            return {
                'total_interactions': 0,
                'common_topics': {},
                'emotional_patterns': {},
                'most_recent': None
            }
        
        topic_counts = {}
        emotion_counts = {}
        for entry_dict in self.conversation_memory: # Iterate through dicts
            for topic in entry_dict.get('topics', ['general']):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            emotion = entry_dict.get('emotion', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            'total_interactions': len(self.conversation_memory),
            'common_topics': dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)),
            'emotional_patterns': dict(sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)),
            'most_recent': self.conversation_memory[-1]
        }

    # Helper methods for _prepare_context (if FinancialDataSummary not fully populated)
    def _get_date_range(self, transactions: List[EnhancedTransaction]) -> str:
        """Helper to get the date range of transactions."""
        if not transactions:
            return "No transaction data"
        dates = [datetime.fromisoformat(tx.date) for tx in transactions if tx.date]
        if not dates: return "N/A" # Handle cases where all dates might be invalid
        return f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"

    def _get_top_spending_categories(self, transactions: List[EnhancedTransaction]) -> List[Tuple[str, float]]:
        """Helper to get top spending categories and their total amounts."""
        category_spending = {}
        for tx in transactions:
            if tx.amount < 0: # Only consider expenses
                category = tx.category
                category_spending[category] = category_spending.get(category, 0.0) + abs(tx.amount)
        return sorted(category_spending.items(), key=lambda x: x[1], reverse=True)[:5]

    def _get_monthly_income_data(self, transactions: List[EnhancedTransaction]) -> Dict[str, float]:
        """Calculates monthly total income."""
        monthly_income = {}
        for tx in transactions:
            if tx.amount > 0 and tx.date:
                month_key = datetime.fromisoformat(tx.date).strftime('%Y-%m')
                monthly_income[month_key] = monthly_income.get(month_key, 0.0) + tx.amount
        return dict(sorted(monthly_income.items()))

    def _get_monthly_expenses_data(self, transactions: List[EnhancedTransaction]) -> Dict[str, float]:
        """Calculates monthly total expenses."""
        monthly_expenses = {}
        for tx in transactions:
            if tx.amount < 0 and tx.date:
                month_key = datetime.fromisoformat(tx.date).strftime('%Y-%m')
                monthly_expenses[month_key] = monthly_expenses.get(month_key, 0.0) + abs(tx.amount)
        return dict(sorted(monthly_expenses.items()))
    
    def _get_spending_trend_summary(self, transactions: List[EnhancedTransaction]) -> str:
        """Provides a simple summary of recent spending trend."""
        monthly_expenses = self._get_monthly_expenses_data(transactions)
        
        if len(monthly_expenses) < 2:
            return "Not enough data to determine spending trend."
        
        sorted_months = sorted(monthly_expenses.keys())
        latest_month = monthly_expenses[sorted_months[-1]]
        previous_month = monthly_expenses[sorted_months[-2]]
        
        if latest_month > previous_month:
            return f"Recent spending increased by {((latest_month - previous_month) / previous_month * 100):.1f}%."
        elif latest_month < previous_month:
            return f"Recent spending decreased by {((previous_month - latest_month) / previous_month * 100):.1f}%."
        else:
            return "Recent spending has remained stable."


