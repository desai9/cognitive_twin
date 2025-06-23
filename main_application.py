# main_application.py - Main Streamlit Application Orchestrator
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from typing import Dict, List, Any, Optional
import asyncio # For running async SQLite operations

# Voice input/output libraries
try:
    import speech_recognition as sr
    from gtts import gTTS
    from io import BytesIO # Required for gTTS to work with st.audio
    VOICE_SUPPORT_AVAILABLE = True
except ImportError as e:
    st.warning(f"Voice processing libraries (SpeechRecognition, gTTS) not found: {e}. Voice features disabled.")
    VOICE_SUPPORT_AVAILABLE = False
except Exception as e:
    st.warning(f"Error loading voice processing libraries: {e}. Voice features disabled.")
    VOICE_SUPPORT_AVAILABLE = False

# Import agents and models
# Note: config_manager should be imported first as it handles session state initialization
from config_manager import llm_manager, app_config, session_manager 
from data_masker import PiiMasker
from file_processor_agent import FileProcessorAgent
from categorization_agent import TransactionCategorizationAgent
from health_analyzer_agent import LLMHealthAnalyzer
from trend_analyzer_agent import HealthTrendAnalyzer
from conversational_agent import ConversationalAgent
from data_models import EnhancedTransaction, FinancialHealthMetrics, UserProfile, LLMAnalysisResult, ConversationEntry
from knowledge_base import FinancialKnowledgeBase # Import the Knowledge Base


class EmpowerFinApplication:
    """
    Main Streamlit application orchestrating all financial intelligence components.
    Handles UI rendering, file uploads, data processing, analysis, and conversational AI.
    """
    
    def __init__(self):
        # Initialize agents
        self.file_processor = FileProcessorAgent()
        self.categorizer = TransactionCategorizationAgent()
        self.health_analyzer = LLMHealthAnalyzer()
        self.trend_analyzer = HealthTrendAnalyzer()
        self.conversation_agent = ConversationalAgent()
        self.masker = PiiMasker()
        self.kb = FinancialKnowledgeBase() # Initialize the Knowledge Base

        # Initialize session state variables directly within the main application's __init__
        # This acts as a safeguard to ensure they are always present when the app runs.
        if 'transactions' not in st.session_state:
            st.session_state.transactions = []
        if 'health_metrics' not in st.session_state:
            st.session_state.health_metrics = None
        if 'file_processed' not in st.session_state:
            st.session_state.file_processed = False
        if 'last_uploaded_file_name' not in st.session_state: # Added for file re-upload check
            st.session_state.last_uploaded_file_name = None
        if 'user_consent' not in st.session_state: # Ensure user_consent is always initialized
            st.session_state.user_consent = False
        if 'conversation_history' not in st.session_state: # Ensure conversation_history is initialized
            st.session_state.conversation_history = []
        if 'masking_preview_confirmed' not in st.session_state: # New flag for consent flow
            st.session_state.masking_preview_confirmed = False
        if 'voice_mode' not in st.session_state: # Initialize voice mode state for audio feature
            st.session_state.voice_mode = False
        if 'kb_data_loaded' not in st.session_state: # Flag to track if KB data has been loaded
            st.session_state.kb_data_loaded = False


    def run(self):
        """
        Runs the full Streamlit application, setting up page configuration
        and rendering various UI components.
        """
        # Set Streamlit page configuration (must be called once at the top)
        st.set_page_config(
            page_title="EmpowerFin Guardian 2.0",
            page_icon="📊", 
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Asynchronously load data from Knowledge Base on initial run or after clear
        if not st.session_state.kb_data_loaded and self.kb.is_ready():
            asyncio.run(self._load_data_from_kb())
            # After loading, if data was found, ensure the app reruns to display it
            if st.session_state.transactions and st.session_state.health_metrics:
                st.session_state.file_processed = True # Mark as processed if data loaded from KB
                st.session_state.kb_data_loaded = True # Prevent re-loading on every rerun
                st.rerun() # Rerun to display the dashboard with loaded data

        self._render_header()
        self._render_sidebar()
        
        # Main dashboard tabs, only displayed if transactions are loaded
        self._render_main_dashboard()

    async def _load_data_from_kb(self):
        """Loads user data and transactions from SQLite Knowledge Base."""
        if self.kb.is_ready():
            with st.spinner("⏳ Loading your financial data from local database..."): # Corrected message
                user_profile_dict, health_metrics_dict = await self.kb.load_user_data()
                transactions_data = await self.kb.get_transactions()
                conversation_history_data = await self.kb.get_conversation_history()

                if user_profile_dict:
                    # Update conversation agent's user profile
                    self.conversation_agent.set_user_profile(UserProfile(**user_profile_dict))
                    st.session_state.user_consent = user_profile_dict.get('conversation_preferences', {}).get('data_sharing_consent', False)

                if transactions_data:
                    # Convert dicts back to EnhancedTransaction objects and store
                    st.session_state.transactions = [EnhancedTransaction(**tx) for tx in transactions_data]
                    st.session_state.file_processed = True
                
                if health_metrics_dict:
                    st.session_state.health_metrics = FinancialHealthMetrics(**health_metrics_dict)
                    st.session_state.file_processed = True

                if conversation_history_data:
                    # Convert dicts back to ConversationEntry objects (or leave as dicts if ok)
                    # For consistency with how chat history is displayed, store as dicts directly
                    st.session_state.conversation_history = conversation_history_data

                st.session_state.kb_data_loaded = True
                st.success("✅ Financial data loaded from local database.") 
        else:
            st.warning("⚠️ Knowledge Base not ready. Cannot load historical data.")


    def _render_header(self):
        """
        Renders the application header, including app name, description,
        and LLM status (available/basic mode).
        """
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"# 🤖 {app_config.app_name}") 
            st.markdown(f"*{app_config.description}*")
        with col2:
            if llm_manager.is_available():
                st.success(f"🤖 AI: {llm_manager.get_model_info()}") 
            else:
                st.warning("⚠️ AI: Basic Mode (LLM not configured)") 

    def _render_sidebar(self):
        """
        Renders the sidebar, primarily for uploading bank statements.
        """
        st.sidebar.markdown("## 📁 File Upload Bank Statement") 
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV, Excel, TXT, or PDF bank statement", 
            type=["csv", "xlsx", "xls", "txt", "pdf"]
        )

        # Process file only if a new one is uploaded or if an existing one needs reprocessing
        if uploaded_file:
            # If a new file is uploaded, reset the processing flags
            if uploaded_file.name != st.session_state.get('last_uploaded_file_name'):
                st.session_state.last_uploaded_file_name = uploaded_file.name
                st.session_state.file_processed = False
                st.session_state.masking_preview_confirmed = False # Reset confirmation for new file
            
            # If file is not yet processed OR masking preview is not confirmed, start/continue processing
            if not st.session_state.file_processed or not st.session_state.masking_preview_confirmed:
                # Use asyncio.run to call the async _process_single_file
                asyncio.run(self._process_single_file(uploaded_file))
        
        elif st.session_state.file_processed:
            st.sidebar.success("✅ File already processed.")
        
        # Option to clear processed data for new upload
        if st.session_state.file_processed or st.session_state.masking_preview_confirmed: # Show clear button if any processing has started
            if st.sidebar.button("🧹 Clear All Data"): 
                session_manager.clear_session()
                # Also clear SQLite data for this user (conceptual - would need specific button/consent)
                # For now, only clearing session state and setting kb_data_loaded to False
                st.session_state.file_processed = False
                st.session_state.masking_preview_confirmed = False # Ensure reset
                st.session_state.last_uploaded_file_name = None
                st.session_state.kb_data_loaded = False # Force reload from KB next time if any data exists
                st.info("Processed data cleared. Upload a new file. ⬆️")
                st.rerun() # Rerun to reflect cleared state


    async def _process_single_file(self, uploaded_file):
        """
        Orchestrates the entire file processing pipeline:
        1. Reads and extracts data from the uploaded file.
        2. Masks PII and prompts user for consent.
        3. Categorizes transactions.
        4. Analyzes financial health.
        5. Stores results in session state AND to SQLite.
        
        Args:
            uploaded_file: The file object from Streamlit's file_uploader.
        """
        # Only proceed to file processing if we haven't confirmed the masking preview yet
        if not st.session_state.masking_preview_confirmed:
            with st.spinner("⏳ Extracting and Structuring Data... Please wait."): 
                try:
                    # Step 1: File Reading and Initial Column Detection
                    st.markdown("### Step 1: Extracting and Structuring Data... 📊")
                    raw_df, file_processing_result = self.file_processor.process_uploaded_file(uploaded_file)

                    if not file_processing_result.success or raw_df is None or raw_df.empty:
                        st.error(f"❌ File processing failed: {file_processing_result.error_message}") 
                        st.session_state.file_processed = False # Mark as not processed if error
                        return

                    st.success("✅ Data extracted and columns detected! ✨") 
                    st.json(file_processing_result.data) # Show basic file summary

                    # Temporarily store raw_df and masked_df to session state for _render_masking_preview to use
                    st.session_state._temp_raw_df = raw_df
                    st.session_state._temp_masked_df = self.masker.mask_dataframe(raw_df)[1] # Get only masked_df
                    
                    st.markdown("### Step 2: Review PII Masking 🕵️‍♀️")
                    pii_summary = self.masker.get_pii_summary(raw_df) # Use raw_df for summary
                    if pii_summary:
                        st.warning(f"⚠️ Sensitive data detected and masked: {', '.join(pii_summary.keys())}. Please review carefully.") 
                        st.json(pii_summary) 
                    else:
                        st.info("✅ No common PII patterns detected. Data appears clean.") 

                    # Call the masking preview; it will set masking_preview_confirmed when button is pressed
                    self._render_masking_preview(st.session_state._temp_raw_df, st.session_state._temp_masked_df)
                    
                    # If the user has not clicked 'Proceed with Analysis' yet, stop execution here
                    if not st.session_state.masking_preview_confirmed:
                        st.stop() # Stop the script until a rerun is triggered by user interaction
                    
                    # If we reach here, it means masking_preview_confirmed is True on a rerun
                    # So, we can clear the temporary DFs
                    del st.session_state._temp_raw_df
                    del st.session_state._temp_masked_df

                except Exception as e:
                    st.error(f"🐛 An unexpected error occurred during file extraction/masking: {e}") 
                    st.exception(e) 
                    st.session_state.file_processed = False # Mark as not processed if error
                    return

        # If masking_preview_confirmed is True, proceed with the rest of the analysis
        if st.session_state.masking_preview_confirmed:
            with st.spinner("⏳ Analyzing financial data..."):
                try:
                    # Retrieve the masked_df from session state (or re-mask if needed)
                    masked_df_for_analysis = st.session_state.get('_temp_masked_df_storage') 
                    if masked_df_for_analysis is None:
                        if '_temp_raw_df' in st.session_state and st.session_state._temp_raw_df is not None:
                            _, masked_df_for_analysis = self.masker.mask_dataframe(st.session_state._temp_raw_df)
                        else:
                            st.error("❌ Masked data not found in session for analysis after consent. Please clear and re-upload.")
                            st.session_state.file_processed = False # Mark as not processed if error
                            return
                        if '_temp_raw_df' in st.session_state:
                            del st.session_state._temp_raw_df
                        if '_temp_masked_df' in st.session_state: 
                            del st.session_state._temp_masked_df


                    # Update consent message based on what user actually selected
                    if not st.session_state.user_consent:
                        st.info("Analysis proceeding without anonymous data storage for model improvement (consent not given). 🚫")
                    else:
                        st.success("✅ Analysis proceeding with consent for anonymous data storage for model improvement.")


                    # Step 3: Categorize Transactions
                    st.markdown("### Step 3: Categorizing Transactions... 🏷️")
                    categorized_transactions, categorization_result = self.categorizer.categorize_dataframe(masked_df_for_analysis)

                    if not categorization_result.success or not categorized_transactions:
                        st.error(f"🛑 Transaction categorization failed: {categorization_result.error_message}") 
                        st.session_state.file_processed = False # Mark as not processed if error
                        return # Stop if categorization critically fails
                    st.success(f"🎉 Transactions categorized! ({categorization_result.data.get('categorized_count', 0)} categorized)") 
                    st.info(f"💡 Categorization reasoning: {categorization_result.reasoning}")


                    # Step 4: Analyze Financial Health
                    st.markdown("### Step 4: Analyzing Financial Health... ❤️‍🩹")
                    health_metrics = self.health_analyzer.analyze_financial_health(categorized_transactions)
                    
                    if health_metrics is None: # Health analyzer returns None on critical failure
                        st.error("💔 Financial health analysis failed.") 
                        st.session_state.file_processed = False # Mark as not processed if error
                        return # Stop if health analysis critically fails
                    st.success(f"🌟 Financial health analyzed! Overall Score: {health_metrics.overall_score:.1f}/100") 
                    
                    # Save processed data to session state (now they are EnhancedTransaction and FinancialHealthMetrics objects)
                    st.session_state.transactions = categorized_transactions
                    st.session_state.health_metrics = health_metrics
                    st.session_state.file_processed = True # Mark all processing as complete

                    # Initialize or update user profile with consent information for conversation agent
                    user_profile = self.conversation_agent.user_profile # Get current user profile
                    if st.session_state.user_consent:
                        user_profile.conversation_preferences['data_sharing_consent'] = True
                    self.conversation_agent.set_user_profile(user_profile)

                    # --- Save to SQLite Knowledge Base ---
                    if self.kb.is_ready():
                        st.info("☁️ Saving processed data to local knowledge base...")
                        # Convert EnhancedTransaction objects back to dicts for storage
                        await self.kb.add_transactions([tx.to_dict() for tx in categorized_transactions])
                        await self.kb.save_user_data(user_profile.to_dict(), health_metrics.to_dict())
                        st.success("✅ Data successfully saved to local knowledge base.")
                    else:
                        st.warning("⚠️ Knowledge Base not ready. Data not saved to local storage.")

                    st.success("🥳 All processing complete! Redirecting to Dashboard...") 
                    st.balloons()
                    time.sleep(2) # Give user a moment to see success message
                    st.rerun() # Rerun to move to the main dashboard tabs


                except Exception as e:
                    # Catch any unexpected errors during the entire processing pipeline
                    st.error(f"💥 An unexpected error occurred during analysis: {e}") 
                    st.exception(e) # Display full traceback for debugging
                    st.session_state.file_processed = False # Mark as not processed if error

    def _render_masking_preview(self, original_df: pd.DataFrame, masked_df: pd.DataFrame):
        """
        Renders a side-by-side preview of the original and masked dataframes,
        and collects user consent to proceed with analysis.
        Sets st.session_state.masking_preview_confirmed when "Proceed" is clicked.
        
        Args:
            original_df (pd.DataFrame): The DataFrame with original data.
            masked_df (pd.DataFrame): The DataFrame with PII masked.
        """
        st.markdown("## 🔍 PII Masking Preview") 
        st.warning("Please review the masked data below. Only the **masked version** will be used for AI analysis. 🧐")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Original Data (Preview)")
            st.dataframe(original_df.head(5).select_dtypes(include=['object', 'int64', 'float64']), use_container_width=True)
            st.caption("This data is stored temporarily and NOT shared with AI. 🔒")

        with col2:
            st.markdown("### 🕵️ Masked Data (Sent to AI)")
            st.dataframe(masked_df.head(5).select_dtypes(include=['object', 'int64', 'float64']), use_container_width=True)
            st.caption("This masked version is used for all AI analysis. 🤖")

        st.markdown("---")
        st.markdown("#### ✅ Your Privacy Consent")
        
        # This checkbox only sets the consent status, does not control flow immediately
        consent_checkbox_state = st.checkbox(
            "I consent to my masked financial data being used for analysis by the AI model. "
            "I **optionally** consent to the anonymous use of this masked data to improve model performance.",
            value=st.session_state.get('user_consent', False), # Retain state if already checked
            key="pii_consent_checkbox"
        )
        # Update session state with the current checkbox value
        st.session_state.user_consent = consent_checkbox_state
        
        # Store masked_df persistently *after* consent is shown, but before proceeding
        # This ensures the masked_df is available on the next rerun when masking_preview_confirmed is True
        st.session_state._temp_masked_df_storage = masked_df # New temporary key for persistence

        # This button controls the flow: when clicked, it triggers a rerun and sets the flag
        proceed_button = st.button("🚀 Proceed with Analysis", key="proceed_analysis_button") 
        
        if proceed_button:
            # Set the confirmation flag in session state
            st.session_state.masking_preview_confirmed = True
            st.rerun() # Trigger a rerun to proceed with the next steps of _process_single_file
        # No explicit return value needed, as flow is controlled by session_state and st.stop()/st.rerun

    # Conceptual feedback function for RL integration
    def _send_feedback_for_rl(self, event_data: Dict[str, Any], user_consent: bool):
        """
        (Conceptual) Sends feedback data to a potential external system
        for reinforcement learning or model improvement, based on user consent.
        """
        # This would typically interact with a backend service or a separate
        # data logging/processing module that implements the RL feedback loop.
        # For this hackathon, it's a print statement.
        if user_consent:
            print(f"Feedback (for RL) sent: {event_data} (User consented)")
        else:
            print(f"Feedback (for RL) NOT sent: {event_data} (User did NOT consent)")

    def _play_ai_response_audio(self, text: str):
        """Plays the AI's response as audio using gTTS."""
        if VOICE_SUPPORT_AVAILABLE:
            try:
                tts = gTTS(text=text, lang='en')
                fp = BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0) # Rewind to the beginning of the BytesIO object
                st.audio(fp, format='audio/mp3', start_time=0)
            except Exception as e:
                st.error(f"🗣️❌ Error playing AI response audio: {e}")
        else:
            st.warning("🎤🚫 Voice playback is disabled because libraries are not available.")

    def _render_main_dashboard(self):
        """
        Renders the main dashboard with multiple tabs for different features
        once transaction data is available.
        """
        if not st.session_state.file_processed or not st.session_state.transactions:
            st.info("Upload your bank statement in the sidebar to begin financial analysis and insights. ⬆️")
            return

        tab1, tab2, tab3, tab4 = st.tabs([
            "💖 Dashboard Financial Health", 
            "🧠 AI Analysis", 
            "💬 Chat AI Advisor", 
            "📈 Trends" 
        ])

        with tab1:
            self._render_health_dashboard()
        with tab2:
            self._render_ai_analysis()
        with tab3:
            self._render_ai_conversation()
        with tab4:
            self._render_trend_analysis()

    def _render_health_dashboard(self):
        """
        Renders the Financial Health Dashboard tab, displaying key metrics
        and LLM-generated strengths/weaknesses.
        """
        metrics: Optional[FinancialHealthMetrics] = st.session_state.health_metrics
        if not metrics:
            st.warning("⚠️ No financial health metrics found. Please ensure a file is processed successfully.")
            return

        st.markdown("## 📊 Financial Health Dashboard") 

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{metrics.overall_score:.1f}/100", delta=metrics.risk_level)
        with col2:
            st.metric("Cashflow Score", f"{metrics.cashflow_score:.1f}%")
        with col3:
            st.metric("Savings Rate", f"{metrics.savings_ratio_score:.1f}%")
        with col4:
            # Emergency fund score is 0-100. Convert to months for display.
            emergency_fund_months = round(metrics.emergency_fund_score / (100/6), 1)
            st.metric("Emergency Fund", f"{emergency_fund_months:.1f}/6 months", 
                      help="Months of average expenses covered by current savings/cashflow.")

        # Display LLM insights (Strengths, Weaknesses, Overall Assessment)
        if metrics.llm_insights and isinstance(metrics.llm_insights, dict):
            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("### 💪 Strengths") 
                strengths = metrics.llm_insights.get("strengths", [])
                if strengths:
                    for i, strength in enumerate(strengths[:5]): # Limit to top 5
                        st.success(f"{i+1}. {strength}")
                else:
                    st.info("🤷‍♀️ No specific strengths identified by AI yet.")

            with col_w:
                st.markdown("### 📉 Weaknesses") 
                weaknesses = metrics.llm_insights.get("weaknesses", [])
                if weaknesses:
                    for i, weakness in enumerate(weaknesses[:5]): # Limit to top 5
                        st.warning(f"{i+1}. {weakness}")
                else:
                    st.info("🤷‍♂️ No specific weaknesses identified by AI yet.")
            
            overall_assessment = metrics.llm_insights.get("overall_assessment")
            if overall_assessment:
                st.info(f"💬 AI Overall Assessment: {overall_assessment}") 
            else:
                st.info("ℹ️ No overall AI assessment available.")
        else:
            st.info("Detailed AI insights for financial health are not available (LLM not configured or analysis failed).")


    def _render_ai_analysis(self):
        """
        Renders the AI-Powered Insights tab, displaying behavioral observations
        and risk factors from the LLM.
        """
        metrics: Optional[FinancialHealthMetrics] = st.session_state.health_metrics
        if not metrics or not metrics.llm_insights or not isinstance(metrics.llm_insights, dict):
            st.warning("⚠️ No AI-powered insights available. Please ensure a file is processed and LLM is configured.")
            return

        st.markdown("## 💡 AI-Powered Insights") 
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👀 Behavioral Observations") 
            observations = metrics.llm_insights.get("behavioral_insights", [])
            if observations:
                for i, obs in enumerate(observations[:5]): # Limit to top 5
                    st.write(f"- {obs}") 
            else:
                st.info("🤷‍♀️ No behavioral insights available from AI.")
        
        with col2:
            st.markdown("#### 🚩 Risk Factors") 
            risks = metrics.llm_insights.get("risk_factors", [])
            if risks:
                for i, risk in enumerate(risks[:5]): # Limit to top 5
                    st.warning(f"- {risk}") 
            else:
                st.info("🤷‍♂️ No risk factors identified by AI.")

        # Optional: Add key metrics interpretation if available
        key_metrics_interpretation = metrics.llm_insights.get("key_metrics_interpretation")
        if key_metrics_interpretation:
            st.markdown("#### 📈 Trends Key Metrics Interpretation") 
            st.info(key_metrics_interpretation)
        else:
            st.info("ℹ️ No detailed key metrics interpretation available from AI.")


    def _play_ai_response_audio(self, text: str):
        """Plays the AI's response as audio using gTTS."""
        if VOICE_SUPPORT_AVAILABLE:
            try:
                tts = gTTS(text=text, lang='en')
                fp = BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0) # Rewind to the beginning of the BytesIO object
                st.audio(fp, format='audio/mp3', start_time=0)
            except Exception as e:
                st.error(f"🗣️❌ Error playing AI response audio: {e}")
        else:
            st.warning("🎤🚫 Voice playback is disabled because libraries are not available.")

    def _render_ai_conversation(self):
        """
        Renders the AI Financial Advisor interface, displaying conversation history
        and allowing users to chat with the AI. Includes voice input/output.
        """
        st.markdown("## 💬 Chat AI Financial Advisor") 

        if not llm_manager.is_available():
            st.warning("⚠️ The AI advisor requires an LLM. Please configure your LLM API key in the setup wizard.") 
            return # Disable chat if LLM is not available

        # Voice Mode Toggle
        if VOICE_SUPPORT_AVAILABLE:
            st.session_state.voice_mode = st.checkbox("🎙️ Enable Voice Input/Output", value=st.session_state.voice_mode, key="voice_mode_toggle")

        # Display conversation history
        conversation_history = st.session_state.get('conversation_history', [])
        if conversation_history:
            st.markdown("### 🗣️ Your Conversation History")
            # Display messages from oldest to newest to read conversation naturally
            for chat in conversation_history:
                with st.chat_message("user"):
                    st.markdown(chat['user_query'])
                with st.chat_message("assistant"):
                    st.markdown(chat['ai_response'])
                    # Play audio for AI response if voice mode is on
                    if st.session_state.voice_mode and chat.get('ai_response'): # Check if AI response exists
                        self._play_ai_response_audio(chat['ai_response'])

                    # Display metadata (emotion, intent, topics) below AI response
                    emotion_emoji_map = {
                        "stressed": "😥", "anxious": "😟", "confused": "🤔",
                        "excited": "🤩", "confident": "😎", "motivated": "💪",
                        "neutral": "😐", "frustrated": "😠", "curious": "�",
                        "happy": "😊", "sad": "😔" 
                    }
                    emotion_emoji = emotion_emoji_map.get(chat.get('emotion', 'neutral'), "❓")
                    
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.caption(f"Emotion: {emotion_emoji} {chat.get('emotion', 'unknown').replace('_', ' ').title()}")
                    with col_meta2:
                        st.caption(f"🎯 Target Intent: {chat.get('intent', 'general').replace('_', ' ').title()}") 
                    with col_meta3:
                        topics = ', '.join([t.replace('_', ' ').title() for t in chat.get('topics', ['general'])])
                        st.caption(f"📚 Topics: {topics}") 
                    
                    # Display follow-up suggestions if available
                    if chat.get('follow_up_suggestions'):
                        st.markdown("---")
                        st.write("🤔 **Next Questions:** " + " | ".join(chat['follow_up_suggestions']))
                        st.markdown("---")
        else:
            st.info("📝 Start a conversation by typing your financial question below!")

        # Chat input area at the bottom, dynamically changing based on voice mode
        user_query_input = None # This variable will hold the text from either input method
        if st.session_state.voice_mode:
            st.write("Click 'Speak' and talk to the AI advisor. Ensure your microphone is enabled. 🎙️")
            speak_button = st.button("Speak 🎙️", key="voice_input_button")
            
            if speak_button:
                r = sr.Recognizer()
                with st.spinner("👂 Listening for your query..."):
                    try:
                        with sr.Microphone() as source:
                            r.adjust_for_ambient_noise(source)
                            audio = r.listen(source)
                        user_query_input = r.recognize_google(audio)
                        st.write(f"💬 **You said:** {user_query_input}") # Display what was heard
                    except sr.UnknownValueError:
                        st.error("👂❌ Sorry, I could not understand your audio. Please speak more clearly or rephrase.")
                    except sr.RequestError as e:
                        st.error(f"🌐❌ Could not request results from Google Speech Recognition service; check your internet connection and API key: {e}")
                    except Exception as e:
                        st.error(f"🐛 An error occurred during voice input: {e}. Please ensure PyAudio is correctly installed if on Linux/Windows.")
        else:
            # Standard text input
            user_query_input = st.chat_input("Ask a question about your finances (e.g., 'How can I save more?', 'What's my biggest expense?')...", key="text_chat_input")
        
        # Process input (either from text_input or voice_input)
        if user_query_input: # Only proceed if there's actual input
            # --- FIX FOR TypeError: argument after ** must be a mapping ---
            transactions_list_from_session = st.session_state.get('transactions', [])
            transactions_objects = []
            if transactions_list_from_session:
                # Check if the first element is already an EnhancedTransaction object
                if isinstance(transactions_list_from_session[0], EnhancedTransaction):
                    transactions_objects = transactions_list_from_session
                else: # Assume it's a list of dictionaries and convert
                    try:
                        transactions_objects = [EnhancedTransaction.from_dict(tx) for tx in transactions_list_from_session]
                    except Exception as e:
                        st.error(f"Error converting session transactions to objects: {e}. Some chat context may be missing.")
                        transactions_objects = [] # Fallback to empty if conversion fails
            # --- END FIX ---

            # --- FIX FOR health_metrics_obj instantiation ---
            health_metrics_from_session = st.session_state.get('health_metrics')
            health_metrics_obj: Optional[FinancialHealthMetrics] = None
            if health_metrics_from_session:
                if isinstance(health_metrics_from_session, FinancialHealthMetrics):
                    health_metrics_obj = health_metrics_from_session
                else: # Assume it's a dictionary and convert
                    try:
                        health_metrics_obj = FinancialHealthMetrics.from_dict(health_metrics_from_session)
                    except Exception as e:
                        st.error(f"Error converting session health metrics to object: {e}. Some chat context may be missing.")
                        health_metrics_obj = FinancialHealthMetrics(0,"N/A",0,0,0,0) # Fallback default
            else:
                health_metrics_obj = FinancialHealthMetrics(0,"N/A",0,0,0,0) # Default if no metrics in session
            # --- END FIX ---

            context = {
                'transactions': transactions_objects,
                'health_metrics': health_metrics_obj,
                'user_profile': self.conversation_agent.user_profile.to_dict() # Pass current user profile
            }
            # Generate AI response
            conversation_entry = self.conversation_agent.generate_response(user_query_input, context)
            
            # Store conversation entry in session state
            st.session_state.conversation_history.append(conversation_entry.to_dict())

            # --- Save conversation entry to SQLite Knowledge Base ---
            if self.kb.is_ready():
                st.info("☁️ Saving conversation to local knowledge base...")
                # Use asyncio.run to execute the async method within this sync function
                asyncio.run(self.kb.add_conversation_entry(conversation_entry.to_dict()))
            else:
                st.warning("⚠️ Knowledge Base not ready. Conversation not saved to local storage.")
            
            st.rerun() # Rerun to update the chat history display

    def _render_trend_analysis(self):
        """
        Renders the Trend Analysis tab, displaying financial health trends
        over time and LLM-generated insights on trajectory, drivers, etc.
        """
        if not st.session_state.transactions:
            st.info("⬆️ Upload your bank statement to see trend analysis.")
            return

        # Ensure transactions are EnhancedTransaction objects before passing to trend analyzer
        transactions_list_from_session = st.session_state.get('transactions', [])
        transactions = []
        if transactions_list_from_session:
            if isinstance(transactions_list_from_session[0], EnhancedTransaction):
                transactions = transactions_list_from_session
            else:
                try:
                    transactions = [EnhancedTransaction.from_dict(tx) for tx in transactions_list_from_session]
                except Exception as e:
                    st.error(f"Error converting transactions for trend analysis: {e}. Trend data may be inaccurate.")
                    transactions = []

        with st.spinner("⏳ Calculating financial trends..."):
            trend_results = self.trend_analyzer.analyze_trends(transactions)

        if trend_results.get('status') == 'success':
            trend_analysis = trend_results['trend_analysis']
            period_metrics = trend_results['period_metrics']

            st.markdown("## 📈 Financial Health Trends") 

            # Display LLM's trajectory assessment
            st.markdown("### 🧭 Trajectory Assessment") 
            trajectory_assessment = trend_analysis.get('trajectory_assessment')
            if trajectory_assessment:
                st.info(trajectory_assessment)
            else:
                st.info("ℹ️ No detailed trajectory assessment available from AI.")


            # Display LLM's positive developments and warning signs
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Positive Developments") 
                positive_developments = trend_analysis.get('positive_developments', [])
                if positive_developments:
                    for dev in positive_developments:
                        st.success(f"- {dev}") 
                else:
                    st.info("🤷‍♀️ No specific positive developments identified by AI.")
            with col2:
                st.markdown("### 🛑 Warning Signs") 
                warning_signs = trend_analysis.get('warning_signs', [])
                if warning_signs:
                    for warning in warning_signs:
                        st.warning(f"- {warning}") 
                else:
                    st.info("🤷‍♂️ No specific warning signs identified by AI.")
            
            st.markdown("### 💡 AI Recommendations") 
            recommendations = trend_analysis.get('recommendations', [])
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}") 
            else:
                st.info("ℹ️ No specific recommendations from AI for trend improvement.")


            # Plot financial health trend over time
            st.markdown("### 📊 Financial Health Score Over Time") 
            if period_metrics:
                periods = [p['period'] for p in period_metrics]
                scores = [p['metrics']['overall_score'] for p in period_metrics]

                # Create DataFrame for Plotly
                plot_df = pd.DataFrame({'Period': periods, 'Health Score': scores})
                
                fig = px.line(plot_df, x='Period', y='Health Score', 
                              title="Financial Health Score Over Time (Monthly)", markers=True)
                
                # Set y-axis range dynamically if all scores are not zero, otherwise default to 0-100
                min_score = min(scores) if scores else 0
                max_score = max(scores) if scores else 100
                if min_score == 0 and max_score == 0:
                     fig.update_layout(yaxis_range=[0, 100], yaxis_title="Health Score")
                else:
                    # Add a little padding to the range
                    y_min = max(0, min_score * 0.9)
                    y_max = min(100, max_score * 1.1)
                    fig.update_layout(yaxis_range=[y_min, y_max], yaxis_title="Health Score")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("📉 Not enough historical data to plot trends.")

        elif trend_results.get('status') == 'insufficient_data':
            st.warning(f"🗓️ Insufficient data for trend analysis: {trend_results['trend_analysis'].get('analysis', 'Need at least 2 months of data for trend detection.')}")
            st.info("📊 Upload more transaction data covering multiple months to enable comprehensive trend analysis.")
        elif trend_results.get('error'):
            st.error(f"❌ Trend analysis failed: {trend_results['error']}") 
        else:
            st.warning("ℹ️ No trend data available yet.")


if __name__ == "__main__":
    # Import necessary modules for SQLite outside of class scope
    # These are typically available in the environment for Streamlit apps
    import os
    import uuid # For generating anonymous user IDs
    
    app = EmpowerFinApplication()
    app.run()