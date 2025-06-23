EmpowerFin Guardian 2.0 📊🤖💖
EmpowerFin Guardian 2.0 is an intelligent financial analysis and advisory application built with Streamlit and Python. It helps users understand their financial health, categorize transactions, identify trends, and receive personalized advice through a conversational AI, all while prioritizing data privacy with PII masking and local data persistence.

✨ Features
Multi-format File Processing: Upload bank statements in CSV, Excel (XLSX, XLS), TXT, or PDF formats. The AI automatically detects formats and extracts structured data.

PII Masking & Privacy Consent: Automatically detects and masks Personally Identifiable Information (PII) from your bank statements. A preview is presented, and explicit user consent is required before masked data is used for AI analysis.

AI-Powered Transaction Categorization: Transactions are intelligently categorized using an LLM, providing insights into spending habits.

Financial Health Analysis: Calculates an overall financial health score, cashflow, savings rate, and emergency fund status.

LLM-Generated Insights: Provides detailed AI analysis including strengths, weaknesses, behavioral observations, and risk factors based on your financial data.

Financial Health Trend Analysis: Identifies financial trajectories, positive developments, warning signs, and offers AI recommendations for improvement over time.

Conversational AI Financial Advisor: Interact with an AI advisor using natural language (text or voice) to ask questions about your finances, get personalized advice, and understand your spending.

Voice Input/Output: (If PyAudio and gTTS are installed) Engage with the AI advisor using your voice and receive spoken responses.

Local Data Persistence (SQLite): Your processed financial data (transactions, health metrics, conversation history) is securely saved locally to an SQLite database, ensuring your information persists across sessions.

🚀 Architecture Overview
EmpowerFin Guardian 2.0 is structured using a modular agent-based architecture within a Streamlit framework:

main_application.py: The core Streamlit application, orchestrating the UI, managing session state, and coordinating all other agents.

file_processor_agent.py: Handles reading and parsing bank statements from various file formats into a standardized DataFrame.

data_masker.py: Implements PII detection and masking capabilities to protect sensitive user information.

categorization_agent.py: Utilizes LLMs to categorize raw transactions into meaningful financial categories.

health_analyzer_agent.py: Calculates various financial health metrics and generates LLM-powered insights (strengths, weaknesses, etc.).

trend_analyzer_agent.py: Analyzes historical transaction data to identify financial trends and provides AI-driven recommendations.

conversational_agent.py: Manages the interaction with the LLM for chat-based financial advice, including intent recognition, emotion detection, and generating follow-up suggestions.

data_models.py: Defines Pydantic data models for structured data like EnhancedTransaction, FinancialHealthMetrics, UserProfile, and ConversationEntry, ensuring data consistency.

config_manager.py: Manages application-wide configurations, LLM setup, and Streamlit session state initialization.

knowledge_base.py: (The new addition) Manages the local persistence of user data, transactions, and conversation history using an SQLite database.

🛠️ Setup Instructions
To get EmpowerFin Guardian 2.0 up and running on your local machine, follow these steps:

1. Prerequisites
Python 3.8+: Ensure you have a compatible Python version installed.

pip: Python's package installer, usually comes with Python.

2. Clone the Repository (Conceptual)
(If you are running this in a cloud environment, cloning might not be necessary. If on your local machine, typically you would clone the project repository.)

git clone <repository_url>
cd EmpowerFin_Guardian_2.0 # Or whatever your project directory is named

3. Install Dependencies
Navigate to your project root directory in the terminal and install the required Python packages.

pip install -r requirements.txt

Note on requirements.txt: This file should contain all the necessary libraries including:
streamlit, pandas, plotly, python-docx, openpyxl, PyPDF2, python-dotenv, speechrecognition, gtts, pydantic, sentence-transformers (if using HuggingFaceEmbeddings).
Crucially, for voice input on Windows, you might need special steps for PyAudio:
If pip install PyAudio fails, you might need to download a pre-compiled wheel file.

Go to Gohlke's Python Extension Packages for Windows.

Download the .whl file matching your Python version (e.g., cp39 for Python 3.9) and system architecture (win_amd64 for 64-bit).

Then, install it using pip: pip install path/to/PyAudio‑0.x.x‑cpXX‑cpXX‑win_amd64.whl

4. Configure Environment Variables (LLM API Keys)
For AI-powered features, you need to provide an API key for at least one supported Large Language Model. Create a .env file in the root directory of your project (or set them as system environment variables) with one of the following:

# Example for OpenAI
OPENAI_API_KEY="your_openai_api_key_here"

# Or for Groq
# GROQ_API_KEY="your_groq_api_key_here"

# Or for Anthropic
# ANTHROPIC_API_KEY="your_anthropic_api_key_here"

The application uses python-dotenv to load these variables.

▶️ How to Run
Once all dependencies are installed and environment variables are set, run the Streamlit application from your terminal in the project's root directory:

streamlit run main_application.py

This will open the application in your default web browser.

🧑‍💻 Usage Guide
Upload Bank Statement:

Use the "File Upload Bank Statement" section in the sidebar.

Select a CSV, Excel, TXT, or PDF file.

Click "Process Uploaded File".

PII Masking Preview & Consent:

The application will display a "PII Masking Preview" showing snippets of your original and masked data.

Review the masked data.

Check the consent box "I consent to my masked financial data being used for analysis..."

Click "Proceed with Analysis".

View Dashboards:

Once processing is complete, the main area will switch to a tabbed dashboard.

"💖 Dashboard Financial Health": See your overall financial score, cashflow, savings rate, emergency fund status, and AI-identified strengths and weaknesses.

"🧠 AI Analysis": Dive deeper into AI-powered behavioral observations, risk factors, and key metrics interpretation.

"💬 Chat AI Advisor": Engage with the AI. Type your financial questions or, if enabled, use the "🎙️ Speak" button for voice input. The AI will respond based on your data and general financial knowledge.

"📈 Trends": Visualize your financial health score over time, trajectory assessments, positive developments, warning signs, and AI recommendations for trends.

Clear Data:

Use the "🧹 Clear All Data" button in the sidebar to remove all processed data from the current session and local database, allowing you to start fresh.

💾 Data Persistence (Local SQLite)
EmpowerFin Guardian 2.0 uses a local SQLite database to store your financial data and conversation history persistently.

Location: The database files (financial_data.db) are created in a structured directory within your project: ./data/{__app_id}/{your_session_user_id}/.

{__app_id}: This will typically be default-app-id unless your environment explicitly sets __app_id.

{your_session_user_id}: A unique identifier (UUID) generated for your Streamlit session. This ensures data separation if multiple users were to run the app on the same machine.

What's Stored:

user_data table: Your latest user profile and financial health metrics.

transactions table: All processed and categorized transactions.

conversations table: Your chat history with the AI advisor.

Persistence: Data stored in this SQLite database will remain even if you close and restart the Streamlit application. When you launch the app again (and it detects the same session user_id), it will attempt to load your previous data.

Important Note: Since this is a local SQLite database, the data is stored on the machine where the Streamlit application is running. It is not cloud-synced and is tied to the specific user_id and file path on that machine. If you move the project directory or run it on a different machine, your previous data will not automatically transfer unless you manually copy the data directory.

🔮 Future Enhancements (Ideas)
User Authentication: Implement proper user login/registration for multi-user support with dedicated user accounts (instead of session-based user_id).

Cloud Database Option: Provide a configuration option to use a cloud database (like Firestore, PostgreSQL, etc.) for cross-device data synchronization.

Advanced Goal Tracking: Allow users to set specific financial goals and track progress against them.

Budgeting Tools: Integrate interactive budgeting features based on categorized spending.

Predictive Analytics: Use machine learning to forecast future cashflows or identify potential financial risks.

Custom Category Management: Allow users to define their own transaction categories.

More LLM Models: Expand support for a wider range of LLMs and embedding models.
