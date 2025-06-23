# knowledge_base.py - Local SQLite Database Integration for Financial Data Persistence
import json
import os
import uuid
import datetime
import sqlite3 # Import SQLite
import numpy as np # Still needed for serialization helper
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st # Used for st.warning/error messages for user feedback

class FinancialKnowledgeBase:
    """
    Manages persistent storage and retrieval of financial data, metrics,
    and conversation history using a local SQLite database.

    This class handles database initialization, user session management,
    and CRUD-like operations for various data types.
    """

    def __init__(self):
        self.conn = None
        self.cursor = None
        
        self.app_id = os.getenv('__app_id', 'default-app-id') # Fallback if not set
        self.user_id = None # Will be set during authentication

        self.is_initialized = False

        print(f"[KB Debug] Initializing FinancialKnowledgeBase...")
        print(f"[KB Debug] App ID: {self.app_id}")

        try:
            # Authenticate user (just gets a session ID for local scoping)
            self._authenticate_user()
            if not self.user_id:
                st.error("? User ID not determined. Local SQLite cannot scope data.")
                print("[KB Error] User ID not available for SQLite initialization. Exiting init.")
                return

            # Define database file path
            # Create a dedicated directory for SQLite databases per app/user for cleanliness
            db_dir = f"./data/{self.app_id}/{self.user_id}"
            
            print(f"[KB Debug] Attempting to create DB directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True) # Ensure directory exists
            print(f"[KB Debug] Directory creation command issued: os.makedirs('{db_dir}', exist_ok=True)")

            self.db_path = os.path.join(db_dir, "financial_data.db")
            print(f"[KB Debug] Full database file path: {self.db_path}")

            # Initialize SQLite connection and create tables
            self._initialize_database()
            
            self.is_initialized = True
            print(f"[KB] Local SQLite database initialized for user: {self.user_id} at {self.db_path}")

        except Exception as e:
            st.error(f"? Error initializing Local SQLite Database: {e}")
            self.is_initialized = False
            print(f"[KB Error] Local SQLite Database initialization failed: {e}")
            # Optionally print full traceback for more context during debugging
            import traceback
            traceback.print_exc()

    def _initialize_database(self):
        """
        Establishes SQLite connection and creates necessary tables if they don't exist.
        """
        print(f"[KB Debug] Attempting to connect to SQLite database: {self.db_path}")
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print("[KB Debug] SQLite connection established.")

            # Create user_data table for user profiles and latest health metrics
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS user_data (
                    user_id TEXT PRIMARY KEY,
                    user_profile_json TEXT,
                    health_metrics_json TEXT,
                    last_updated TEXT
                )
            """)
            print("[KB Debug] 'user_data' table checked/created.")

            # Create transactions table
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS transactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    date TEXT,
                    description TEXT,
                    amount REAL,
                    category TEXT,
                    subcategory TEXT,
                    spending_type TEXT,
                    confidence REAL,
                    reasoning TEXT,
                    balance REAL,
                    reference TEXT,
                    timestamp TEXT
                )
            """)
            print("[KB Debug] 'transactions' table checked/created.")

            # Create conversations table
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    user_query TEXT,
                    ai_response TEXT,
                    emotion TEXT,
                    intent TEXT,
                    topics_json TEXT,
                    confidence TEXT,
                    follow_up_suggestions_json TEXT,
                    timestamp TEXT
                )
            """)
            print("[KB Debug] 'conversations' table checked/created.")
            self.conn.commit()
            print("[KB] SQLite tables checked/created successfully.")
        except sqlite3.Error as e:
            print(f"[KB Error] SQLite database initialization error in _initialize_database: {e}")
            raise Exception(f"SQLite database initialization error: {e}")
        except Exception as e:
            print(f"[KB Error] Unexpected error in _initialize_database: {e}")
            raise # Re-raise unexpected errors

    def _authenticate_user(self):
        """
        Authenticates the user by assigning a unique session ID.
        This ID is used to scope data within the local SQLite database.
        """
        if 'kb_user_id' not in st.session_state:
            st.session_state.kb_user_id = str(uuid.uuid4())
            print(f"[KB Auth] Generated new session ID: {st.session_state.kb_user_id}")
        self.user_id = st.session_state.kb_user_id
        print(f"[KB Auth] Current session User ID: {self.user_id}")

    def is_ready(self) -> bool:
        """Checks if the knowledge base (SQLite connection) is initialized and ready."""
        return self.is_initialized and self.conn is not None and self.cursor is not None and self.user_id is not None

    async def save_user_data(self, user_profile: Dict, health_metrics: Dict):
        """
        Saves core user profile and latest financial health metrics to the user_data table.
        """
        if not self.is_ready():
            print("[KB Save User] KB not ready.")
            return

        try:
            user_profile_json = json.dumps(user_profile, default=self._serialize_datetime_for_json)
            health_metrics_json = json.dumps(health_metrics, default=self._serialize_datetime_for_json)
            last_updated = datetime.now().isoformat()

            # UPSERT operation: Try to update, if not found, insert
            self.cursor.execute(f"""
                INSERT OR REPLACE INTO user_data (user_id, user_profile_json, health_metrics_json, last_updated)
                VALUES (?, ?, ?, ?)
            """, (self.user_id, user_profile_json, health_metrics_json, last_updated))
            self.conn.commit()
            print(f"[KB] User data (profile & metrics) saved for {self.user_id}")
            st.success("? User profile and latest metrics saved to local database.")
        except sqlite3.Error as e:
            st.error(f"? Error saving user data to SQLite: {e}")
            print(f"[KB Error] Failed to save user data to SQLite: {e}")
        except Exception as e:
            st.error(f"? Unexpected error saving user data: {e}")
            print(f"[KB Error] Unexpected error saving user data: {e}")

    async def load_user_data(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Loads core user profile and latest financial health metrics from the user_data table.
        """
        if not self.is_ready():
            print("[KB Load User] KB not ready.")
            return None, None

        try:
            self.cursor.execute("SELECT user_profile_json, health_metrics_json FROM user_data WHERE user_id = ?", (self.user_id,))
            row = self.cursor.fetchone()
            if row:
                user_profile = json.loads(row[0]) if row[0] else None
                health_metrics = json.loads(row[1]) if row[1] else None
                print(f"[KB] User data loaded from SQLite for {self.user_id}")
                return user_profile, health_metrics
            else:
                print(f"[KB] No user data found in SQLite for {self.user_id}")
                return None, None
        except sqlite3.Error as e:
            st.error(f"? Error loading user data from SQLite: {e}")
            print(f"[KB Error] Failed to load user data from SQLite: {e}")
            return None, None
        except Exception as e:
            st.error(f"? Unexpected error loading user data: {e}")
            print(f"[KB Error] Unexpected error loading user data: {e}")
            return None, None

    async def add_transactions(self, transactions: List[Dict]):
        """
        Adds a batch of enhanced transactions to the transactions table.
        """
        if not self.is_ready():
            print("[KB Add TXN] KB not ready.")
            return

        try:
            # Prepare data for insertion, ensuring all fields are present and correctly typed
            insert_data = []
            for tx in transactions:
                # Generate a unique ID for each transaction
                tx_id = str(uuid.uuid4())
                
                # Ensure all fields expected by the schema are present, provide defaults if missing
                insert_data.append((
                    tx_id,
                    self.user_id,
                    tx.get('date', ''),
                    tx.get('description', ''),
                    float(tx.get('amount', 0.0)),
                    tx.get('category', 'other'),
                    tx.get('subcategory', 'uncategorized'),
                    tx.get('spending_type', 'regular'),
                    float(tx.get('confidence', 0.0)),
                    tx.get('reasoning', ''),
                    float(tx.get('balance')) if tx.get('balance') is not None else None, # Allow NULL for balance
                    tx.get('reference'), # Allow NULL for reference
                    datetime.now().isoformat() # Timestamp when added
                ))
            
            self.cursor.executemany(f"""
                INSERT INTO transactions (
                    id, user_id, date, description, amount, category, subcategory,
                    spending_type, confidence, reasoning, balance, reference, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, insert_data)
            self.conn.commit()
            print(f"[KB] Added {len(transactions)} transactions to local SQLite for {self.user_id}")
            st.success(f"? {len(transactions)} transactions saved to local database.")
        except sqlite3.Error as e:
            st.error(f"? Error adding transactions to SQLite: {e}")
            print(f"[KB Error] Failed to add transactions to SQLite: {e}")
        except Exception as e:
            st.error(f"? Unexpected error adding transactions: {e}")
            print(f"[KB Error] Unexpected error adding transactions: {e}")
            
    # Helper for serialization (used by save functions)
    def _serialize_datetime_for_json(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


    async def get_transactions(self, limit: int = 1000) -> List[Dict]:
        """
        Retrieves transactions for the current user from the transactions table, ordered by date.
        """
        if not self.is_ready():
            print("[KB Get TXN] KB not ready.")
            return []

        try:
            self.cursor.execute(f"""
                SELECT id, user_id, date, description, amount, category, subcategory,
                       spending_type, confidence, reasoning, balance, reference
                FROM transactions
                WHERE user_id = ?
                ORDER BY date ASC
                LIMIT ?
            """, (self.user_id, limit))
            
            rows = self.cursor.fetchall()
            
            transactions_data = []
            for row in rows:
                tx_dict = {
                    'id': row[0], # SQLite primary key 'id'
                    'user_id': row[1],
                    'date': row[2],
                    'description': row[3],
                    'amount': row[4],
                    'category': row[5],
                    'subcategory': row[6],
                    'spending_type': row[7],
                    'confidence': row[8],
                    'reasoning': row[9],
                    'balance': row[10],
                    'reference': row[11]
                }
                # Ensure all fields match EnhancedTransaction dataclass
                # Reconstruct EnhancedTransaction object to ensure all fields are present
                # Use from_dict from data_models if available or direct dict creation
                transactions_data.append(tx_dict)
            
            print(f"[KB] Retrieved {len(transactions_data)} transactions from SQLite for {self.user_id}")
            return transactions_data
        except sqlite3.Error as e:
            st.error(f"? Error retrieving transactions from SQLite: {e}")
            print(f"[KB Error] Failed to get transactions from SQLite: {e}")
            return []
        except Exception as e:
            st.error(f"? Unexpected error retrieving transactions: {e}")
            print(f"[KB Error] Unexpected error retrieving transactions: {e}")
            return []

    async def add_conversation_entry(self, entry: Dict):
        """
        Adds a single conversation entry to the conversations table.
        """
        if not self.is_ready():
            print("[KB Add Conv] KB not ready.")
            return

        try:
            entry_id = str(uuid.uuid4())
            timestamp_str = datetime.now().isoformat()
            
            # Serialize list/dict fields to JSON strings
            topics_json = json.dumps(entry.get('topics', []))
            follow_up_suggestions_json = json.dumps(entry.get('follow_up_suggestions', []))

            self.cursor.execute(f"""
                INSERT INTO conversations (
                    id, user_id, user_query, ai_response, emotion, intent, 
                    topics_json, confidence, follow_up_suggestions_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id,
                self.user_id,
                entry.get('user_query', ''),
                entry.get('ai_response', ''),
                entry.get('emotion', 'neutral'),
                entry.get('intent', 'general_inquiry'),
                topics_json,
                entry.get('confidence', 'medium'),
                follow_up_suggestions_json,
                timestamp_str
            ))
            self.conn.commit()
            print(f"[KB] Added conversation entry to local SQLite for {self.user_id}")
        except sqlite3.Error as e:
            st.error(f"? Error adding conversation entry to SQLite: {e}")
            print(f"[KB Error] Failed to add conversation entry to SQLite: {e}")
        except Exception as e:
            st.error(f"? Unexpected error adding conversation entry: {e}")
            print(f"[KB Error] Unexpected error adding conversation entry: {e}")

    async def get_conversation_history(self, limit: int = 20) -> List[Dict]:
        """
        Retrieves the latest conversation history for the current user from the conversations table.
        """
        if not self.is_ready():
            print("[KB Get Conv] KB not ready.")
            return []

        try:
            self.cursor.execute(f"""
                SELECT user_query, ai_response, emotion, intent, topics_json,
                       confidence, follow_up_suggestions_json, timestamp
                FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (self.user_id, limit))
            
            rows = self.cursor.fetchall()
            
            history_data = []
            for row in rows:
                entry_dict = {
                    'user_query': row[0],
                    'ai_response': row[1],
                    'emotion': row[2],
                    'intent': row[3],
                    'topics': json.loads(row[4]) if row[4] else [], # Deserialize topics
                    'confidence': row[5],
                    'follow_up_suggestions': json.loads(row[6]) if row[6] else [], # Deserialize follow-up suggestions
                    'timestamp': row[7]
                }
                history_data.append(entry_dict)
            
            print(f"[KB] Retrieved {len(history_data)} conversation entries from SQLite for {self.user_id}")
            # Return in chronological order (oldest first)
            return list(reversed(history_data)) 
        except sqlite3.Error as e:
            st.error(f"? Error retrieving conversation history from SQLite: {e}")
            print(f"[KB Error] Failed to get conversation history from SQLite: {e}")
            return []
        except Exception as e:
            st.error(f"? Unexpected error retrieving conversation history: {e}")
            print(f"[KB Error] Unexpected error retrieving conversation history: {e}")
            return []
