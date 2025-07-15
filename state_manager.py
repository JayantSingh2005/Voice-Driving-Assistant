# state_manager.py
from abc import ABC, abstractmethod
import json
import sqlite3
import redis # pip install redis
# import psycopg2 # pip install psycopg2-binary
# from psycopg2 import Error

class ConversationStateStore(ABC):
    """Abstract Base Class for conversation state storage."""
    @abstractmethod
    def get_state(self, user_id: str) -> dict:
        """Retrieves the conversation state for a given user ID."""
        pass

    @abstractmethod
    def save_state(self, state: dict):
        """Saves the conversation state for a given user ID."""
        pass

class SQLiteConversationStateStore(ConversationStateStore):
    """Concrete implementation using SQLite for conversation state storage."""
    def __init__(self, db_path="conversation_state.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Create table if it doesn't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_state (
            user_id TEXT PRIMARY KEY,
            current_intent TEXT,
            destination TEXT,
            context_flags TEXT,
            proposed_action TEXT,
            history TEXT
        )
        ''')
        
        # Check if new columns exist, if not add them
        self.cursor.execute("PRAGMA table_info(conversation_state)")
        columns = [column[1] for column in self.cursor.fetchall()]
        if 'proposed_action' not in columns:
            self.cursor.execute('ALTER TABLE conversation_state ADD COLUMN proposed_action TEXT')
            print("Added proposed_action column to existing table")
        if 'history' not in columns:
            self.cursor.execute('ALTER TABLE conversation_state ADD COLUMN history TEXT')
            print("Added history column to existing table")
        
        self.conn.commit()

    def get_state(self, user_id: str) -> dict:
        self.cursor.execute("SELECT current_intent, destination, context_flags, proposed_action, history FROM conversation_state WHERE user_id=?", (user_id,))
        row = self.cursor.fetchone()
        state = {
            "user_id": user_id,
            "current_intent": row[0] if row else None,
            "destination": row[1] if row else None,
            "context_flags": json.loads(row[2]) if row and row[2] else {},
            "proposed_action": json.loads(row[3]) if row and row[3] else None,
            "history": json.loads(row[4]) if row and row[4] else []
        }
        return state

    def save_state(self, state: dict):
        # Only keep the last 10 messages in history
        history = state.get("history", [])
        if len(history) > 10:
            history = history[-10:]
        self.cursor.execute('''
        INSERT INTO conversation_state (user_id, current_intent, destination, context_flags, proposed_action, history)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            current_intent=excluded.current_intent,
            destination=excluded.destination,
            context_flags=excluded.context_flags,
            proposed_action=excluded.proposed_action,
            history=excluded.history
        ''', (
            state["user_id"],
            state["current_intent"],
            state["destination"],
            json.dumps(state["context_flags"]),
            json.dumps(state.get("proposed_action")) if state.get("proposed_action") else None,
            json.dumps(history)
        ))
        self.conn.commit()

class RedisConversationStateStore(ConversationStateStore):
    """Concrete implementation using Redis for conversation state storage."""
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.r.ping() # Test connection
            print("Connected to Redis successfully.")
        except redis.exceptions.ConnectionError as e:
            print(f"Could not connect to Redis: {e}. Please ensure Redis server is running.")
            # Fallback or raise error, depending on desired behavior
            raise e

    def get_state(self, user_id: str) -> dict:
        state_json = self.r.get(f"user_state:{user_id}")
        if state_json:
            return json.loads(state_json)
        # Return default empty state if not found
        return {
            "user_id": user_id,
            "current_intent": None,
            "destination": None,
            "context_flags": {}
        }

    def save_state(self, state: dict):
        # Redis stores strings, so convert dict to JSON string
        self.r.set(f"user_state:{state['user_id']}", json.dumps(state))

# Uncomment and configure if you plan to use PostgreSQL
# class PostgreSQLConversationStateStore(ConversationStateStore):
#     """Concrete implementation using PostgreSQL for conversation state storage."""
#     def __init__(self, dbname, user, password, host='localhost', port=5432):
#         self.conn_params = f"dbname={dbname} user={user} password={password} host={host} port={port}"
#         self._create_table_if_not_exists()

#     def _get_connection(self):
#         return psycopg2.connect(self.conn_params)

#     def _create_table_if_not_exists(self):
#         conn = None
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute('''
#             CREATE TABLE IF NOT EXISTS conversation_state (
#                 user_id TEXT PRIMARY KEY,
#                 current_intent TEXT,
#                 destination TEXT,
#                 context_flags JSONB
#             )
#             ''')
#             conn.commit()
#             cursor.close()
#             print("PostgreSQL table 'conversation_state' ensured.")
#         except Error as e:
#             print(f"Error connecting to or creating PostgreSQL table: {e}")
#         finally:
#             if conn:
#                 conn.close()

#     def get_state(self, user_id: str) -> dict:
#         conn = None
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute("SELECT current_intent, destination, context_flags FROM conversation_state WHERE user_id=%s", (user_id,))
#             row = cursor.fetchone()
#             cursor.close()
#             if row:
#                 return {
#                     "user_id": user_id,
#                     "current_intent": row[0],
#                     "destination": row[1],
#                     "context_flags": row[2] if row[2] else {} # JSONB is auto-parsed by psycopg2
#                 }
#             return {
#                 "user_id": user_id,
#                 "current_intent": None,
#                 "destination": None,
#                 "context_flags": {}
#             }
#         except Error as e:
#             print(f"Error fetching state from PostgreSQL: {e}")
#             return {
#                 "user_id": user_id,
#                 "current_intent": None,
#                 "destination": None,
#                 "context_flags": {}
#             }
#         finally:
#             if conn:
#                 conn.close()

#     def save_state(self, state: dict):
#         conn = None
#         try:
#             conn = self._get_connection()
#             cursor = conn.cursor()
#             cursor.execute('''
#             INSERT INTO conversation_state (user_id, current_intent, destination, context_flags)
#             VALUES (%s, %s, %s, %s::jsonb)
#             ON CONFLICT(user_id) DO UPDATE SET
#                 current_intent=excluded.current_intent,
#                 destination=excluded.destination,
#                 context_flags=excluded.context_flags
#             ''', (
#                 state["user_id"],
#                 state["current_intent"],
#                 state["destination"],
#                 json.dumps(state["context_flags"]) # psycopg2 expects JSON as string for JSONB type
#             ))
#             conn.commit()
#             cursor.close()
#         except Error as e:
#             print(f"Error saving state to PostgreSQL: {e}")
#         finally:
#             if conn:
#                 conn.close()