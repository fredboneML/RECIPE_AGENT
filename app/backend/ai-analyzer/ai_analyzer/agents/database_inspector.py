# In database_inspector.py

from sqlalchemy import create_engine, inspect, MetaData, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from .base import DatabaseContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """Context information for conversations"""
    topics: Dict[str, int] = None
    current_focus: Optional[str] = None
    sentiment_context: Optional[str] = None
    time_range: Optional[str] = None
    last_query_type: Optional[str] = None

    def __post_init__(self):
        if self.topics is None:
            self.topics = {}

    def update_from_response(self, response: str):
        """Update context based on a response"""
        try:
            # Extract topics mentioned in the response
            if "topic" in response.lower():
                topics = [line.split(":")[1].strip()
                          for line in response.split("\n")
                          if "topic" in line.lower() and ":" in line]
                for topic in topics:
                    self.topics[topic] = self.topics.get(topic, 0) + 1
                    if not self.current_focus:
                        self.current_focus = topic

            # Extract sentiment context
            if "sentiment" in response.lower():
                self.sentiment_context = next(
                    (line for line in response.split("\n")
                     if "sentiment" in line.lower()),
                    self.sentiment_context
                )

            # Extract time range context
            time_indicators = ["day", "week", "month", "year"]
            for indicator in time_indicators:
                if indicator in response.lower():
                    self.time_range = next(
                        (line for line in response.split("\n")
                         if indicator in line.lower()),
                        self.time_range
                    )
                    break

        except Exception as e:
            logger.error(f"Error updating context: {e}")


class DatabaseInspectorAgent:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.metadata = MetaData()
        self.conversation_contexts = {}

    def inspect_database(self, restricted_tables: List[str] = None) -> DatabaseContext:
        """Inspect database structure and return proper DatabaseContext instance"""
        inspector = inspect(self.engine)
        restricted_tables = restricted_tables or []
        allowed_tables = []
        table_schemas = {}

        for table_name in inspector.get_table_names():
            if table_name not in restricted_tables:
                allowed_tables.append(table_name)
                columns = inspector.get_columns(table_name)
                table_schemas[table_name] = {
                    col['name']: str(col['type'])
                    for col in columns
                }

        # Return a proper DatabaseContext instance
        return DatabaseContext(
            allowed_tables=allowed_tables,
            restricted_tables=restricted_tables,
            table_schemas=table_schemas
        )

    def get_conversation_context(self, db_session, conversation_id: str) -> Tuple[Optional[str], Optional[ConversationContext]]:
        """Get context from previous conversation messages"""
        try:
            if not conversation_id:
                return None, None

            # Get or create conversation context
            if conversation_id not in self.conversation_contexts:
                self.conversation_contexts[conversation_id] = ConversationContext(
                )

            context = self.conversation_contexts[conversation_id]

            # Query previous messages
            query = text("""
                SELECT query, response, message_order
                FROM user_memory
                WHERE conversation_id = :conversation_id
                AND is_active = true 
                AND expires_at > NOW()
                ORDER BY message_order DESC
                LIMIT 10  -- Get last 10 messages for context
            """)

            result = db_session.execute(
                query, {"conversation_id": conversation_id}
            ).fetchall()

            if not result:
                return None, context

            # Build context string and update context object
            context_parts = []
            for row in reversed(result):  # Process in chronological order
                context_parts.extend([
                    f"Previous Question: {row.query}",
                    f"Previous Answer: {row.response}\n"
                ])
                context.update_from_response(row.response)

            context_text = "\n".join(context_parts)

            return context_text, context

        except SQLAlchemyError as e:
            logger.error(f"Error retrieving conversation context: {e}")
            return None, None

    def enhance_question_with_context(self, question: str, context: ConversationContext) -> str:
        """Enhance the question using conversation context"""
        if not context:
            return question

        enhanced_question = question

        # Replace references to "that topic" with actual topic
        if context.current_focus and any(phrase in question.lower()
                                         for phrase in ["that topic", "this topic", "the topic"]):
            enhanced_question = enhanced_question.replace(
                "that topic", f'"{context.current_focus}"'
            ).replace(
                "this topic", f'"{context.current_focus}"'
            ).replace(
                "the topic", f'"{context.current_focus}"'
            )

        # Add time context if missing and previously established
        if context.time_range and not any(indicator in question.lower()
                                          for indicator in ["day", "week", "month", "year"]):
            enhanced_question = f"{enhanced_question} {context.time_range}"

        # Add sentiment context if relevant
        if context.sentiment_context and "sentiment" in question.lower():
            enhanced_question = f"{enhanced_question} (considering {
                context.sentiment_context})"

        return enhanced_question

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific table"""
        inspector = inspect(self.engine)
        if table_name in inspector.get_table_names():
            return {
                "columns": inspector.get_columns(table_name),
                "indexes": inspector.get_indexes(table_name),
                "primary_key": inspector.get_primary_keys(table_name)
            }
        return {}
