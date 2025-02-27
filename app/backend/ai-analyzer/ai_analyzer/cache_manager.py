import hashlib
import logging
import json
from sqlalchemy import Table, Column, String, DateTime, MetaData, create_engine, select, func, Integer, Boolean, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from typing import Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DatabaseCacheManager:
    """Manages caching of queries and results in the database"""

    def __init__(self, db_session):
        """Initialize with database session"""
        self.db_session = db_session
        logger.info("Database cache manager initialized")

    def get_query_result(self, question, tenant_code, conversation_id=None):
        """Get cached result for a question if available"""
        try:
            # Create a hash key for the question
            hash_key = self._create_hash_key(question)

            # Query the cache table
            stmt = text("""
                SELECT sql, result, created_at
                FROM query_cache
                WHERE hash_key = :hash_key
                AND tenant_code = :tenant_code
                AND expires_at > CURRENT_TIMESTAMP
                LIMIT 1
            """)

            result = self.db_session.execute(
                stmt,
                {"hash_key": hash_key, "tenant_code": tenant_code}
            ).fetchone()

            if result:
                # If we have a conversation ID, verify this is the right context
                if conversation_id:
                    # Check if this is a new conversation (no history yet)
                    stmt_check = text("""
                        SELECT COUNT(*) FROM user_memory
                        WHERE conversation_id = :conversation_id
                    """)

                    conversation_count = self.db_session.execute(
                        stmt_check,
                        {"conversation_id": conversation_id}
                    ).scalar()

                    # If this is a new conversation or the first question, we can use the cache
                    if conversation_count == 0:
                        logger.info(
                            f"Cache hit for new conversation: {question[:50]}...")
                        return {
                            "sql": result[0],
                            "result": result[1],
                            "timestamp": result[2].isoformat() if result[2] else None
                        }

                    # For existing conversations, check if this exact question exists in this conversation
                    stmt_check = text("""
                        SELECT 1 FROM user_memory
                        WHERE conversation_id = :conversation_id
                        AND query = :question
                        LIMIT 1
                    """)

                    context_match = self.db_session.execute(
                        stmt_check,
                        {"conversation_id": conversation_id, "question": question}
                    ).fetchone()

                    # If this question doesn't match the conversation context, don't use the cache
                    if not context_match:
                        logger.info(
                            f"Cache hit ignored - different conversation context for: {question[:50]}...")
                        return None

                logger.info(f"Cache hit for question: {question[:50]}...")
                return {
                    "sql": result[0],
                    "result": result[1],
                    "timestamp": result[2].isoformat() if result[2] else None
                }

            logger.info(f"Cache miss for question: {question[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def store_query_result(self, question, sql, result, tenant_code, expiry_days=7):
        """Store query result in cache"""
        try:
            # Create a hash key for the question
            hash_key = self._create_hash_key(question)

            # Calculate expiry date
            expires_at = datetime.utcnow() + timedelta(days=expiry_days)

            # Count the number of results (approximate by counting newlines)
            result_count = result.count('\n') + 1 if result else 0

            # Insert or update cache entry
            stmt = text("""
                INSERT INTO query_cache 
                (question, sql, result, tenant_code, created_at, expires_at, hash_key, result_count)
                VALUES (:question, :sql, :result, :tenant_code, CURRENT_TIMESTAMP, :expires_at, :hash_key, :result_count)
                ON CONFLICT (hash_key, tenant_code) 
                DO UPDATE SET 
                    result = EXCLUDED.result,
                    sql = EXCLUDED.sql,
                    created_at = CURRENT_TIMESTAMP,
                    expires_at = EXCLUDED.expires_at,
                    result_count = EXCLUDED.result_count
            """)

            self.db_session.execute(stmt, {
                "question": question,
                "sql": sql,
                "result": result,
                "tenant_code": tenant_code,
                "expires_at": expires_at,
                "hash_key": hash_key,
                "result_count": result_count
            })

            self.db_session.commit()
            logger.info(f"Cached result for question: {question[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")
            self.db_session.rollback()
            return False

    def _create_hash_key(self, question):
        """Create a hash key for the question"""
        # Normalize the question by removing extra whitespace and converting to lowercase
        normalized = ' '.join(question.lower().split())
        # Create SHA-256 hash
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def clear_expired_cache(self):
        """Clear expired cache entries"""
        try:
            stmt = text(
                "DELETE FROM query_cache WHERE expires_at < CURRENT_TIMESTAMP")
            result = self.db_session.execute(stmt)
            self.db_session.commit()
            deleted_count = result.rowcount
            logger.info(f"Cleared {deleted_count} expired cache entries")
            return deleted_count
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
            self.db_session.rollback()
            return 0

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def _get_record_count(self) -> int:
        """Get total count of records in the transcription table"""
        Session = sessionmaker(bind=self.db_session)
        session = Session()
        try:
            result = session.execute(
                select(func.count()).select_from(
                    Table('transcription', MetaData(), schema='public'))
            ).scalar()
            return result or 0
        finally:
            session.close()

    def check_cache(self, query: str) -> Tuple[bool, Optional[str], bool]:
        """
        Check if query exists in cache and if database has new records.
        Returns: (is_cached, cached_response, has_new_records)
        """
        query_hash = self._hash_query(query)
        current_record_count = self._get_record_count()

        Session = sessionmaker(bind=self.db_session)
        session = Session()

        try:
            cached_result = session.execute(
                select(self.cache_table).where(
                    self.cache_table.c.query_hash == query_hash)
            ).first()

            if cached_result:
                has_new_records = current_record_count != cached_result.last_record_count
                return True, cached_result.response, has_new_records

            return False, None, True

        finally:
            session.close()

    def cache_response(self, query: str, response: str):
        """Store query and response in cache using UPSERT"""
        query_hash = self._hash_query(query)
        current_record_count = self._get_record_count()
        current_time = datetime.utcnow()

        Session = sessionmaker(bind=self.db_session)
        session = Session()

        try:
            # Create an insert statement
            insert_stmt = insert(self.cache_table).values(
                query_hash=query_hash,
                query_text=query,
                response=response,
                last_record_count=current_record_count,
                timestamp=current_time
            )

            # Add the ON CONFLICT DO UPDATE clause
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=['query_hash'],
                set_={
                    'query_text': insert_stmt.excluded.query_text,
                    'response': insert_stmt.excluded.response,
                    'last_record_count': insert_stmt.excluded.last_record_count,
                    'timestamp': insert_stmt.excluded.timestamp
                }
            )

            # Execute the upsert statement
            session.execute(upsert_stmt)
            session.commit()
        finally:
            session.close()

    def track_query_performance(self, query: str, was_answered: bool, response_time: Optional[int] = None,
                                error_message: Optional[str] = None, topic_category: Optional[str] = None,
                                tokens_used: Optional[int] = None):
        """
        Track the performance of a query, including whether it was successfully answered.

        Args:
            query (str): The user's question
            was_answered (bool): Whether the question was successfully answered
            response_time (int, optional): Response time in milliseconds
            error_message (str, optional): Error message if the query failed
            topic_category (str, optional): Category of the question (e.g., 'sentiment', 'topic_analysis')
            tokens_used (int, optional): Number of tokens used in processing the query
        """
        Session = sessionmaker(bind=self.db_session)
        session = Session()

        try:
            # Create an insert statement for performance tracking
            insert_stmt = insert(self.performance_table).values(
                query_text=query,
                was_answered=was_answered,
                response_time=response_time,
                error_message=error_message,
                topic_category=topic_category,
                tokens_used=tokens_used,
                timestamp=datetime.utcnow()
            )

            # Execute the insert statement
            session.execute(insert_stmt)
            session.commit()

        finally:
            session.close()

    def get_performance_metrics(self) -> dict:
        """
        Get performance metrics for queries.
        Returns a dictionary with various performance statistics.
        """
        Session = sessionmaker(bind=self.db_session)
        session = Session()

        try:
            # Get total queries
            total_queries = session.execute(
                select(func.count()).select_from(self.performance_table)
            ).scalar()

            # Get successful queries
            successful_queries = session.execute(
                select(func.count()).select_from(self.performance_table).where(
                    self.performance_table.c.was_answered == True
                )
            ).scalar()

            # Calculate average response time
            avg_response_time = session.execute(
                select(func.avg(self.performance_table.c.response_time)).select_from(
                    self.performance_table
                ).where(self.performance_table.c.response_time.isnot(None))
            ).scalar()

            # Get performance by topic category
            performance_by_topic = session.execute(
                select(
                    self.performance_table.c.topic_category,
                    func.count().label('total'),
                    func.sum(func.cast(self.performance_table.c.was_answered, Integer)).label(
                        'successful')
                ).select_from(self.performance_table)
                .where(self.performance_table.c.topic_category.isnot(None))
                .group_by(self.performance_table.c.topic_category)
            ).fetchall()

            return {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': (successful_queries / total_queries * 100) if total_queries > 0 else 0,
                'average_response_time': avg_response_time,
                'performance_by_topic': {
                    topic: {
                        'total': total,
                        'successful': successful,
                        'success_rate': (successful / total * 100) if total > 0 else 0
                    }
                    for topic, total, successful in performance_by_topic
                }
            }

        finally:
            session.close()
