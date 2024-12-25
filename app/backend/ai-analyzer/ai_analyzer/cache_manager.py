import hashlib
from datetime import datetime
from sqlalchemy import Table, Column, String, DateTime, MetaData, create_engine, select, func, Integer, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from typing import Optional, Tuple


class DatabaseCacheManager:
    def __init__(self, engine):
        self.engine = engine
        self._init_tables()

    def _init_tables(self):
        metadata = MetaData()

        # Existing cache table
        self.cache_table = Table(
            'query_cache',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('query_hash', String, unique=True, nullable=False),
            Column('query_text', String, nullable=False),
            Column('response', String, nullable=False),
            Column('last_record_count', Integer, nullable=False),
            Column('timestamp', DateTime, default=datetime.utcnow)
        )

        # New performance tracking table
        self.performance_table = Table(
            'query_performance',
            metadata,
            Column('id', Integer, primary_key=True),
            Column('query_text', String, nullable=False),
            Column('was_answered', Boolean, nullable=False),
            Column('error_message', String, nullable=True),
            Column('response_time', Integer, nullable=True),  # in milliseconds
            Column('timestamp', DateTime, default=datetime.utcnow),
            # for categorizing types of questions
            Column('topic_category', String, nullable=True),
            # to track token usage if available
            Column('tokens_used', Integer, nullable=True)
        )

        metadata.create_all(self.engine)

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def _get_record_count(self) -> int:
        """Get total count of records in the transcription table"""
        Session = sessionmaker(bind=self.engine)
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

        Session = sessionmaker(bind=self.engine)
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

        Session = sessionmaker(bind=self.engine)
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
        Session = sessionmaker(bind=self.engine)
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
        Session = sessionmaker(bind=self.engine)
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
