import hashlib
from datetime import datetime
from sqlalchemy import Table, Column, String, DateTime, MetaData, create_engine, select, func, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from typing import Optional, Tuple

class DatabaseCacheManager:
    def __init__(self, engine):
        self.engine = engine
        self._init_cache_table()
        
    def _init_cache_table(self):
        metadata = MetaData()
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
        metadata.create_all(self.engine)
        
    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()
        
    def _get_record_count(self) -> int:
        """Get total count of records in the transcription table"""
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            result = session.execute(
                select(func.count()).select_from(Table('transcription', MetaData(), schema='public'))
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
                select(self.cache_table).where(self.cache_table.c.query_hash == query_hash)
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