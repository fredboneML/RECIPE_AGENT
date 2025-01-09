# backend/ai-analyzer/ai_analyzer/agents/database_inspector.py

from sqlalchemy import create_engine, inspect, MetaData
from typing import List
from .base import DatabaseContext


class DatabaseInspectorAgent:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.metadata = MetaData()

    def inspect_database(self, restricted_tables: List[str] = None) -> DatabaseContext:
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

        return DatabaseContext(
            allowed_tables=allowed_tables,
            restricted_tables=restricted_tables,
            table_schemas=table_schemas
        )
