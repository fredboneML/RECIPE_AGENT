# ai_analyzer/auth/models.py
"""
Database models for Azure AD SSO integration.
"""

from sqlalchemy import Column, String, DateTime, Boolean, Integer, ForeignKey
from datetime import datetime, timezone

# Import Base from the main models module
from ai_analyzer.data_import_postgresql import Base


class AzureADGroupMapping(Base):
    """Maps Azure AD group IDs to application roles"""
    __tablename__ = 'azure_ad_group_mappings'

    group_id = Column(String(255), primary_key=True)  # Azure AD Group Object ID
    group_name = Column(String(255), nullable=False)  # Human-readable name
    app_role = Column(String(50), nullable=False)     # admin, write, read_only
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<AzureADGroupMapping(group={self.group_name}, role={self.app_role})>"


class AzureADUser(Base):
    """Links Azure AD users to local user records"""
    __tablename__ = 'azure_ad_users'

    azure_oid = Column(String(255), primary_key=True)  # Azure AD Object ID
    email = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255), nullable=True)
    local_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    first_login = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    def __repr__(self):
        return f"<AzureADUser(email={self.email}, active={self.is_active})>"


# SQL for creating tables manually (for migration)
CREATE_TABLES_SQL = """
-- Group-to-role mapping
CREATE TABLE IF NOT EXISTS azure_ad_group_mappings (
    group_id VARCHAR(255) PRIMARY KEY,
    group_name VARCHAR(255) NOT NULL,
    app_role VARCHAR(50) NOT NULL CHECK (app_role IN ('admin', 'write', 'read_only')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Azure AD user tracking
CREATE TABLE IF NOT EXISTS azure_ad_users (
    azure_oid VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255),
    local_user_id INTEGER REFERENCES users(id),
    first_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_azure_users_email ON azure_ad_users(email);
CREATE INDEX IF NOT EXISTS idx_azure_users_local_id ON azure_ad_users(local_user_id);
CREATE INDEX IF NOT EXISTS idx_azure_group_mappings_active ON azure_ad_group_mappings(is_active);
"""
