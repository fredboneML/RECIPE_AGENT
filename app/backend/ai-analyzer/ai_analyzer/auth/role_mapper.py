# ai_analyzer/auth/role_mapper.py
"""
Maps Azure AD groups to application roles.
Provides flexible role assignment based on group membership.
"""

from sqlalchemy.orm import Session
from typing import List
import logging

logger = logging.getLogger(__name__)


class RoleMapper:
    """Maps Azure AD groups to application roles"""

    # Role hierarchy (higher = more permissions)
    ROLE_HIERARCHY = {
        'admin': 3,
        'write': 2,
        'read_only': 1
    }

    DEFAULT_ROLE = 'read_only'

    def map_groups_to_role(self, db: Session, group_ids: List[str]) -> str:
        """
        Determine application role based on Azure AD group membership.
        If user is in multiple groups, returns highest-privilege role.

        Args:
            db: Database session
            group_ids: List of Azure AD group IDs from token

        Returns:
            Application role string ('admin', 'write', 'read_only')
        """
        # Import here to avoid circular imports
        from ai_analyzer.auth.models import AzureADGroupMapping

        if not group_ids:
            logger.warning("No groups provided, defaulting to read_only")
            return self.DEFAULT_ROLE

        try:
            # Query mappings for user's groups
            mappings = db.query(AzureADGroupMapping).filter(
                AzureADGroupMapping.group_id.in_(group_ids),
                AzureADGroupMapping.is_active == True
            ).all()

            if not mappings:
                logger.info(f"No group mappings found for groups: {group_ids[:5]}...")  # Log first 5 groups
                return self.DEFAULT_ROLE

            # Find highest privilege role
            highest_role = self.DEFAULT_ROLE
            highest_level = 0

            for mapping in mappings:
                level = self.ROLE_HIERARCHY.get(mapping.app_role, 0)
                if level > highest_level:
                    highest_level = level
                    highest_role = mapping.app_role

            logger.info(f"Mapped {len(group_ids)} groups to role: {highest_role}")
            return highest_role

        except Exception as e:
            logger.error(f"Error mapping groups to role: {e}")
            return self.DEFAULT_ROLE

    def add_group_mapping(
        self,
        db: Session,
        group_id: str,
        group_name: str,
        app_role: str
    ):
        """Add or update a group mapping"""
        from ai_analyzer.auth.models import AzureADGroupMapping

        if app_role not in self.ROLE_HIERARCHY:
            raise ValueError(f"Invalid role: {app_role}. Must be one of: {list(self.ROLE_HIERARCHY.keys())}")

        existing = db.query(AzureADGroupMapping).filter_by(
            group_id=group_id
        ).first()

        if existing:
            existing.group_name = group_name
            existing.app_role = app_role
            existing.is_active = True
            logger.info(f"Updated group mapping: {group_name} -> {app_role}")
        else:
            existing = AzureADGroupMapping(
                group_id=group_id,
                group_name=group_name,
                app_role=app_role
            )
            db.add(existing)
            logger.info(f"Created group mapping: {group_name} -> {app_role}")

        db.commit()
        return existing

    def delete_group_mapping(self, db: Session, group_id: str) -> bool:
        """Soft delete a group mapping"""
        from ai_analyzer.auth.models import AzureADGroupMapping

        mapping = db.query(AzureADGroupMapping).filter_by(group_id=group_id).first()
        if mapping:
            mapping.is_active = False
            db.commit()
            logger.info(f"Deactivated group mapping: {mapping.group_name}")
            return True
        return False

    def get_all_mappings(self, db: Session, include_inactive: bool = False) -> List[dict]:
        """Get all group mappings"""
        from ai_analyzer.auth.models import AzureADGroupMapping

        query = db.query(AzureADGroupMapping)
        if not include_inactive:
            query = query.filter(AzureADGroupMapping.is_active == True)

        return [
            {
                "group_id": m.group_id,
                "group_name": m.group_name,
                "app_role": m.app_role,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in query.all()
        ]


# Singleton instance
role_mapper = RoleMapper()
