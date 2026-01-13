# ai_analyzer/auth/azure_ad.py
"""
Azure AD token validation for SSO integration.
Validates ID tokens from Microsoft Entra ID and extracts user claims.
"""

import httpx
from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError
from cachetools import TTLCache
from typing import Optional, Dict, List
import logging
import os

logger = logging.getLogger(__name__)

# Cache JWKS keys for 24 hours
_jwks_cache = TTLCache(maxsize=10, ttl=86400)


class AzureADTokenValidator:
    """Validates Azure AD ID tokens"""

    def __init__(self):
        self.tenant_id = os.getenv('AZURE_AD_TENANT_ID')
        self.client_id = os.getenv('AZURE_AD_CLIENT_ID')

        if self.tenant_id:
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
            self.issuer = f"https://login.microsoftonline.com/{self.tenant_id}/v2.0"
            self.jwks_uri = f"https://login.microsoftonline.com/{self.tenant_id}/discovery/v2.0/keys"
        else:
            self.authority = None
            self.issuer = None
            self.jwks_uri = None

    def is_configured(self) -> bool:
        """Check if Azure AD is properly configured"""
        return bool(self.tenant_id and self.client_id)

    async def get_jwks(self) -> Dict:
        """Fetch JWKS from Azure AD with caching"""
        if not self.jwks_uri:
            raise ValueError("Azure AD not configured - missing AZURE_AD_TENANT_ID")

        if 'keys' in _jwks_cache:
            return _jwks_cache['keys']

        async with httpx.AsyncClient() as client:
            response = await client.get(self.jwks_uri, timeout=10.0)
            response.raise_for_status()
            jwks = response.json()
            _jwks_cache['keys'] = jwks
            logger.info("Fetched and cached JWKS from Azure AD")
            return jwks

    async def validate_token(self, token: str) -> Optional[Dict]:
        """
        Validate Azure AD ID token and return claims.

        Args:
            token: The ID token from Azure AD

        Returns:
            Dict with claims if valid, None if invalid
        """
        if not self.is_configured():
            logger.error("Azure AD not configured")
            return None

        try:
            # Get JWKS
            jwks = await self.get_jwks()

            # Get unverified header to find key
            unverified_header = jwt.get_unverified_header(token)

            # Find matching key
            rsa_key = None
            for key in jwks.get('keys', []):
                if key.get('kid') == unverified_header.get('kid'):
                    rsa_key = {
                        'kty': key['kty'],
                        'kid': key['kid'],
                        'use': key.get('use', 'sig'),
                        'n': key['n'],
                        'e': key['e']
                    }
                    break

            if not rsa_key:
                logger.error(f"No matching key found in JWKS for kid: {unverified_header.get('kid')}")
                return None

            # Verify and decode token
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=['RS256'],
                audience=self.client_id,
                issuer=self.issuer,
                options={
                    'verify_exp': True,
                    'verify_iat': True,
                    'verify_aud': True,
                    'verify_iss': True
                }
            )

            logger.info(f"Token validated for user: {payload.get('preferred_username', payload.get('email', 'unknown'))}")
            return payload

        except ExpiredSignatureError:
            logger.warning("Azure AD token has expired")
            return None
        except JWTError as e:
            logger.error(f"JWT validation error: {e}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching JWKS: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error validating token: {e}")
            return None

    def extract_groups(self, payload: Dict) -> List[str]:
        """Extract group IDs from token claims"""
        return payload.get('groups', [])

    def extract_user_info(self, payload: Dict) -> Dict:
        """Extract user information from token claims"""
        return {
            'oid': payload.get('oid'),  # Azure AD Object ID
            'email': payload.get('email') or payload.get('preferred_username'),
            'name': payload.get('name'),
            'preferred_username': payload.get('preferred_username'),
            'upn': payload.get('upn'),
            'groups': self.extract_groups(payload),
            'tid': payload.get('tid'),  # Tenant ID
        }


# Singleton instance
azure_ad_validator = AzureADTokenValidator()
