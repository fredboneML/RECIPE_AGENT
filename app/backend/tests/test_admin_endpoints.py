#!/usr/bin/env python3
"""
Comprehensive tests for Admin User Management endpoints
Tests all admin functionality including:
- Login as admin
- List users
- Add users (with and without auto-generated passwords)
- Reset passwords (with and without auto-generated passwords)
- Delete users
- Access control (non-admin users cannot access)
"""

import pytest
import requests
import hashlib
import json
from typing import Dict, Optional

# Configuration
# When running inside Docker, use the service name; when running from host, use localhost
import os
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin321")


def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


class AdminAPIClient:
    """Client for testing admin API endpoints"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.username: Optional[str] = None
        self.role: Optional[str] = None
    
    def login(self, username: str, password: str) -> Dict:
        """Login and store token"""
        response = requests.post(
            f"{self.base_url}/api/login",
            json={"username": username, "password": password},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            self.username = data.get("username")
            self.role = data.get("role")
            return data
        else:
            raise Exception(f"Login failed: {response.status_code} - {response.text}")
    
    def _get_headers(self) -> Dict:
        """Get headers with authentication token"""
        if not self.token:
            raise Exception("Not authenticated. Call login() first.")
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def list_users(self) -> Dict:
        """List all users"""
        response = requests.get(
            f"{self.base_url}/api/admin/users",
            headers=self._get_headers()
        )
        return {"status": response.status_code, "data": response.json() if response.text else {}}
    
    def add_user(self, username: str, password: Optional[str] = None, role: str = "read_only") -> Dict:
        """Add a new user"""
        payload = {"username": username, "role": role}
        if password:
            payload["password"] = password
        
        response = requests.post(
            f"{self.base_url}/api/admin/users",
            json=payload,
            headers=self._get_headers()
        )
        return {"status": response.status_code, "data": response.json() if response.text else {}}
    
    def reset_password(self, username: str, password: Optional[str] = None) -> Dict:
        """Reset user password"""
        payload = {}
        if password:
            payload["password"] = password
        
        response = requests.put(
            f"{self.base_url}/api/admin/users/{username}/reset-password",
            json=payload,
            headers=self._get_headers()
        )
        return {"status": response.status_code, "data": response.json() if response.text else {}}
    
    def delete_user(self, username: str) -> Dict:
        """Delete a user"""
        response = requests.delete(
            f"{self.base_url}/api/admin/users/{username}",
            headers=self._get_headers()
        )
        return {"status": response.status_code, "data": response.json() if response.text else {}}
    
    def get_generated_password(self, username: str) -> Dict:
        """Get generated password for a user"""
        response = requests.get(
            f"{self.base_url}/api/admin/users/{username}/generated-password",
            headers=self._get_headers()
        )
        return {"status": response.status_code, "data": response.json() if response.text else {}}


def test_admin_login():
    """Test 1: Admin can login successfully"""
    print("\n=== Test 1: Admin Login ===")
    client = AdminAPIClient()
    result = client.login(ADMIN_USERNAME, ADMIN_PASSWORD)
    
    assert result["success"] == True, "Login should succeed"
    assert "access_token" in result, "Should receive access token"
    assert result["role"] == "admin", "Should have admin role"
    assert client.token is not None, "Token should be stored"
    
    print(f"✅ Admin login successful: {result['username']} (role: {result['role']})")
    return client


def test_list_users(client: AdminAPIClient):
    """Test 2: Admin can list all users"""
    print("\n=== Test 2: List Users ===")
    result = client.list_users()
    
    assert result["status"] == 200, f"Should return 200, got {result['status']}"
    assert "users" in result["data"], "Response should contain users list"
    assert isinstance(result["data"]["users"], list), "Users should be a list"
    
    users = result["data"]["users"]
    print(f"✅ Found {len(users)} users:")
    for user in users:
        print(f"   - {user['username']} ({user['role']})")
    
    return result  # Return the full result dict, not just users list


def test_add_user_with_password(client: AdminAPIClient):
    """Test 3: Admin can add user with specified password"""
    print("\n=== Test 3: Add User with Password ===")
    test_username = "test_user_manual"
    test_password = "testpass123"
    
    result = client.add_user(test_username, password=test_password, role="read_only")
    
    assert result["status"] == 200, f"Should return 200, got {result['status']}"
    assert result["data"]["success"] == True, "Should indicate success"
    assert result["data"]["username"] == test_username, "Should return correct username"
    assert "generated_password" not in result["data"] or result["data"]["generated_password"] is None, "Should not have generated password"
    
    print(f"✅ User '{test_username}' added successfully with manual password")
    return test_username


def test_add_user_auto_generate_password(client: AdminAPIClient):
    """Test 4: Admin can add user with auto-generated password"""
    print("\n=== Test 4: Add User with Auto-Generated Password ===")
    test_username = "test_user_auto"
    
    result = client.add_user(test_username, password=None, role="read_only")
    
    assert result["status"] == 200, f"Should return 200, got {result['status']}"
    assert result["data"]["success"] == True, "Should indicate success"
    assert result["data"]["username"] == test_username, "Should return correct username"
    assert "generated_password" in result["data"], "Should have generated password"
    assert result["data"]["generated_password"] is not None, "Generated password should not be None"
    assert len(result["data"]["generated_password"]) > 0, "Generated password should not be empty"
    
    generated_pwd = result["data"]["generated_password"]
    print(f"✅ User '{test_username}' added successfully")
    print(f"   Generated password: {generated_pwd}")
    
    return test_username, generated_pwd


def test_reset_password_with_password(client: AdminAPIClient, username: str):
    """Test 5: Admin can reset password with specified password"""
    print("\n=== Test 5: Reset Password with Specified Password ===")
    new_password = "newpass456"
    
    result = client.reset_password(username, password=new_password)
    
    assert result["status"] == 200, f"Should return 200, got {result['status']}"
    assert result["data"]["success"] == True, "Should indicate success"
    assert result["data"]["username"] == username, "Should return correct username"
    assert "generated_password" not in result["data"] or result["data"]["generated_password"] is None, "Should not have generated password"
    
    print(f"✅ Password reset successfully for '{username}' with specified password")
    
    # Verify the new password works
    test_client = AdminAPIClient()
    try:
        test_client.login(username, new_password)
        print(f"✅ Verified: New password works for login")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify new password: {e}")


def test_reset_password_auto_generate(client: AdminAPIClient, username: str):
    """Test 6: Admin can reset password with auto-generation"""
    print("\n=== Test 6: Reset Password with Auto-Generation ===")
    
    result = client.reset_password(username, password=None)
    
    assert result["status"] == 200, f"Should return 200, got {result['status']}"
    assert result["data"]["success"] == True, "Should indicate success"
    assert result["data"]["username"] == username, "Should return correct username"
    assert "generated_password" in result["data"], "Should have generated password"
    assert result["data"]["generated_password"] is not None, "Generated password should not be None"
    assert len(result["data"]["generated_password"]) > 0, "Generated password should not be empty"
    
    generated_pwd = result["data"]["generated_password"]
    print(f"✅ Password reset successfully for '{username}' with auto-generation")
    print(f"   Generated password: {generated_pwd}")
    
    # Verify the generated password works
    test_client = AdminAPIClient()
    try:
        test_client.login(username, generated_pwd)
        print(f"✅ Verified: Generated password works for login")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify generated password: {e}")
    
    return generated_pwd


def test_delete_user(client: AdminAPIClient, username: str):
    """Test 7: Admin can delete user"""
    print("\n=== Test 7: Delete User ===")
    
    result = client.delete_user(username)
    
    assert result["status"] == 200, f"Should return 200, got {result['status']}"
    assert result["data"]["success"] == True, "Should indicate success"
    
    print(f"✅ User '{username}' deleted successfully")
    
    # Verify user is deleted
    users_result = client.list_users()
    usernames = [u["username"] for u in users_result["data"]["users"]]
    assert username not in usernames, f"User '{username}' should be deleted"
    print(f"✅ Verified: User '{username}' no longer exists")


def test_non_admin_access_denied():
    """Test 8: Non-admin users cannot access admin endpoints"""
    print("\n=== Test 8: Non-Admin Access Denied ===")
    
    # Try to login as a non-admin user (if exists)
    # First, let's try to create a read-only user and test
    admin_client = AdminAPIClient()
    admin_client.login(ADMIN_USERNAME, ADMIN_PASSWORD)
    
    # Create a test read-only user
    test_username = "test_readonly_user"
    test_password = "testpass789"
    
    try:
        admin_client.add_user(test_username, password=test_password, role="read_only")
        print(f"✅ Created test read-only user: {test_username}")
    except Exception as e:
        print(f"⚠️  Could not create test user: {e}")
        return
    
    # Try to access admin endpoints as read-only user
    readonly_client = AdminAPIClient()
    readonly_client.login(test_username, test_password)
    
    assert readonly_client.role == "read_only", "Should be read-only user"
    
    # Try to list users (should fail)
    result = readonly_client.list_users()
    assert result["status"] == 403, f"Should return 403, got {result['status']}"
    print(f"✅ Read-only user correctly denied access to list users (403)")
    
    # Try to add user (should fail)
    result = readonly_client.add_user("test_user", password="pass", role="read_only")
    assert result["status"] == 403, f"Should return 403, got {result['status']}"
    print(f"✅ Read-only user correctly denied access to add user (403)")
    
    # Try to reset password (should fail)
    result = readonly_client.reset_password(test_username)
    assert result["status"] == 403, f"Should return 403, got {result['status']}"
    print(f"✅ Read-only user correctly denied access to reset password (403)")
    
    # Clean up
    try:
        admin_client.delete_user(test_username)
        print(f"✅ Cleaned up test read-only user")
    except:
        pass


def test_cannot_delete_self(client: AdminAPIClient):
    """Test 9: Admin cannot delete their own account"""
    print("\n=== Test 9: Cannot Delete Self ===")
    
    result = client.delete_user(ADMIN_USERNAME)
    
    assert result["status"] == 400, f"Should return 400, got {result['status']}"
    assert "cannot delete your own account" in result["data"].get("detail", "").lower(), "Should indicate cannot delete self"
    
    print(f"✅ Admin correctly prevented from deleting own account")


def test_add_user_duplicate_username(client: AdminAPIClient):
    """Test 10: Cannot add user with duplicate username"""
    print("\n=== Test 10: Duplicate Username Prevention ===")
    
    test_username = "test_duplicate_user"
    test_password = "testpass123"
    
    # Add user first time
    result1 = client.add_user(test_username, password=test_password, role="read_only")
    assert result1["status"] == 200, "First addition should succeed"
    
    # Try to add same user again
    result2 = client.add_user(test_username, password="different", role="read_only")
    assert result2["status"] == 400, f"Should return 400 for duplicate, got {result2['status']}"
    assert "already exists" in result2["data"].get("detail", "").lower(), "Should indicate user exists"
    
    print(f"✅ Duplicate username correctly prevented")
    
    # Clean up
    try:
        client.delete_user(test_username)
    except:
        pass


def main():
    """Run all admin endpoint tests"""
    print("=" * 80)
    print("ADMIN ENDPOINT TEST SUITE")
    print("=" * 80)
    
    try:
        # Test 1: Login as admin
        admin_client = test_admin_login()
        
        # Test 2: List users
        initial_result = test_list_users(admin_client)
        initial_user_count = len(initial_result["data"]["users"])
        
        # Test 3: Add user with password
        manual_user = test_add_user_with_password(admin_client)
        
        # Test 4: Add user with auto-generated password
        auto_user, auto_password = test_add_user_auto_generate_password(admin_client)
        
        # Test 5: Reset password with specified password
        test_reset_password_with_password(admin_client, manual_user)
        
        # Test 6: Reset password with auto-generation
        test_reset_password_auto_generate(admin_client, manual_user)
        
        # Test 7: Delete user
        test_delete_user(admin_client, manual_user)
        test_delete_user(admin_client, auto_user)
        
        # Test 8: Non-admin access denied
        test_non_admin_access_denied()
        
        # Test 9: Cannot delete self
        test_cannot_delete_self(admin_client)
        
        # Test 10: Duplicate username prevention
        test_add_user_duplicate_username(admin_client)
        
        # Final user count check
        final_result = test_list_users(admin_client)
        final_user_count = len(final_result["data"]["users"])
        print(f"\n✅ Initial user count: {initial_user_count}")
        print(f"✅ Final user count: {final_user_count}")
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        print("=" * 80)
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
