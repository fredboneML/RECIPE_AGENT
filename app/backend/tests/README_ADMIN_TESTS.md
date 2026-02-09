# Admin Endpoint Tests

This directory contains comprehensive tests for the Admin User Management functionality.

## Running the Tests

### Option 1: Run inside Docker container
```bash
cd app
docker-compose exec backend_app python3 /usr/src/app/tests/test_admin_endpoints.py
```

### Option 2: Use the test script
```bash
./app/backend/tests/run_admin_tests.sh
```

### Option 3: Run from host (if backend is accessible)
```bash
cd app/backend/tests
python3 test_admin_endpoints.py
```

## Test Coverage

The test suite covers:

1. **Admin Login** - Verify admin can login with credentials
2. **List Users** - Admin can retrieve list of all users
3. **Add User with Password** - Admin can create user with specified password
4. **Add User Auto-Generate** - Admin can create user with auto-generated password
5. **Reset Password with Password** - Admin can reset password with specified value
6. **Reset Password Auto-Generate** - Admin can reset password with auto-generation
7. **Delete User** - Admin can delete users
8. **Access Control** - Non-admin users are denied access
9. **Self-Deletion Prevention** - Admin cannot delete own account
10. **Duplicate Username Prevention** - Cannot create users with duplicate usernames

## Configuration

The test uses environment variables or defaults:
- `BASE_URL` - Backend API URL (default: http://localhost:8000)
- `ADMIN_USERNAME` - Admin username (default: admin)
- `ADMIN_PASSWORD` - Admin password (default: admin321)

You can override these by setting environment variables:
```bash
BASE_URL=http://localhost:8000 ADMIN_USERNAME=admin ADMIN_PASSWORD=admin321 python3 test_admin_endpoints.py
```

## Expected Output

On success, you should see:
```
================================================================================
ADMIN ENDPOINT TEST SUITE
================================================================================

=== Test 1: Admin Login ===
✅ Admin login successful: admin (role: admin)

=== Test 2: List Users ===
✅ Found X users:
   - admin (admin)
   ...

=== Test 3: Add User with Password ===
✅ User 'test_user_manual' added successfully with manual password

... (more tests)

================================================================================
✅ ALL TESTS PASSED!
================================================================================
```
