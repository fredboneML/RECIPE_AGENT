# Azure AD SSO Setup Guide

This guide walks you through setting up Microsoft Entra ID (Azure AD) Single Sign-On for the Recipe Agent application.

## Prerequisites

- Azure subscription with access to Microsoft Entra ID
- Admin access to create app registrations and security groups
- Access to the application's deployment environment

---

## Step 1: Create Azure AD Security Groups

Create security groups to control access levels in the application.

1. Navigate to **Azure Portal** > **Microsoft Entra ID** > **Groups**
2. Click **New group** and create the following groups:

| Group Name | Group Type | Description |
|------------|------------|-------------|
| `Recipe-Agent-Admins` | Security | Full administrative access |
| `Recipe-Agent-Writers` | Security | Can create and edit recipes |
| `Recipe-Agent-Users` | Security | Read-only access |

3. Add users to appropriate groups based on their required access level
4. **Record the Object ID** of each group (you'll need these later)

---

## Step 2: Register the Application in Azure AD

### 2.1 Create App Registration

1. Navigate to **Azure Portal** > **Microsoft Entra ID** > **App registrations**
2. Click **New registration**
3. Fill in the details:
   - **Name**: `Recipe Agent - Production` (or environment-specific name)
   - **Supported account types**: "Accounts in this organizational directory only (Single tenant)"
   - **Redirect URI**:
     - Platform: `Single-page application (SPA)`
     - URI: `https://recipe-agent-agrana.westeurope.cloudapp.azure.com/auth/callback`

4. Click **Register**

### 2.2 Record Important Values

From the app registration **Overview** page, record:
- **Application (client) ID**: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- **Directory (tenant) ID**: `yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy`

### 2.3 Configure Authentication

1. Go to **Authentication** in the left menu
2. Under **Single-page application**, add additional redirect URIs for development:
   - `http://localhost:3000/auth/callback`
3. Under **Implicit grant and hybrid flows**:
   - Check **ID tokens**
   - Leave **Access tokens** unchecked
4. Click **Save**

### 2.4 Configure API Permissions

1. Go to **API permissions** in the left menu
2. Click **Add a permission** > **Microsoft Graph** > **Delegated permissions**
3. Add the following permissions:
   - `openid` (Sign users in)
   - `profile` (View users' basic profile)
   - `email` (View users' email address)
   - `User.Read` (Sign in and read user profile)
   - `GroupMember.Read.All` (Read group memberships)

4. Click **Grant admin consent for [Your Organization]**
5. Verify all permissions show "Granted for [Your Organization]"

### 2.5 Configure Token Claims

1. Go to **Token configuration** in the left menu
2. Click **Add optional claim**
3. Select **ID** token type
4. Add the following claims:
   - `email`
   - `preferred_username`
   - `upn` (User Principal Name)
5. Click **Add**

6. Click **Add groups claim**
7. Select **Security groups**
8. Under **Customize token properties by type**, for ID token select **Group ID**
9. Click **Add**

---

## Step 3: Configure the Application

### 3.1 Update Environment Variables

Edit your `.env` file and replace the placeholder values:

```bash
# Azure AD SSO Configuration
AZURE_AD_TENANT_ID=yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy  # Your Directory (tenant) ID
AZURE_AD_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx  # Your Application (client) ID

# SSO Feature Flags
SSO_ENABLED=true
LOCAL_AUTH_ENABLED=true  # Set to false after migration is complete
```

### 3.2 Update docker-compose.yml (if not already done)

Ensure the backend service has the Azure AD environment variables:

```yaml
backend_app:
  environment:
    # ... existing variables ...
    - AZURE_AD_TENANT_ID=${AZURE_AD_TENANT_ID}
    - AZURE_AD_CLIENT_ID=${AZURE_AD_CLIENT_ID}
    - SSO_ENABLED=${SSO_ENABLED:-true}
    - LOCAL_AUTH_ENABLED=${LOCAL_AUTH_ENABLED:-true}
```

---

## Step 4: Database Setup

### 4.1 Create Required Tables

Connect to your PostgreSQL database and run the following SQL:

```sql
-- Group-to-role mapping table
CREATE TABLE IF NOT EXISTS azure_ad_group_mappings (
    group_id VARCHAR(255) PRIMARY KEY,
    group_name VARCHAR(255) NOT NULL,
    app_role VARCHAR(50) NOT NULL CHECK (app_role IN ('admin', 'write', 'read_only')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Azure AD user tracking table
CREATE TABLE IF NOT EXISTS azure_ad_users (
    azure_oid VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    display_name VARCHAR(255),
    local_user_id INTEGER REFERENCES users(id),
    first_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_azure_users_email ON azure_ad_users(email);
CREATE INDEX IF NOT EXISTS idx_azure_users_local_id ON azure_ad_users(local_user_id);
CREATE INDEX IF NOT EXISTS idx_azure_group_mappings_active ON azure_ad_group_mappings(is_active);
```

### 4.2 Add Group Mappings

Insert the group mappings using the Object IDs from Step 1:

```sql
INSERT INTO azure_ad_group_mappings (group_id, group_name, app_role) VALUES
('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'Recipe-Agent-Admins', 'admin'),
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'Recipe-Agent-Writers', 'write'),
('cccccccc-cccc-cccc-cccc-cccccccccccc', 'Recipe-Agent-Users', 'read_only');
```

Replace the UUIDs with your actual group Object IDs from Azure AD.

---

## Step 5: Deploy and Test

### 5.1 Rebuild and Restart

```bash
# Rebuild containers
docker-compose build

# Restart services
docker-compose down
docker-compose up -d
```

### 5.2 Install Frontend Dependencies

If running locally without Docker rebuild:

```bash
cd app/frontend
npm install
```

### 5.3 Test the SSO Flow

1. Open the application in a browser
2. You should see "Sign in with Microsoft" button on the login page
3. Click the button - you'll be redirected to Microsoft login
4. Sign in with your corporate credentials
5. Grant consent if prompted
6. You should be redirected back to the application dashboard

---

## Step 6: Managing Group Mappings

### Via API (Recommended)

Use the admin endpoints to manage group mappings:

**Get all mappings:**
```bash
curl -H "Authorization: Bearer <admin-token>" \
  https://recipe-agent-agrana.westeurope.cloudapp.azure.com/api/admin/group-mappings
```

**Add/Update a mapping:**
```bash
curl -X POST \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"group_id": "uuid-here", "group_name": "Group Name", "app_role": "write"}' \
  https://recipe-agent-agrana.westeurope.cloudapp.azure.com/api/admin/group-mappings
```

**Delete a mapping:**
```bash
curl -X DELETE \
  -H "Authorization: Bearer <admin-token>" \
  https://recipe-agent-agrana.westeurope.cloudapp.azure.com/api/admin/group-mappings/<group-id>
```

### Via Database

```sql
-- Add a new mapping
INSERT INTO azure_ad_group_mappings (group_id, group_name, app_role)
VALUES ('new-group-uuid', 'New Group Name', 'write');

-- Update a mapping
UPDATE azure_ad_group_mappings
SET app_role = 'admin'
WHERE group_id = 'existing-group-uuid';

-- Disable a mapping (soft delete)
UPDATE azure_ad_group_mappings
SET is_active = false
WHERE group_id = 'group-to-disable';
```

---

## Troubleshooting

### "Azure AD is not properly configured"

- Verify `AZURE_AD_TENANT_ID` and `AZURE_AD_CLIENT_ID` are set correctly
- Check that the values don't have extra quotes or spaces
- Restart the backend service after changing environment variables

### "Invalid or expired Azure AD token"

- Ensure the app registration has the correct redirect URI
- Check that ID tokens are enabled in the app registration
- Verify the token configuration includes the required claims

### User gets "read_only" role when they should have higher access

- Verify the user is a member of the correct Azure AD group
- Check that the group's Object ID matches the one in `azure_ad_group_mappings`
- Ensure `GroupMember.Read.All` permission is granted and admin consented

### "No group mappings found"

- The database tables may not be created - run the SQL from Step 4.1
- Group mappings may not be inserted - run the SQL from Step 4.2
- The group IDs in the database may not match the actual Azure AD group Object IDs

### SSO button doesn't appear

- Check browser console for JavaScript errors
- Verify `SSO_ENABLED=true` in environment variables
- Ensure frontend was rebuilt after adding MSAL dependencies

### Redirect loop after authentication

- Clear browser cache and cookies
- Check that the redirect URI in Azure AD exactly matches the application URL
- Verify the `/auth/callback` route is properly configured

---

## Disabling Local Authentication

Once SSO is fully tested and all users have Azure AD accounts:

1. Update `.env`:
   ```bash
   LOCAL_AUTH_ENABLED=false
   ```

2. Restart the backend:
   ```bash
   docker-compose restart backend_app
   ```

The username/password form will no longer appear, and the `/api/login` endpoint will return an error if called directly.

---

## Security Considerations

1. **Token Storage**: MSAL stores tokens in `sessionStorage` (cleared when browser closes)
2. **Token Validation**: All Azure AD tokens are validated server-side before issuing internal JWTs
3. **Group Claims**: Only necessary group claims are requested to minimize token size
4. **CORS**: Ensure CORS is configured to only allow your application domains
5. **Secrets**: Never commit Azure AD credentials to source control

---

## Reference

- [Microsoft Entra ID Documentation](https://learn.microsoft.com/en-us/entra/identity/)
- [MSAL.js Documentation](https://github.com/AzureAD/microsoft-authentication-library-for-js)
- [Azure AD App Registration](https://learn.microsoft.com/en-us/entra/identity-platform/quickstart-register-app)
