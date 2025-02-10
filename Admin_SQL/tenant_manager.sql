


-- Add new tenant
INSERT INTO tenant_codes (tenant_code, tenant_code_alias) 
VALUES ('maccare', 'maccare');

-- Add new tenant
INSERT INTO tenant_codes (tenant_code, tenant_code_alias) 
VALUES ('stroevemotorsport', 'stroevemotorsport');
-- Trigger automatically creates partition, indexes, and role

-- 1. Insert the new user
INSERT INTO users (username, tenant_code, role)
VALUES ('newuser_mcre', 'maccare', 'read_only');

-- 2. Immediately get the generated password
SELECT * FROM get_generated_password('newuser_mcre');


INSERT INTO users (username, tenant_code) 
VALUES ('newuser', 'newtenant');
-- Trigger automatically generates id, password, and sets role

-- Add new user (with custom role)
INSERT INTO users (username, tenant_code, role) 
VALUES ('poweruser', 'newtenant', 'admin');
-- Trigger handles id and password generation