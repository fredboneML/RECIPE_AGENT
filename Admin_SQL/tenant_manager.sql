-- Add new tenant
INSERT INTO tenant_codes (tenant_code, tenant_code_alias) 
VALUES ('maccare', 'maccare');

-- Add new tenant
INSERT INTO tenant_codes (tenant_code, tenant_code_alias) 
VALUES ('stroevemotorsport', 'stroevemotorsport');
-- Trigger automatically creates partition, indexes, and role

-- Add new user (minimal)
INSERT INTO users (username, tenant_code) 
VALUES ('newuser', 'newtenant');
-- Trigger automatically generates id, password, and sets role

-- Add new user (with custom role)
INSERT INTO users (username, tenant_code, role) 
VALUES ('poweruser', 'newtenant', 'admin');
-- Trigger handles id and password generation