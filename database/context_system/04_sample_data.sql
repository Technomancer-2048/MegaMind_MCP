-- Context Database System - Sample Data
-- Phase 1: Test data for validation

-- Sample chunks for testing (with access_count = 1 since creation counts as first access)
INSERT INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count) VALUES
('sql_standards_triggers_001', 
'# Database Triggers\n\nTriggers are special stored procedures that automatically execute in response to database events.\n\n## Types of Triggers\n- BEFORE triggers: Execute before the triggering event\n- AFTER triggers: Execute after the triggering event\n- INSTEAD OF triggers: Replace the triggering event\n\n## Best Practices\n- Keep trigger logic simple and fast\n- Avoid recursive triggers\n- Use triggers for data validation and auditing', 
'sql_standards.md', 
'/database/triggers', 
'section', 
15, 
120, 
1),

('sql_standards_functions_001', 
'# Stored Functions\n\nStored functions are reusable code blocks that return a single value.\n\n```sql\nCREATE FUNCTION calculate_discount(price DECIMAL, discount_rate DECIMAL)\nRETURNS DECIMAL\nDETERMINISTIC\nBEGIN\n    RETURN price * (1 - discount_rate);\nEND\n```\n\n## Function Guidelines\n- Functions should be deterministic when possible\n- Use proper parameter validation\n- Return appropriate data types', 
'sql_standards.md', 
'/database/functions', 
'function', 
18, 
150, 
1),

('sql_standards_security_001', 
'# Database Security\n\n## Access Control\n- Use principle of least privilege\n- Create role-based access patterns\n- Regularly audit user permissions\n\n## SQL Injection Prevention\n- Always use parameterized queries\n- Validate input data types\n- Escape special characters\n\n```sql\n-- Good: Parameterized query\nSELECT * FROM users WHERE id = ?\n\n-- Bad: String concatenation\nSELECT * FROM users WHERE id = '' + user_input + ''\n```', 
'sql_standards.md', 
'/security/access_control', 
'rule', 
22, 
180, 
1),

('mcp_server_setup_001', 
'# MCP Server Setup\n\n## Installation\n```bash\nnpm install @modelcontextprotocol/server\n```\n\n## Basic Configuration\n```python\nfrom mcp import Server\n\nserver = Server("my-server")\n\n@server.tool()\ndef my_tool(input: str) -> str:\n    return f"Processed: {input}"\n```\n\n## Error Handling\n- Implement proper exception handling\n- Return meaningful error messages\n- Log errors for debugging', 
'mcp_documentation.md', 
'/setup/installation', 
'example', 
25, 
200, 
1);

-- Sample relationships
INSERT INTO megamind_chunk_relationships (chunk_id, related_chunk_id, relationship_type, strength, discovered_by) VALUES
('sql_standards_triggers_001', 'sql_standards_functions_001', 'references', 0.75, 'semantic_similarity'),
('sql_standards_functions_001', 'sql_standards_security_001', 'enhances', 0.60, 'ai_analysis'),
('sql_standards_security_001', 'sql_standards_triggers_001', 'depends_on', 0.80, 'manual');

-- Sample tags
INSERT INTO megamind_chunk_tags (chunk_id, tag_type, tag_value, confidence, created_by) VALUES
('sql_standards_triggers_001', 'subsystem', 'database', 1.0, 'manual'),
('sql_standards_triggers_001', 'function_type', 'automation', 0.9, 'ai_analysis'),
('sql_standards_triggers_001', 'difficulty', 'intermediate', 0.8, 'ai_analysis'),

('sql_standards_functions_001', 'subsystem', 'database', 1.0, 'manual'),
('sql_standards_functions_001', 'function_type', 'utility', 0.9, 'ai_analysis'),
('sql_standards_functions_001', 'language', 'sql', 1.0, 'automatic'),

('sql_standards_security_001', 'subsystem', 'security', 1.0, 'manual'),
('sql_standards_security_001', 'function_type', 'validation', 0.95, 'ai_analysis'),
('sql_standards_security_001', 'difficulty', 'advanced', 0.9, 'ai_analysis'),

('mcp_server_setup_001', 'subsystem', 'mcp', 1.0, 'manual'),
('mcp_server_setup_001', 'function_type', 'configuration', 0.9, 'ai_analysis'),
('mcp_server_setup_001', 'language', 'python', 1.0, 'automatic');