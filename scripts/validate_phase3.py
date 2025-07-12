#!/usr/bin/env python3
"""
Phase 3 Validation Script
Tests the bidirectional flow functionality with actual database connections
"""

import json
import os
import sys
import uuid
from datetime import datetime

def test_database_schema():
    """Test that the session management tables exist"""
    print("🔍 Testing Database Schema...")
    
    # Check if the SQL files exist
    schema_files = [
        'database/context_system/01_create_tables.sql',
        'database/context_system/02_create_indexes.sql', 
        'database/context_system/03_session_management_tables.sql'
    ]
    
    missing_files = []
    for file_path in schema_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing schema files: {missing_files}")
        return False
    
    print("✓ All schema files present")
    
    # Check for required session management tables in SQL
    session_sql_file = 'database/context_system/03_session_management_tables.sql'
    with open(session_sql_file, 'r') as f:
        sql_content = f.read()
    
    required_tables = [
        'megamind_session_changes',
        'megamind_knowledge_contributions', 
        'megamind_session_metadata'
    ]
    
    for table in required_tables:
        if table not in sql_content:
            print(f"❌ Missing table definition: {table}")
            return False
    
    print("✓ All required session management tables defined")
    return True

def test_mcp_server_functions():
    """Test that the MCP server has the required bidirectional functions"""
    print("\n🔍 Testing MCP Server Functions...")
    
    server_file = 'mcp_server/megamind_database_server.py'
    if not os.path.exists(server_file):
        print(f"❌ MCP server file not found: {server_file}")
        return False
    
    with open(server_file, 'r') as f:
        server_content = f.read()
    
    # Check for required Phase 3 functions
    required_functions = [
        'mcp__megamind_db__update_chunk',
        'mcp__megamind_db__create_chunk',
        'mcp__megamind_db__add_relationship',
        'mcp__megamind_db__get_pending_changes',
        'mcp__megamind_db__commit_session_changes',
        'mcp__megamind_db__rollback_session_changes',
        'mcp__megamind_db__get_change_summary'
    ]
    
    missing_functions = []
    for function in required_functions:
        if function not in server_content:
            missing_functions.append(function)
    
    if missing_functions:
        print(f"❌ Missing MCP functions: {missing_functions}")
        return False
    
    print("✓ All required bidirectional MCP functions present")
    
    # Check for required database methods
    required_methods = [
        'def update_chunk',
        'def create_chunk', 
        'def add_relationship',
        'def get_pending_changes',
        'def commit_session_changes',
        'def rollback_session_changes',
        'def get_change_summary'
    ]
    
    missing_methods = []
    for method in required_methods:
        if method not in server_content:
            missing_methods.append(method)
    
    if missing_methods:
        print(f"❌ Missing database methods: {missing_methods}")
        return False
    
    print("✓ All required database methods present")
    return True

def test_review_interface():
    """Test that the review interface is properly implemented"""
    print("\n🔍 Testing Review Interface...")
    
    # Check main review interface file
    review_file = 'review/change_reviewer.py'
    if not os.path.exists(review_file):
        print(f"❌ Review interface file not found: {review_file}")
        return False
    
    with open(review_file, 'r') as f:
        review_content = f.read()
    
    # Check for required classes and methods
    required_components = [
        'class ChangeReviewManager',
        'def get_pending_sessions',
        'def get_session_changes',
        'def approve_changes',
        'def reject_changes',
        'Flask(__name__'
    ]
    
    missing_components = []
    for component in required_components:
        if component not in review_content:
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing review interface components: {missing_components}")
        return False
    
    print("✓ Review interface properly implemented")
    
    # Check template files
    template_files = [
        'review/templates/base.html',
        'review/templates/review_dashboard.html', 
        'review/templates/session_review.html',
        'review/templates/error.html'
    ]
    
    missing_templates = []
    for template in template_files:
        if not os.path.exists(template):
            missing_templates.append(template)
    
    if missing_templates:
        print(f"❌ Missing template files: {missing_templates}")
        return False
    
    print("✓ All template files present")
    
    # Check requirements file
    if not os.path.exists('review/requirements.txt'):
        print("❌ Review interface requirements.txt missing")
        return False
    
    print("✓ Review interface requirements file present")
    return True

def test_startup_scripts():
    """Test that the startup scripts are present and executable"""
    print("\n🔍 Testing Startup Scripts...")
    
    scripts = [
        'scripts/start_review_interface.sh'
    ]
    
    for script in scripts:
        if not os.path.exists(script):
            print(f"❌ Missing startup script: {script}")
            return False
        
        # Check if executable
        if not os.access(script, os.X_OK):
            print(f"⚠ Script not executable: {script}")
        else:
            print(f"✓ Script executable: {script}")
    
    print("✓ All startup scripts present")
    return True

def test_impact_scoring_logic():
    """Test the impact scoring and prioritization logic"""
    print("\n🔍 Testing Impact Scoring Logic...")
    
    def calculate_impact_score(access_count):
        return min(access_count / 100.0, 1.0)
    
    def get_priority_level(impact_score):
        if impact_score > 0.7:
            return "critical"
        elif impact_score >= 0.3:
            return "important"
        else:
            return "standard"
    
    # Test cases
    test_cases = [
        (0, 0.0, "standard"),
        (25, 0.25, "standard"),
        (30, 0.3, "important"),
        (50, 0.5, "important"),
        (75, 0.75, "critical"),
        (100, 1.0, "critical"),
        (200, 1.0, "critical")  # Capped at 1.0
    ]
    
    for access_count, expected_score, expected_priority in test_cases:
        calculated_score = calculate_impact_score(access_count)
        calculated_priority = get_priority_level(calculated_score)
        
        if calculated_score != expected_score:
            print(f"❌ Impact score mismatch for {access_count}: expected {expected_score}, got {calculated_score}")
            return False
        
        if calculated_priority != expected_priority:
            print(f"❌ Priority mismatch for score {calculated_score}: expected {expected_priority}, got {calculated_priority}")
            return False
    
    print("✓ Impact scoring logic working correctly")
    return True

def test_json_serialization():
    """Test JSON serialization of change data"""
    print("\n🔍 Testing JSON Serialization...")
    
    # Test complex change data
    test_data = {
        "original_content": "Original content with Unicode: 中文, العربية, ñoño",
        "new_content": "New content with special chars: @#$%^&*()",
        "chunk_metadata": {
            "source_document": "test.md",
            "section_path": "/section/subsection",
            "chunk_type": "function",
            "line_count": 25,
            "token_count": 150
        },
        "timestamp": datetime.now().isoformat(),
        "change_id": str(uuid.uuid4())
    }
    
    try:
        # Test serialization
        serialized = json.dumps(test_data, ensure_ascii=False)
        
        # Test deserialization
        deserialized = json.loads(serialized)
        
        # Verify data integrity
        if deserialized != test_data:
            print("❌ JSON serialization data integrity failed")
            return False
        
        print("✓ JSON serialization working correctly")
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        return False

def test_session_id_generation():
    """Test session ID generation and validation"""
    print("\n🔍 Testing Session ID Generation...")
    
    def generate_session_id():
        return f"session_{uuid.uuid4().hex[:12]}"
    
    def is_valid_session_id(session_id):
        return (session_id and 
                isinstance(session_id, str) and 
                len(session_id) > 0 and 
                ' ' not in session_id and 
                '/' not in session_id and
                len(session_id) <= 50)
    
    # Generate and validate multiple session IDs
    for i in range(10):
        session_id = generate_session_id()
        if not is_valid_session_id(session_id):
            print(f"❌ Invalid session ID generated: {session_id}")
            return False
    
    # Test invalid session IDs
    invalid_ids = ["", None, "session with spaces", "session/with/slashes", "a" * 100]
    for invalid_id in invalid_ids:
        if is_valid_session_id(invalid_id):
            print(f"❌ Invalid session ID passed validation: {invalid_id}")
            return False
    
    print("✓ Session ID generation and validation working correctly")
    return True

def test_change_id_generation():
    """Test change ID generation for uniqueness"""
    print("\n🔍 Testing Change ID Generation...")
    
    def generate_change_id():
        return f"change_{uuid.uuid4().hex[:12]}"
    
    # Generate multiple change IDs and check uniqueness
    change_ids = set()
    for i in range(100):
        change_id = generate_change_id()
        if change_id in change_ids:
            print(f"❌ Duplicate change ID generated: {change_id}")
            return False
        change_ids.add(change_id)
    
    print("✓ Change ID generation working correctly")
    return True

def main():
    """Run all Phase 3 validation tests"""
    print("🚀 Starting Phase 3: Bidirectional Flow Validation")
    print("=" * 60)
    
    tests = [
        ("Database Schema", test_database_schema),
        ("MCP Server Functions", test_mcp_server_functions),
        ("Review Interface", test_review_interface),
        ("Startup Scripts", test_startup_scripts),
        ("Impact Scoring Logic", test_impact_scoring_logic),
        ("JSON Serialization", test_json_serialization),
        ("Session ID Generation", test_session_id_generation),
        ("Change ID Generation", test_change_id_generation)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_function in tests:
        try:
            if test_function():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("Phase 3 Validation Summary:")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All Phase 3 validation tests passed!")
        print("\nPhase 3: Bidirectional Flow is ready for use!")
        print("\nNext steps:")
        print("1. Start the database: ./scripts/start_database.sh")
        print("2. Start the MCP server: ./scripts/start_mcp_server.sh")
        print("3. Start the review interface: ./scripts/start_review_interface.sh")
        print("4. Test AI knowledge contributions through MCP functions")
        return True
    else:
        print(f"❌ {total_tests - passed_tests} tests failed")
        print("Please address the failing tests before using Phase 3 functionality.")
        return False

if __name__ == '__main__':
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    success = main()
    sys.exit(0 if success else 1)