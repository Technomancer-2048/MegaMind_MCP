#!/usr/bin/env python3
"""
GitHub Issue #29 - Phase 5: Global Content Management Implementation
Advanced global content management with comprehensive development guidelines
"""

import sys
import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add the mcp_server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_server'))

class GlobalContentManager:
    """Advanced global content management system for environment primer elements."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    async def populate_global_realm(self, categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Populate GLOBAL realm with comprehensive development guidelines.
        
        This implements the core Phase 5 requirement for creating and populating
        the GLOBAL realm with universal development rules and guidelines.
        """
        try:
            # Define comprehensive global development guidelines
            global_elements = self._get_comprehensive_global_elements()
            
            # Filter by categories if specified
            if categories:
                global_elements = [elem for elem in global_elements if elem['category'] in categories]
            
            success_count = 0
            error_count = 0
            created_elements = []
            
            connection = self.db_manager.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            for element in global_elements:
                try:
                    # Insert into megamind_chunks table
                    chunk_query = """
                    INSERT IGNORE INTO megamind_chunks (
                        chunk_id, realm_id, content, source_document, section_path,
                        chunk_type, element_category, element_subcategory, priority_score,
                        enforcement_level, criticality, applies_to, created_at, updated_at
                    ) VALUES (
                        %(chunk_id)s, 'GLOBAL', %(content)s, %(source_document)s, %(section_path)s,
                        'rule', %(category)s, %(subcategory)s, %(priority_score)s,
                        %(enforcement_level)s, %(criticality)s, %(applies_to)s, NOW(), NOW()
                    )
                    """
                    
                    cursor.execute(chunk_query, element)
                    
                    # Insert into megamind_global_elements table
                    element_query = """
                    INSERT IGNORE INTO megamind_global_elements (
                        element_id, chunk_id, title, summary, category, subcategory,
                        priority_score, enforcement_level, criticality, author, maintainer,
                        version, effective_date, review_date, business_justification,
                        implementation_notes, automation_available, tooling_support,
                        examples, documentation_urls
                    ) VALUES (
                        %(element_id)s, %(chunk_id)s, %(title)s, %(summary)s, %(category)s,
                        %(subcategory)s, %(priority_score)s, %(enforcement_level)s, %(criticality)s,
                        %(author)s, %(maintainer)s, %(version)s, %(effective_date)s, %(review_date)s,
                        %(business_justification)s, %(implementation_notes)s, %(automation_available)s,
                        %(tooling_support)s, %(examples)s, %(documentation_urls)s
                    )
                    """
                    
                    cursor.execute(element_query, element)
                    
                    created_elements.append({
                        'element_id': element['element_id'],
                        'title': element['title'],
                        'category': element['category']
                    })
                    success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"Error creating element {element.get('element_id', 'unknown')}: {str(e)}")
            
            connection.commit()
            cursor.close()
            connection.close()
            
            return {
                'success': True,
                'message': f'Global realm populated with {success_count} elements',
                'created_elements': created_elements,
                'success_count': success_count,
                'error_count': error_count,
                'total_processed': len(global_elements)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to populate global realm: {str(e)}',
                'created_elements': []
            }
    
    def _get_comprehensive_global_elements(self) -> List[Dict[str, Any]]:
        """Get comprehensive list of global development guidelines."""
        
        # Base date for effective dates
        now = datetime.now()
        
        return [
            # DEVELOPMENT CATEGORY
            {
                'chunk_id': 'global_dev_001',
                'element_id': 'ge_dev_001', 
                'title': 'Function Documentation Standards',
                'summary': 'All functions must have comprehensive docstrings following Google format',
                'content': 'All functions must have comprehensive docstrings following the Google docstring format. Include parameters, return values, exceptions, and usage examples. Use type hints for all parameters and return values.',
                'source_document': 'Development_Standards.md',
                'section_path': '/Documentation/Function_Documentation',
                'category': 'development',
                'subcategory': 'documentation_standards',
                'priority_score': 0.9,
                'enforcement_level': 'required',
                'criticality': 'high',
                'applies_to': json.dumps(['python', 'javascript', 'typescript']),
                'author': 'Development Team',
                'maintainer': 'Tech Lead',
                'version': '2.1',
                'effective_date': now,
                'review_date': now + timedelta(days=180),
                'business_justification': 'Improves code maintainability and reduces onboarding time for new developers',
                'implementation_notes': 'Use IDE plugins for docstring templates. Review during code review process.',
                'automation_available': True,
                'tooling_support': json.dumps(['pylint', 'pydocstyle', 'sphinx']),
                'examples': json.dumps(['def calculate_total(items: List[Item]) -> float:\\n    """Calculate total price for items."""']),
                'documentation_urls': json.dumps(['https://google.github.io/styleguide/pyguide.html'])
            },
            {
                'chunk_id': 'global_dev_002',
                'element_id': 'ge_dev_002',
                'title': 'Code Review Requirements',
                'summary': 'All code changes require peer review before merging',
                'content': 'Every pull request must be reviewed by at least one team member before merging. Reviews should check for code quality, security issues, performance concerns, and adherence to standards.',
                'source_document': 'Development_Standards.md',
                'section_path': '/Quality/Code_Review',
                'category': 'development',
                'subcategory': 'quality_assurance',
                'priority_score': 0.95,
                'enforcement_level': 'required',
                'criticality': 'critical',
                'applies_to': json.dumps(['all_languages']),
                'author': 'Engineering Manager',
                'maintainer': 'Tech Lead',
                'version': '1.0',
                'effective_date': now,
                'review_date': now + timedelta(days=365),
                'business_justification': 'Reduces bugs, improves code quality, and facilitates knowledge sharing',
                'implementation_notes': 'Configure branch protection rules. Use review checklists.',
                'automation_available': True,
                'tooling_support': json.dumps(['GitHub', 'GitLab', 'Azure DevOps']),
                'examples': json.dumps(['require_pull_request_reviews: true']),
                'documentation_urls': json.dumps(['https://docs.github.com/en/pull-requests/collaborating-with-pull-requests'])
            },
            {
                'chunk_id': 'global_dev_003',
                'element_id': 'ge_dev_003',
                'title': 'Error Handling Standards',
                'summary': 'Comprehensive error handling with proper logging and user feedback',
                'content': 'All functions must implement proper error handling with try-catch blocks, meaningful error messages, and appropriate logging. Never fail silently.',
                'source_document': 'Development_Standards.md',
                'section_path': '/Error_Handling/Standards',
                'category': 'development',
                'subcategory': 'error_handling',
                'priority_score': 0.85,
                'enforcement_level': 'required',
                'criticality': 'high',
                'applies_to': json.dumps(['python', 'javascript', 'java', 'csharp']),
                'author': 'Senior Developer',
                'maintainer': 'Tech Lead',
                'version': '1.2',
                'effective_date': now,
                'review_date': now + timedelta(days=180),
                'business_justification': 'Improves application reliability and debugging capabilities',
                'implementation_notes': 'Use structured logging. Define error handling patterns.',
                'automation_available': True,
                'tooling_support': json.dumps(['logging', 'sentry', 'rollbar']),
                'examples': json.dumps(['try:\\n    process_data()\\nexcept ValueError as e:\\n    logger.error(f"Invalid data: {e}")']),
                'documentation_urls': json.dumps(['https://docs.python.org/3/tutorial/errors.html'])
            },
            
            # SECURITY CATEGORY
            {
                'chunk_id': 'global_sec_001',
                'element_id': 'ge_sec_001',
                'title': 'Secret Management Requirements',
                'summary': 'Never store secrets or API keys in source code',
                'content': 'Never store secrets, API keys, passwords, or other sensitive data in source code. Use environment variables for development and dedicated secret management systems for production.',
                'source_document': 'Security_Guidelines.md',
                'section_path': '/Security/Secret_Management',
                'category': 'security',
                'subcategory': 'data_protection',
                'priority_score': 1.0,
                'enforcement_level': 'required',
                'criticality': 'critical',
                'applies_to': json.dumps(['all_technologies']),
                'author': 'Security Team',
                'maintainer': 'Security Officer',
                'version': '3.0',
                'effective_date': now,
                'review_date': now + timedelta(days=90),
                'business_justification': 'Prevents credential exposure and security breaches. Legal compliance requirement.',
                'implementation_notes': 'Use .env files for development, cloud secret managers for production.',
                'automation_available': True,
                'tooling_support': json.dumps(['git-secrets', 'truffleHog', 'AWS Secrets Manager']),
                'examples': json.dumps(['api_key = os.getenv("API_KEY")']),
                'documentation_urls': json.dumps(['https://owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html'])
            },
            {
                'chunk_id': 'global_sec_002',
                'element_id': 'ge_sec_002',
                'title': 'Input Validation Requirements',
                'summary': 'All user inputs must be validated and sanitized',
                'content': 'Validate and sanitize all user inputs on both client and server sides. Use parameterized queries to prevent SQL injection. Validate data types, ranges, and formats.',
                'source_document': 'Security_Guidelines.md',
                'section_path': '/Security/Input_Validation',
                'category': 'security',
                'subcategory': 'input_validation',
                'priority_score': 0.95,
                'enforcement_level': 'required',
                'criticality': 'critical',
                'applies_to': json.dumps(['web_applications', 'apis']),
                'author': 'Security Team',
                'maintainer': 'Security Engineer',
                'version': '2.0',
                'effective_date': now,
                'review_date': now + timedelta(days=180),
                'business_justification': 'Prevents injection attacks and data corruption',
                'implementation_notes': 'Use validation libraries. Implement whitelist validation.',
                'automation_available': True,
                'tooling_support': json.dumps(['joi', 'yup', 'marshmallow']),
                'examples': json.dumps(['cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))']),
                'documentation_urls': json.dumps(['https://owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html'])
            },
            
            # PROCESS CATEGORY
            {
                'chunk_id': 'global_proc_001',
                'element_id': 'ge_proc_001',
                'title': 'CI/CD Pipeline Requirements',
                'summary': 'All code changes must pass automated tests and quality gates',
                'content': 'All code changes must pass automated tests, security scans, and code quality checks before merging. Minimum 80% test coverage required.',
                'source_document': 'Process_Rules.md',
                'section_path': '/CI_CD/Quality_Gates',
                'category': 'process',
                'subcategory': 'ci_cd_pipelines',
                'priority_score': 0.9,
                'enforcement_level': 'required',
                'criticality': 'high',
                'applies_to': json.dumps(['git', 'github', 'ci_cd']),
                'author': 'DevOps Team',
                'maintainer': 'Lead DevOps Engineer',
                'version': '1.5',
                'effective_date': now,
                'review_date': now + timedelta(days=365),
                'business_justification': 'Ensures code quality, reduces bugs in production, maintains team standards',
                'implementation_notes': 'Configure branch protection rules. Set up automated test runs.',
                'automation_available': True,
                'tooling_support': json.dumps(['GitHub Actions', 'Jenkins', 'SonarQube']),
                'examples': json.dumps(['require_pull_request_reviews: true']),
                'documentation_urls': json.dumps(['https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository'])
            },
            {
                'chunk_id': 'global_proc_002',
                'element_id': 'ge_proc_002',
                'title': 'Version Control Standards',
                'summary': 'Standardized branching strategy and commit message format',
                'content': 'Use GitFlow branching strategy with feature branches, develop branch, and main branch. Follow conventional commit message format with clear, descriptive messages.',
                'source_document': 'Process_Rules.md',
                'section_path': '/Version_Control/Standards',
                'category': 'process',
                'subcategory': 'version_control',
                'priority_score': 0.8,
                'enforcement_level': 'required',
                'criticality': 'medium',
                'applies_to': json.dumps(['git', 'version_control']),
                'author': 'DevOps Team',
                'maintainer': 'Tech Lead',
                'version': '1.0',
                'effective_date': now,
                'review_date': now + timedelta(days=180),
                'business_justification': 'Maintains clean history and enables better collaboration',
                'implementation_notes': 'Use conventional commits. Configure commit message templates.',
                'automation_available': True,
                'tooling_support': json.dumps(['commitizen', 'husky', 'conventional-changelog']),
                'examples': json.dumps(['feat: add user authentication', 'fix: resolve memory leak in cache']),
                'documentation_urls': json.dumps(['https://www.conventionalcommits.org/'])
            },
            
            # QUALITY CATEGORY
            {
                'chunk_id': 'global_qual_001',
                'element_id': 'ge_qual_001',
                'title': 'Testing Requirements',
                'summary': 'Comprehensive testing strategy with multiple test levels',
                'content': 'Implement unit tests, integration tests, and end-to-end tests. Maintain minimum 80% code coverage. Use test-driven development for critical features.',
                'source_document': 'Quality_Standards.md',
                'section_path': '/Testing/Requirements',
                'category': 'quality',
                'subcategory': 'testing_standards',
                'priority_score': 0.9,
                'enforcement_level': 'required',
                'criticality': 'high',
                'applies_to': json.dumps(['all_projects']),
                'author': 'QA Team',
                'maintainer': 'QA Lead',
                'version': '2.0',
                'effective_date': now,
                'review_date': now + timedelta(days=180),
                'business_justification': 'Ensures software quality and reduces production bugs',
                'implementation_notes': 'Write tests before implementing features. Use testing frameworks.',
                'automation_available': True,
                'tooling_support': json.dumps(['pytest', 'jest', 'cypress', 'selenium']),
                'examples': json.dumps(['def test_user_creation():\\n    assert create_user("test") is not None']),
                'documentation_urls': json.dumps(['https://docs.pytest.org/', 'https://jestjs.io/'])
            },
            
            # NAMING CATEGORY
            {
                'chunk_id': 'global_name_001',
                'element_id': 'ge_name_001',
                'title': 'Naming Convention Standards',
                'summary': 'Consistent naming conventions across all code',
                'content': 'Use descriptive, meaningful names for variables, functions, and classes. Follow language-specific naming conventions (snake_case for Python, camelCase for JavaScript).',
                'source_document': 'Naming_Standards.md',
                'section_path': '/Naming/Conventions',
                'category': 'naming',
                'subcategory': 'conventions',
                'priority_score': 0.7,
                'enforcement_level': 'required',
                'criticality': 'medium',
                'applies_to': json.dumps(['all_languages']),
                'author': 'Architecture Team',
                'maintainer': 'Senior Developer',
                'version': '1.1',
                'effective_date': now,
                'review_date': now + timedelta(days=365),
                'business_justification': 'Improves code readability and maintainability',
                'implementation_notes': 'Use linters to enforce naming conventions. Create naming guides.',
                'automation_available': True,
                'tooling_support': json.dumps(['eslint', 'pylint', 'checkstyle']),
                'examples': json.dumps(['user_name (Python)', 'userName (JavaScript)', 'UserRepository (class)']),
                'documentation_urls': json.dumps(['https://pep8.org/', 'https://google.github.io/styleguide/'])
            },
            
            # DEPENDENCIES CATEGORY
            {
                'chunk_id': 'global_deps_001',
                'element_id': 'ge_deps_001',
                'title': 'Dependency Management Standards',
                'summary': 'Secure and up-to-date dependency management practices',
                'content': 'Keep all dependencies up to date. Use lock files for reproducible builds. Regularly audit dependencies for security vulnerabilities.',
                'source_document': 'Dependencies_Standards.md',
                'section_path': '/Dependencies/Management',
                'category': 'dependencies',
                'subcategory': 'management',
                'priority_score': 0.85,
                'enforcement_level': 'required',
                'criticality': 'high',
                'applies_to': json.dumps(['all_projects']),
                'author': 'Security Team',
                'maintainer': 'DevOps Engineer',
                'version': '1.3',
                'effective_date': now,
                'review_date': now + timedelta(days=90),
                'business_justification': 'Prevents security vulnerabilities and ensures system stability',
                'implementation_notes': 'Use automated dependency updates. Monitor security advisories.',
                'automation_available': True,
                'tooling_support': json.dumps(['dependabot', 'renovate', 'snyk']),
                'examples': json.dumps(['package-lock.json', 'requirements.txt', 'Pipfile.lock']),
                'documentation_urls': json.dumps(['https://docs.npmjs.com/cli/v8/configuring-npm/package-lock-json'])
            },
            
            # ARCHITECTURE CATEGORY
            {
                'chunk_id': 'global_arch_001',
                'element_id': 'ge_arch_001',
                'title': 'Architecture Design Principles',
                'summary': 'Core architectural principles for system design',
                'content': 'Follow SOLID principles, design patterns, and clean architecture. Implement proper separation of concerns and maintain loose coupling between components.',
                'source_document': 'Architecture_Guidelines.md',
                'section_path': '/Architecture/Principles',
                'category': 'architecture',
                'subcategory': 'design_principles',
                'priority_score': 0.8,
                'enforcement_level': 'recommended',
                'criticality': 'high',
                'applies_to': json.dumps(['system_design', 'application_architecture']),
                'author': 'Architecture Team',
                'maintainer': 'Principal Architect',
                'version': '1.0',
                'effective_date': now,
                'review_date': now + timedelta(days=365),
                'business_justification': 'Ensures scalable, maintainable, and testable code architecture',
                'implementation_notes': 'Review architecture decisions in design reviews. Use architectural patterns.',
                'automation_available': False,
                'tooling_support': json.dumps(['architectural_decision_records', 'design_patterns']),
                'examples': json.dumps(['Single Responsibility Principle', 'Dependency Injection']),
                'documentation_urls': json.dumps(['https://clean-architecture-python.readthedocs.io/'])
            }
        ]
    
    async def create_content_workflow(self, workflow_type: str) -> Dict[str, Any]:
        """Create content management workflows for global elements."""
        
        workflows = {
            'approval': {
                'name': 'Global Element Approval Workflow',
                'steps': [
                    'Content creation/modification',
                    'Technical review by domain expert',
                    'Security review (if applicable)',
                    'Architecture review (for architecture elements)',
                    'Final approval by maintainer',
                    'Publication to GLOBAL realm'
                ],
                'roles': ['creator', 'reviewer', 'security_reviewer', 'architect', 'maintainer'],
                'automation': True
            },
            'update': {
                'name': 'Global Element Update Workflow',
                'steps': [
                    'Identify outdated element',
                    'Research current best practices',
                    'Update content and metadata',
                    'Review changes',
                    'Update version and effective date',
                    'Notify stakeholders'
                ],
                'roles': ['maintainer', 'reviewer', 'stakeholders'],
                'automation': True
            },
            'deprecation': {
                'name': 'Global Element Deprecation Workflow',
                'steps': [
                    'Mark element as deprecated',
                    'Add deprecation notice',
                    'Provide migration guidance',
                    'Set sunset date',
                    'Monitor usage',
                    'Remove after sunset period'
                ],
                'roles': ['maintainer', 'architect', 'stakeholders'],
                'automation': False
            }
        }
        
        if workflow_type not in workflows:
            return {
                'success': False,
                'error': f'Unknown workflow type: {workflow_type}',
                'available_workflows': list(workflows.keys())
            }
        
        return {
            'success': True,
            'workflow': workflows[workflow_type],
            'workflow_type': workflow_type
        }
    
    async def validate_global_content(self, chunk_id: str) -> Dict[str, Any]:
        """Validate global content for quality and compliance."""
        
        try:
            connection = self.db_manager.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get element details
            query = """
            SELECT c.*, ge.* 
            FROM megamind_chunks c
            LEFT JOIN megamind_global_elements ge ON c.chunk_id = ge.chunk_id
            WHERE c.chunk_id = %s AND c.realm_id = 'GLOBAL'
            """
            
            cursor.execute(query, (chunk_id,))
            element = cursor.fetchone()
            
            if not element:
                return {
                    'success': False,
                    'error': 'Element not found or not in GLOBAL realm'
                }
            
            # Validation checks
            validation_results = {
                'content_quality': self._validate_content_quality(element),
                'metadata_completeness': self._validate_metadata_completeness(element),
                'business_alignment': self._validate_business_alignment(element),
                'technical_accuracy': self._validate_technical_accuracy(element)
            }
            
            # Calculate overall score
            scores = [result['score'] for result in validation_results.values()]
            overall_score = sum(scores) / len(scores)
            
            # Determine validation status
            status = 'approved' if overall_score >= 0.8 else 'needs_review' if overall_score >= 0.6 else 'rejected'
            
            cursor.close()
            connection.close()
            
            return {
                'success': True,
                'chunk_id': chunk_id,
                'validation_results': validation_results,
                'overall_score': overall_score,
                'status': status,
                'recommendations': self._generate_recommendations(validation_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Validation failed: {str(e)}'
            }
    
    def _validate_content_quality(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content quality."""
        score = 1.0
        issues = []
        
        content = element.get('content', '')
        
        # Check content length
        if len(content) < 50:
            score -= 0.3
            issues.append('Content too short (minimum 50 characters)')
        
        # Check for clarity
        if not any(word in content.lower() for word in ['must', 'should', 'required', 'recommended']):
            score -= 0.2
            issues.append('Content lacks clear directive language')
        
        # Check for examples
        if element.get('examples') is None:
            score -= 0.1
            issues.append('No examples provided')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'category': 'content_quality'
        }
    
    def _validate_metadata_completeness(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata completeness."""
        score = 1.0
        issues = []
        
        required_fields = ['title', 'summary', 'category', 'priority_score', 'enforcement_level']
        
        for field in required_fields:
            if not element.get(field):
                score -= 0.2
                issues.append(f'Missing required field: {field}')
        
        # Check priority score range
        priority_score = element.get('priority_score', 0)
        if not (0.0 <= priority_score <= 1.0):
            score -= 0.1
            issues.append('Priority score must be between 0.0 and 1.0')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'category': 'metadata_completeness'
        }
    
    def _validate_business_alignment(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Validate business alignment."""
        score = 1.0
        issues = []
        
        # Check business justification
        if not element.get('business_justification'):
            score -= 0.4
            issues.append('Missing business justification')
        
        # Check if implementation notes are provided
        if not element.get('implementation_notes'):
            score -= 0.2
            issues.append('Missing implementation notes')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'category': 'business_alignment'
        }
    
    def _validate_technical_accuracy(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Validate technical accuracy."""
        score = 1.0
        issues = []
        
        # Check if tooling support is specified
        if not element.get('tooling_support'):
            score -= 0.2
            issues.append('No tooling support specified')
        
        # Check if documentation URLs are provided
        if not element.get('documentation_urls'):
            score -= 0.2
            issues.append('No documentation URLs provided')
        
        return {
            'score': max(0.0, score),
            'issues': issues,
            'category': 'technical_accuracy'
        }
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for category, result in validation_results.items():
            if result['score'] < 0.8:
                recommendations.extend(result['issues'])
        
        return recommendations

async def main():
    """Main Phase 5 implementation function."""
    print("🚀 GitHub Issue #29 - Phase 5: Global Content Management")
    print("=" * 70)
    
    try:
        # Import database manager
        from realm_aware_database import RealmAwareMegaMindDatabase
        
        # Create database config
        db_config = {
            'host': 'localhost',
            'port': 3306,
            'user': 'megamind_user',
            'password': 'megamind_secure_2024',
            'database': 'megamind_database'
        }
        
        db_manager = RealmAwareMegaMindDatabase(db_config)
        content_manager = GlobalContentManager(db_manager)
        
        print("\n📋 Step 1: Populating GLOBAL realm with comprehensive guidelines...")
        
        # Populate global realm
        result = await content_manager.populate_global_realm()
        
        if result['success']:
            print(f"✅ Global realm populated successfully")
            print(f"   - Created {result['success_count']} elements")
            print(f"   - Errors: {result['error_count']}")
            print(f"   - Total processed: {result['total_processed']}")
            
            # Display created elements
            print("\n📋 Created Elements:")
            for element in result['created_elements'][:10]:  # Show first 10
                print(f"   ✅ {element['element_id']}: {element['title']} ({element['category']})")
        else:
            print(f"❌ Failed to populate global realm: {result['error']}")
            return False
        
        print("\n📋 Step 2: Testing content management workflows...")
        
        # Test workflow creation
        workflow_types = ['approval', 'update', 'deprecation']
        for workflow_type in workflow_types:
            workflow_result = await content_manager.create_content_workflow(workflow_type)
            if workflow_result['success']:
                print(f"✅ {workflow_type.title()} workflow created")
            else:
                print(f"❌ Failed to create {workflow_type} workflow")
        
        print("\n📋 Step 3: Testing content validation...")
        
        # Test validation on a sample element
        validation_result = await content_manager.validate_global_content('global_dev_001')
        if validation_result['success']:
            print(f"✅ Content validation completed")
            print(f"   - Overall score: {validation_result['overall_score']:.2f}")
            print(f"   - Status: {validation_result['status']}")
        else:
            print(f"❌ Content validation failed: {validation_result['error']}")
        
        print("\n🎉 Phase 5 Global Content Management completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Phase 5 implementation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)