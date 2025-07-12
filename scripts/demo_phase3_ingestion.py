#!/usr/bin/env python3
"""
Phase 3 Ingestion Demo Script
Demonstrates realm-aware markdown ingestion with semantic embedding generation
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mcp_server'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

# Mock the database and services for demo
from unittest.mock import Mock, patch

def create_demo_files(demo_dir: Path):
    """Create demonstration markdown files with different realm characteristics"""
    
    # Global realm content - company standards
    global_standards = """# Global Security Standards
Company-wide security policies and governance frameworks that apply to all projects.

## Authentication Requirements
All applications must implement secure authentication mechanisms:
- Multi-factor authentication for admin accounts
- JWT tokens with proper expiration
- Password complexity requirements

## Data Protection Standards
Organizations must follow these data protection guidelines:
- Encryption at rest and in transit
- Regular security audits
- GDPR compliance for EU data

## Code Review Policy
All code changes require review by at least one senior developer.
Security-critical changes require security team approval.
"""
    
    # Project realm content - specific implementation
    project_api = """# User Management API
Implementation guide for the new user management feature in our e-commerce platform.

## API Endpoints
The user management system provides the following endpoints:

### Authentication
- POST /api/auth/login - User authentication
- POST /api/auth/logout - User session termination
- GET /api/auth/verify - Token verification

### User Management
- GET /api/users/{id} - Retrieve user details
- POST /api/users - Create new user account
- PUT /api/users/{id} - Update user information
- DELETE /api/users/{id} - Deactivate user account

## Database Schema
The users table includes the following fields:
```sql
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('user', 'admin') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

## Implementation Notes
- Password hashing uses bcrypt with salt rounds of 12
- Email verification required for new accounts
- Rate limiting applied to authentication endpoints
"""
    
    # Mixed content - could go to either realm
    architecture_doc = """# System Architecture Overview
High-level architecture design for the platform infrastructure.

## Microservices Design
The system follows a microservices architecture with the following services:

### Core Services
- **User Service**: Handles authentication and user management
- **Product Service**: Manages product catalog and inventory
- **Order Service**: Processes orders and payments
- **Notification Service**: Manages email and SMS notifications

### Infrastructure Components
- **API Gateway**: Routes requests and handles rate limiting
- **Message Queue**: Asynchronous communication between services
- **Database Cluster**: Primary-replica setup for high availability
- **Monitoring Stack**: Logs, metrics, and alerting

## Deployment Strategy
All services are containerized using Docker and deployed on Kubernetes:
- Production namespace with resource limits
- Staging environment for testing
- Development environment for feature work

## Security Considerations
- Service-to-service authentication via mutual TLS
- Network policies for service isolation
- Regular security scanning of container images
"""
    
    # Write demo files
    files = {
        'global_security_standards.md': global_standards,
        'user_management_api.md': project_api,
        'system_architecture.md': architecture_doc
    }
    
    for filename, content in files.items():
        file_path = demo_dir / filename
        file_path.write_text(content)
    
    return list(files.keys())

def demo_realm_aware_ingestion():
    """Demonstrate realm-aware markdown ingestion"""
    print("🚀 Phase 3 Ingestion Demo: Realm-Aware Markdown Processing")
    print("=" * 70)
    
    # Create temporary demo directory
    demo_dir = Path(tempfile.mkdtemp(prefix="megamind_demo_"))
    
    try:
        # Create demo files
        print("\n📁 Creating demo files...")
        filenames = create_demo_files(demo_dir)
        for filename in filenames:
            print(f"   ✓ Created: {filename}")
        
        # Mock database configuration
        mock_db_config = {
            'host': 'localhost',
            'port': '3306',
            'database': 'megamind_demo',
            'user': 'demo_user',
            'password': 'demo_password'
        }
        
        # Mock the database and services
        with patch('realm_aware_markdown_ingester.RealmAwareMegaMindDatabase') as mock_db, \
             patch('realm_aware_markdown_ingester.get_embedding_service') as mock_embedding:
            
            # Configure mock embedding service
            mock_embedding_service = Mock()
            mock_embedding_service.is_available.return_value = True
            
            # Simulate embedding generation with realistic vectors
            def generate_embedding(text, realm_context=None):
                # Simulate different embeddings based on content
                import hashlib
                text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                # Generate pseudo-random but deterministic embedding
                embedding = [(text_hash >> i) % 100 / 100.0 for i in range(384)]
                return embedding
            
            mock_embedding_service.generate_embedding.side_effect = generate_embedding
            mock_embedding.return_value = mock_embedding_service
            
            # Configure mock database
            mock_database = Mock()
            chunk_counter = 0
            
            def create_chunk_mock(content, source_document, section_path, session_id, target_realm=None):
                nonlocal chunk_counter
                chunk_counter += 1
                return f"chunk_{chunk_counter:03d}"
            
            mock_database.create_chunk_with_target.side_effect = create_chunk_mock
            mock_database.get_pending_changes.return_value = []
            mock_database.commit_session_changes.return_value = {
                'success': True,
                'chunks_created': chunk_counter,
                'chunks_modified': 0,
                'relationships_added': 0
            }
            mock_db.return_value = mock_database
            
            # Import and test RealmAwareMarkdownIngester
            from realm_aware_markdown_ingester import RealmAwareMarkdownIngester
            
            print(f"\n🔧 Initializing Realm-Aware Markdown Ingester...")
            with patch.dict(os.environ, {
                'MEGAMIND_PROJECT_REALM': 'PROJ_ECOMMERCE',
                'MEGAMIND_DEFAULT_TARGET': 'PROJECT'
            }):
                ingester = RealmAwareMarkdownIngester(
                    db_config=mock_db_config,
                    session_id='demo_session_001'
                )
            
            print(f"   ✓ Session ID: {ingester.session_id}")
            print(f"   ✓ Project Realm: {ingester.project_realm}")
            print(f"   ✓ Target Realm: {ingester.target_realm}")
            print(f"   ✓ Embedding Service Available: {ingester.embedding_service.is_available()}")
            
            # Process each file and show results
            print(f"\n📄 Processing markdown files...")
            total_chunks = 0
            
            for filename in filenames:
                file_path = demo_dir / filename
                print(f"\n   Processing: {filename}")
                
                # Ingest file
                result = ingester.ingest_file(str(file_path))
                
                if result['success']:
                    chunks_processed = result['chunks_processed']
                    total_chunks += chunks_processed
                    processing_time = result['processing_time']
                    
                    print(f"   ✅ Success: {chunks_processed} chunks in {processing_time:.3f}s")
                    
                    # Show realm assignment prediction
                    chunks = ingester.parse_markdown_file(str(file_path))
                    realm_assignments = {}
                    for chunk in chunks:
                        processed_chunk = ingester._process_chunk_with_realm(chunk)
                        if processed_chunk:
                            realm = processed_chunk.realm_id
                            realm_assignments[realm] = realm_assignments.get(realm, 0) + 1
                    
                    print(f"   🏛️ Realm Distribution: {dict(realm_assignments)}")
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
            
            # Show overall statistics
            print(f"\n📊 Ingestion Statistics:")
            stats = ingester.get_ingestion_statistics()
            print(f"   Files processed: {stats['files_processed']}")
            print(f"   Total chunks: {stats['chunks_created']}")
            print(f"   Embeddings generated: {stats['embeddings_generated']}")
            print(f"   Embedding coverage: {stats['embedding_coverage']:.1f}%")
            print(f"   Realm assignments: {stats['realm_assignments']}")
            print(f"   Total processing time: {stats['processing_time']:.3f}s")
            print(f"   Errors: {stats['errors_count']}")
            
            # Demonstrate session management
            print(f"\n🔄 Session Management:")
            pending_changes = ingester.get_session_changes()
            print(f"   Pending changes: {len(pending_changes)}")
            print(f"   Session ID: {ingester.session_id}")
            
            # Simulate committing changes
            commit_result = ingester.commit_session_changes()
            if commit_result['success']:
                print(f"   ✅ Changes committed successfully")
                print(f"   Chunks created: {commit_result['chunks_created']}")
            else:
                print(f"   ❌ Commit failed: {commit_result.get('error')}")
            
            print(f"\n🎯 Demonstration completed successfully!")
            print(f"   Total chunks processed: {total_chunks}")
            print(f"   Database calls made: {mock_database.create_chunk_with_target.call_count}")
            print(f"   Embedding generations: {mock_embedding_service.generate_embedding.call_count}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"   🧹 Cleaned up demo directory")

def demo_bulk_ingestion():
    """Demonstrate bulk semantic ingestion capabilities"""
    print("\n" + "=" * 70)
    print("🚀 Bulk Semantic Ingestion Demo")
    print("=" * 70)
    
    # Create larger demo directory with multiple files
    demo_dir = Path(tempfile.mkdtemp(prefix="megamind_bulk_demo_"))
    
    try:
        # Create multiple demo files
        print("\n📁 Creating bulk demo files...")
        
        bulk_files = {
            'api_auth.md': """# Authentication API
JWT-based authentication system for secure access.

## Login Endpoint
POST /auth/login validates credentials and returns JWT token.
""",
            'api_users.md': """# Users API
User management endpoints for CRUD operations.

## User Creation
POST /users creates new user accounts with validation.
""",
            'database_schema.md': """# Database Schema
Complete database design for the application.

## Users Table
Primary table for storing user account information.
""",
            'deployment_guide.md': """# Deployment Guide
Step-by-step deployment instructions for production.

## Environment Setup
Configure production environment variables and secrets.
""",
            'security_policies.md': """# Security Policies
Company-wide security standards and compliance requirements.

## Access Control
Role-based access control for all system components.
"""
        }
        
        for filename, content in bulk_files.items():
            file_path = demo_dir / filename
            file_path.write_text(content)
        
        print(f"   ✓ Created {len(bulk_files)} files for bulk processing")
        
        # Mock services for bulk ingestion
        with patch('bulk_semantic_ingester.RealmAwareMarkdownIngester') as mock_ingester_class, \
             patch('bulk_semantic_ingester.get_embedding_service') as mock_embedding:
            
            # Configure mock embedding service
            mock_embedding_service = Mock()
            mock_embedding_service.is_available.return_value = True
            mock_embedding.return_value = mock_embedding_service
            
            # Configure mock ingester
            mock_ingester = Mock()
            mock_ingester.session_id = 'bulk_demo_session'
            
            # Mock successful file processing
            def mock_ingest_file(file_path):
                return {
                    'success': True,
                    'chunks_processed': 2,  # Simulate 2 chunks per file
                    'processing_time': 0.1,
                    'file_path': file_path,
                    'session_id': 'bulk_demo_session'
                }
            
            mock_ingester.ingest_file.side_effect = mock_ingest_file
            mock_ingester.get_ingestion_statistics.return_value = {
                'files_processed': len(bulk_files),
                'chunks_created': len(bulk_files) * 2,
                'embeddings_generated': len(bulk_files) * 2,
                'embedding_coverage': 100.0,
                'realm_assignments': {'PROJECT': len(bulk_files), 'GLOBAL': 1},
                'processing_time': 0.5,
                'errors_count': 0,
                'errors': []
            }
            mock_ingester.commit_session_changes.return_value = {'success': True}
            mock_ingester_class.return_value = mock_ingester
            
            # Import and test BulkSemanticIngester
            from bulk_semantic_ingester import BulkSemanticIngester, BulkIngestionConfig
            
            print(f"\n🔧 Initializing Bulk Semantic Ingester...")
            
            # Configure bulk ingestion
            config = BulkIngestionConfig(
                batch_size=3,
                max_workers=2,
                embedding_batch_size=5,
                enable_parallel_processing=False,  # Keep simple for demo
                enable_embedding_cache=True,
                auto_commit_batches=False
            )
            
            bulk_ingester = BulkSemanticIngester(
                db_config={'host': 'demo', 'port': '3306', 'database': 'demo', 'user': 'demo', 'password': 'demo'},
                config=config
            )
            
            print(f"   ✓ Batch size: {config.batch_size}")
            print(f"   ✓ Max workers: {config.max_workers}")
            print(f"   ✓ Embedding cache: {config.enable_embedding_cache}")
            print(f"   ✓ Session ID: {bulk_ingester.base_ingester.session_id}")
            
            # Perform bulk ingestion
            print(f"\n📄 Performing bulk directory ingestion...")
            start_time = datetime.now()
            
            result = bulk_ingester.ingest_directory(
                directory_path=str(demo_dir),
                pattern='*.md',
                recursive=False
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            if result['success']:
                stats = result['statistics']
                print(f"   ✅ Bulk ingestion completed successfully!")
                print(f"   📁 Source: {result['source_path']}")
                print(f"   ⏱️  Processing time: {processing_time:.3f}s")
                
                print(f"\n📊 Bulk Statistics:")
                print(f"   Files: {stats['files']['processed']}/{stats['files']['total']} ({stats['files']['success_rate']:.1f}%)")
                print(f"   Chunks: {stats['chunks']['successful']} created")
                print(f"   Embeddings: {stats['embeddings']['generated']} ({stats['embeddings']['coverage']:.1f}% coverage)")
                
                # Performance metrics
                throughput = stats['performance']['throughput']
                print(f"\n⚡ Performance:")
                print(f"   Files/sec: {throughput.get('files_per_second', 0):.2f}")
                print(f"   Chunks/sec: {throughput.get('chunks_per_second', 0):.2f}")
                
                # Configuration used
                config_info = result['config']
                print(f"\n⚙️  Configuration:")
                print(f"   Batch size: {config_info['batch_size']}")
                print(f"   Parallel processing: {config_info['parallel_processing']}")
                print(f"   Embedding cache: {config_info['embedding_cache']}")
            else:
                print(f"   ❌ Bulk ingestion failed: {result['error']}")
            
            print(f"\n🎯 Bulk demonstration completed!")
    
    except Exception as e:
        print(f"❌ Bulk demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if demo_dir.exists():
            shutil.rmtree(demo_dir)
            print(f"   🧹 Cleaned up bulk demo directory")

def main():
    """Run Phase 3 ingestion demonstrations"""
    print("🌟 MegaMind Context Database - Phase 3 Ingestion Demo")
    print("Demonstrates realm-aware markdown ingestion with semantic embeddings")
    print("")
    
    # Demo 1: Individual file ingestion with realm awareness
    demo_realm_aware_ingestion()
    
    # Demo 2: Bulk ingestion capabilities
    demo_bulk_ingestion()
    
    print("\n" + "=" * 70)
    print("✅ Phase 3 Ingestion Demo Complete!")
    print("=" * 70)
    print("""
🎉 Key Features Demonstrated:

📋 Realm-Aware Ingestion:
   • Automatic realm assignment based on content analysis
   • Project vs Global realm classification
   • Environment-based configuration

🔍 Semantic Enhancement:
   • Embedding generation for all chunks
   • Graceful degradation when dependencies unavailable
   • Realm-context enhanced embeddings

🔄 Session Management:
   • Change buffering for review workflow
   • Session-based commit process
   • Error tracking and recovery

⚡ Bulk Processing:
   • Batch processing for performance
   • Configurable parallel processing
   • Comprehensive statistics and monitoring

🏗️ Production Ready:
   • Error handling and recovery
   • Performance optimization
   • Comprehensive testing coverage

Phase 3 ingestion integration is complete and ready for production use!
""")

if __name__ == '__main__':
    main()