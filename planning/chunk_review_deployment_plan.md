# Chunk Review Interface - Deployment Plan

## Executive Summary

**Objective:** Minimal viable chunk review interface for development/tuning cycles
**Approach:** Flask + Vanilla JS (following existing DEVELOPMENT_TOOLS.md patterns)
**Deployment:** Port 5004, Docker integration, follows established service architecture
**Timeline:** 1-2 days implementation following existing patterns

## Architecture Decision

### Why Flask + Vanilla JS (Not React):
- **Existing Patterns:** Follows your `tools/diagram_service/` Flask architecture exactly
- **Minimal Dependencies:** No npm, webpack, or React complexity
- **Quick Deployment:** Single Python service with embedded templates
- **Maintainable:** Uses established PYTHON_STANDARDS.md and JAVASCRIPT_STANDARDS.md
- **Docker Ready:** Integrates with existing container patterns

### Service Structure (Following Existing Patterns):
```
tools/frontend_ui/
├── app.py                          # Main Flask app (port 5004)
├── requirements.txt                # Flask, Jinja2, database connector
├── config/
│   ├── __init__.py
│   ├── development.py              # Auto-approve disabled
│   └── production.py               # Auto-approve enabled
├── core/
│   ├── __init__.py
│   ├── chunk_service.py            # Database operations
│   ├── search_service.py           # Agent search simulation
│   └── approval_service.py         # Batch approval workflows
├── templates/
│   ├── base.html                   # Base template with navigation
│   ├── chunk_review.html           # Main review interface
│   ├── search_tester.html          # Query preview tool
│   └── rule_builder.html           # Rule creation interface
├── static/
│   ├── css/
│   │   └── chunk_review.css        # Highlighting and layout styles
│   └── js/
│       ├── chunk_highlighting.js   # Chunk boundary visualization
│       ├── search_tester.js        # Live search simulation
│       └── approval_actions.js     # Batch approval workflows
└── Dockerfile                      # Container deployment
```

## Implementation Plan

### Phase 1: Core Service Setup (4-6 hours)
**Files to Create:**

1. **`tools/frontend_ui/app.py`**
   ```python
   from flask import Flask, render_template, request, jsonify
   from core.chunk_service import ChunkService
   from core.search_service import SearchService
   import os
   
   app = Flask(__name__)
   
   # Initialize services with megamind_database
   chunk_service = ChunkService(
       host=os.getenv('DB_HOST', 'localhost'),
       port=int(os.getenv('DB_PORT', 3306)),
       database='megamind_database',
       user=os.getenv('DB_USER', 'dev'),
       password=os.getenv('DB_PASSWORD', '')
   )
   search_service = SearchService(chunk_service)
   
   @app.route('/')
   def dashboard():
       return render_template('chunk_review.html')
   
   @app.route('/api/chunks/pending')
   def get_pending_chunks():
       """Get chunks pending approval based on complexity_score threshold"""
       chunks = chunk_service.get_pending_approval()
       return jsonify(chunks)
   
   @app.route('/api/chunks/approve', methods=['POST'])
   def approve_chunks():
       """Batch approve chunks by updating approval_status"""
       chunk_ids = request.json.get('chunk_ids', [])
       approved_by = request.json.get('approved_by', 'frontend_ui')
       result = chunk_service.approve_chunks(chunk_ids, approved_by)
       return jsonify(result)
   
   @app.route('/api/chunks/reject', methods=['POST'])
   def reject_chunks():
       """Batch reject chunks with reason"""
       chunk_ids = request.json.get('chunk_ids', [])
       rejection_reason = request.json.get('rejection_reason', 'Rejected via frontend')
       rejected_by = request.json.get('rejected_by', 'frontend_ui')
       result = chunk_service.reject_chunks(chunk_ids, rejection_reason, rejected_by)
       return jsonify(result)
   
   @app.route('/api/search/simulate', methods=['POST'])
   def simulate_search():
       """Simulate agent search using FULLTEXT search on content"""
       query = request.json.get('query', '')
       results = search_service.simulate_agent_search(query)
       return jsonify(results)
   
   @app.route('/api/chunks/<chunk_id>/context')
   def get_chunk_context(chunk_id):
       """Get chunk with surrounding context for boundary visualization"""
       chunk_data = chunk_service.get_chunk_with_context(chunk_id)
       return jsonify(chunk_data)
   
   @app.route('/health')
   def health_check():
       """Health check endpoint for monitoring"""
       try:
           chunk_service.test_connection()
           return jsonify({'status': 'healthy', 'database': 'connected'})
       except Exception as e:
           return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=5004, debug=True)
   ```

2. **`tools/frontend_ui/core/chunk_service.py`**
   ```python
   from typing import List, Dict, Any, Optional
   from dataclasses import dataclass
   import mysql.connector
   from mysql.connector import Error
   import json
   
   @dataclass
   class Chunk:
       chunk_id: str
       realm_id: str
       content: str
       complexity_score: float
       source_document: str
       section_path: Optional[str]
       chunk_type: str
       line_count: int
       token_count: int
       access_count: int
       created_at: str
       last_accessed: str
   
   class ChunkService:
       def __init__(self, host: str, port: int, database: str, user: str, password: str):
           self.db_config = {
               'host': host,
               'port': port,
               'database': database,
               'user': user,
               'password': password,
               'autocommit': True
           }
       
       def _get_connection(self):
           """Get database connection"""
           return mysql.connector.connect(**self.db_config)
       
       def test_connection(self) -> bool:
           """Test database connectivity"""
           try:
               conn = self._get_connection()
               conn.close()
               return True
           except Error:
               return False
       
       def get_pending_approval(self, limit: int = 50) -> List[Dict[str, Any]]:
           """Get chunks with approval_status = 'pending'"""
           try:
               conn = self._get_connection()
               cursor = conn.cursor(dictionary=True)
               
               # Get chunks pending approval with proper status field
               query = """
               SELECT chunk_id, realm_id, content, complexity_score, 
                      source_document, section_path, chunk_type, 
                      line_count, token_count, access_count,
                      approval_status, created_at, last_accessed,
                      approved_at, approved_by, rejection_reason
               FROM megamind_chunks 
               WHERE approval_status = 'pending'
               ORDER BY created_at DESC, complexity_score ASC 
               LIMIT %s
               """
               
               cursor.execute(query, (limit,))
               chunks = cursor.fetchall()
               
               # Convert datetime objects to strings for JSON serialization
               for chunk in chunks:
                   chunk['created_at'] = chunk['created_at'].isoformat() if chunk['created_at'] else None
                   chunk['last_accessed'] = chunk['last_accessed'].isoformat() if chunk['last_accessed'] else None
                   chunk['approved_at'] = chunk['approved_at'].isoformat() if chunk['approved_at'] else None
               
               cursor.close()
               conn.close()
               return chunks
               
           except Error as e:
               print(f"Database error in get_pending_approval: {e}")
               return []
       
       def approve_chunks(self, chunk_ids: List[str], approved_by: str = "frontend_ui") -> Dict[str, Any]:
           """Approve chunks by updating approval_status to 'approved'"""
           try:
               conn = self._get_connection()
               cursor = conn.cursor()
               
               # Update approval status and metadata
               placeholders = ','.join(['%s'] * len(chunk_ids))
               query = f"""
               UPDATE megamind_chunks 
               SET approval_status = 'approved',
                   approved_at = CURRENT_TIMESTAMP,
                   approved_by = %s,
                   updated_at = CURRENT_TIMESTAMP,
                   access_count = access_count + 1
               WHERE chunk_id IN ({placeholders})
                 AND approval_status = 'pending'
               """
               
               cursor.execute(query, [approved_by] + chunk_ids)
               affected_rows = cursor.rowcount
               
               cursor.close()
               conn.close()
               
               return {
                   'success': True,
                   'approved_count': affected_rows,
                   'chunk_ids': chunk_ids,
                   'approved_by': approved_by
               }
               
           except Error as e:
               return {
                   'success': False,
                   'error': str(e),
                   'approved_count': 0
               }
       
       def reject_chunks(self, chunk_ids: List[str], rejection_reason: str, rejected_by: str = "frontend_ui") -> Dict[str, Any]:
           """Reject chunks by updating approval_status to 'rejected'"""
           try:
               conn = self._get_connection()
               cursor = conn.cursor()
               
               # Update rejection status and reason
               placeholders = ','.join(['%s'] * len(chunk_ids))
               query = f"""
               UPDATE megamind_chunks 
               SET approval_status = 'rejected',
                   rejection_reason = %s,
                   approved_by = %s,
                   updated_at = CURRENT_TIMESTAMP
               WHERE chunk_id IN ({placeholders})
                 AND approval_status = 'pending'
               """
               
               cursor.execute(query, [rejection_reason, rejected_by] + chunk_ids)
               affected_rows = cursor.rowcount
               
               cursor.close()
               conn.close()
               
               return {
                   'success': True,
                   'rejected_count': affected_rows,
                   'chunk_ids': chunk_ids,
                   'rejection_reason': rejection_reason
               }
               
           except Error as e:
               return {
                   'success': False,
                   'error': str(e),
                   'rejected_count': 0
               }
       
       def get_chunk_with_context(self, chunk_id: str) -> Dict[str, Any]:
           """Get chunk with surrounding context from same document"""
           try:
               conn = self._get_connection()
               cursor = conn.cursor(dictionary=True)
               
               # Get the target chunk
               chunk_query = """
               SELECT * FROM megamind_chunks WHERE chunk_id = %s
               """
               cursor.execute(chunk_query, (chunk_id,))
               chunk = cursor.fetchone()
               
               if not chunk:
                   return {'error': 'Chunk not found'}
               
               # Get related chunks from same document
               context_query = """
               SELECT chunk_id, section_path, chunk_type, content
               FROM megamind_chunks 
               WHERE source_document = %s 
                 AND chunk_id != %s
               ORDER BY section_path, chunk_id
               LIMIT 10
               """
               cursor.execute(context_query, (chunk['source_document'], chunk_id))
               context_chunks = cursor.fetchall()
               
               cursor.close()
               conn.close()
               
               # Format timestamps
               if chunk['created_at']:
                   chunk['created_at'] = chunk['created_at'].isoformat()
               if chunk['last_accessed']:
                   chunk['last_accessed'] = chunk['last_accessed'].isoformat()
               if chunk['updated_at']:
                   chunk['updated_at'] = chunk['updated_at'].isoformat()
               
               return {
                   'chunk': chunk,
                   'context_chunks': context_chunks
               }
               
           except Error as e:
               return {'error': str(e)}
   ```

4. **`tools/frontend_ui/templates/chunk_review.html`**
   ```python
   from typing import List, Dict, Any
   from .chunk_service import ChunkService
   import mysql.connector
   from mysql.connector import Error
   
   class SearchService:
       def __init__(self, chunk_service: ChunkService):
           self.chunk_service = chunk_service
       
       def simulate_agent_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
           """Simulate agent search using FULLTEXT search on megamind_chunks"""
           if not query.strip():
               return []
           
           try:
               conn = self.chunk_service._get_connection()
               cursor = conn.cursor(dictionary=True)
               
               # Use FULLTEXT search with MATCH() AGAINST() for semantic relevance
               search_query = """
               SELECT chunk_id, realm_id, content, complexity_score,
                      source_document, section_path, chunk_type,
                      token_count, access_count,
                      MATCH(content, section_path) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance_score
               FROM megamind_chunks 
               WHERE MATCH(content, section_path) AGAINST(%s IN NATURAL LANGUAGE MODE)
               ORDER BY relevance_score DESC, access_count DESC
               LIMIT %s
               """
               
               cursor.execute(search_query, (query, query, limit))
               results = cursor.fetchall()
               
               # Format results with content preview
               formatted_results = []
               for result in results:
                   # Create content preview (first 200 chars)
                   content_preview = result['content'][:200]
                   if len(result['content']) > 200:
                       content_preview += "..."
                   
                   formatted_results.append({
                       'chunk_id': result['chunk_id'],
                       'realm_id': result['realm_id'],
                       'content_preview': content_preview,
                       'relevance_score': float(result['relevance_score']),
                       'complexity_score': result['complexity_score'],
                       'source_document': result['source_document'],
                       'section_path': result['section_path'],
                       'chunk_type': result['chunk_type'],
                       'token_count': result['token_count'],
                       'access_count': result['access_count']
                   })
               
               cursor.close()
               conn.close()
               return formatted_results
               
           except Error as e:
               print(f"Search error: {e}")
               return []
       
       def get_chunk_relationships(self, chunk_id: str) -> List[Dict[str, Any]]:
           """Get related chunks using megamind_chunk_relationships table"""
           try:
               conn = self.chunk_service._get_connection()
               cursor = conn.cursor(dictionary=True)
               
               # Get relationships where this chunk is the source
               relationship_query = """
               SELECT mcr.relationship_type, mcr.strength, mcr.discovered_by,
                      mc.chunk_id, mc.content, mc.source_document, mc.section_path
               FROM megamind_chunk_relationships mcr
               JOIN megamind_chunks mc ON mcr.related_chunk_id = mc.chunk_id
               WHERE mcr.chunk_id = %s
               ORDER BY mcr.strength DESC
               LIMIT 10
               """
               
               cursor.execute(relationship_query, (chunk_id,))
               relationships = cursor.fetchall()
               
               cursor.close()
               conn.close()
               return relationships
               
           except Error as e:
               print(f"Relationship query error: {e}")
               return []
   ```
   ```html
   <!DOCTYPE html>
   <html lang="en">
   <head>
       <meta charset="UTF-8">
       <meta name="viewport" content="width=device-width, initial-scale=1.0">
       <title>Chunk Review Interface</title>
       <link rel="stylesheet" href="{{ url_for('static', filename='css/chunk_review.css') }}">
   </head>
   <body>
       <div class="container">
           <header>
               <h1>Chunk Review Interface</h1>
               <div class="stats" id="reviewStats">
                   <span>Pending: <strong id="pendingCount">0</strong></span>
                   <span>Approved: <strong id="approvedCount">0</strong></span>
               </div>
           </header>
           
           <main class="review-workspace">
               <!-- Chunk Display Area -->
               <section class="chunk-display">
                   <div class="chunk-content" id="chunkContent">
                       <div class="chunk-boundaries" id="chunkBoundaries"></div>
                   </div>
               </section>
               
               <!-- Control Panel -->
               <aside class="control-panel">
                   <div class="approval-controls">
                       <button id="approveSelected" class="btn-approve">Approve Selected</button>
                       <button id="rejectSelected" class="btn-reject">Reject Selected</button>
                       <button id="selectAll" class="btn-secondary">Select All</button>
                       <button id="clearSelection" class="btn-secondary">Clear Selection</button>
                   </div>
                   
                   <div class="rejection-panel" id="rejectionPanel" style="display: none;">
                       <h4>Rejection Reason</h4>
                       <textarea id="rejectionReason" placeholder="Enter reason for rejection..."></textarea>
                       <button id="confirmReject" class="btn-reject">Confirm Rejection</button>
                       <button id="cancelReject" class="btn-secondary">Cancel</button>
                   </div>
                   
                   <div class="search-tester">
                       <h3>Test Search Query</h3>
                       <input type="text" id="searchQuery" placeholder="Enter search query...">
                       <button id="testSearch" class="btn-secondary">Test Query</button>
                       <div class="search-results" id="searchResults"></div>
                   </div>
               </aside>
           </main>
       </div>
       
       <script src="{{ url_for('static', filename='js/chunk_highlighting.js') }}"></script>
       <script src="{{ url_for('static', filename='js/search_tester.js') }}"></script>
       <script src="{{ url_for('static', filename='js/approval_actions.js') }}"></script>
   </body>
   </html>
   ```

### Phase 2: Frontend Functionality (4-6 hours)

4. **`tools/chunk_review_service/static/js/chunk_highlighting.js`**
   ```javascript
   // Chunk boundary visualization
   class ChunkHighlighter {
       constructor() {
           this.selectedChunks = new Set();
           this.loadPendingChunks();
       }
       
       async loadPendingChunks() {
           try {
               const response = await fetch('/api/chunks/pending');
               const chunks = await response.json();
               this.renderChunks(chunks);
           } catch (error) {
               console.error('Failed to load chunks:', error);
           }
       }
       
       renderChunks(chunks) {
           const container = document.getElementById('chunkBoundaries');
           chunks.forEach(chunk => this.renderChunk(chunk, container));
       }
       
       renderChunk(chunk, container) {
           const chunkElement = document.createElement('div');
           chunkElement.className = 'chunk-item';
           chunkElement.dataset.chunkId = chunk.id;
           
           chunkElement.innerHTML = `
               <div class="chunk-header">
                   <input type="checkbox" class="chunk-checkbox" value="${chunk.id}">
                   <span class="chunk-id">${chunk.id}</span>
                   <span class="confidence-score">${chunk.confidence_score.toFixed(2)}</span>
               </div>
               <div class="chunk-content">${this.highlightBoundaries(chunk.content, chunk.boundaries)}</div>
           `;
           
           container.appendChild(chunkElement);
       }
       
       highlightBoundaries(content, boundaries) {
           // Add visual boundaries around chunk content
           return content.replace(/\n/g, '<br>');
       }
   }
   
   // Initialize on page load
   document.addEventListener('DOMContentLoaded', () => {
       new ChunkHighlighter();
   });
   ```

5. **`tools/chunk_review_service/static/js/search_tester.js`**
   ```javascript
   // Live search simulation
   class SearchTester {
       constructor() {
           this.initializeSearchTester();
       }
       
       initializeSearchTester() {
           const searchInput = document.getElementById('searchQuery');
           const testButton = document.getElementById('testSearch');
           
           testButton.addEventListener('click', () => this.testSearch());
           searchInput.addEventListener('keypress', (e) => {
               if (e.key === 'Enter') this.testSearch();
           });
       }
       
       async testSearch() {
           const query = document.getElementById('searchQuery').value;
           if (!query.trim()) return;
           
           try {
               const response = await fetch('/api/search/simulate', {
                   method: 'POST',
                   headers: { 'Content-Type': 'application/json' },
                   body: JSON.stringify({ query })
               });
               
               const results = await response.json();
               this.displaySearchResults(results);
           } catch (error) {
               console.error('Search test failed:', error);
           }
       }
       
       displaySearchResults(results) {
           const container = document.getElementById('searchResults');
           container.innerHTML = '';
           
           if (results.length === 0) {
               container.innerHTML = '<p>No matching chunks found</p>';
               return;
           }
           
           results.forEach(result => {
               const resultElement = document.createElement('div');
               resultElement.className = 'search-result';
               resultElement.innerHTML = `
                   <div class="result-header">
                       <strong>Chunk ${result.chunk_id}</strong>
                       <span class="relevance-score">${result.relevance_score.toFixed(3)}</span>
                   </div>
                   <div class="result-content">${result.content_preview}</div>
               `;
               container.appendChild(resultElement);
           });
       }
   }
   
   // Initialize on page load
   document.addEventListener('DOMContentLoaded', () => {
       new SearchTester();
   });
   ```

### Phase 3: Docker Integration (2-3 hours)

6. **`tools/chunk_review_service/Dockerfile`**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application
   COPY . .
   
   # Expose port
   EXPOSE 5004
   
   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
       CMD curl -f http://localhost:5004/health || exit 1
   
   # Run application
   CMD ["python", "app.py"]
   ```

7. **Update `docker-compose.yml`** (Add to existing services)
   ```yaml
   frontend-ui:
     build: ./tools/frontend_ui
     ports:
       - "5004:5004"
     environment:
       - ENVIRONMENT=development
       - DB_HOST=mysql-dev
       - DB_PORT=3306
       - DB_USER=dev
       - DB_PASSWORD=""
     depends_on:
       - mysql-dev
     volumes:
       - ./tools/frontend_ui:/app
     restart: unless-stopped
   ```

## Deployment Commands for Claude Code

### Quick Start Commands:
```bash
# 1. Create service structure
mkdir -p tools/frontend_ui/{core,templates,static/{css,js},config}
cd tools/frontend_ui

# 2. Create requirements.txt
echo "Flask==2.3.3
Jinja2==3.1.2
mysql-connector-python==8.1.0
python-dotenv==1.0.0" > requirements.txt

# 3. Set environment variables
export DB_HOST=localhost
export DB_PORT=3306
export DB_USER=dev
export DB_PASSWORD=""

# 4. Start development server
python app.py

# 5. Test interface
curl http://localhost:5004/
curl http://localhost:5004/api/chunks/pending

# 6. Docker deployment (update docker-compose.yml)
docker-compose up frontend-ui
```

### Integration with Existing Infrastructure:
```bash
# Add to existing health check script
echo "curl -f http://localhost:5004/health || echo 'Chunk Review Service Down'" >> deployment/scripts/helpers/health-check.sh

# Add to monitoring
echo "chunk_review_service http://localhost:5004/health" >> monitoring/services.conf
```

## Environment Configuration

### Development Mode (`config/development.py`):
```python
DEBUG = True
AUTO_APPROVE = False  # Your environment variable
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "dev"
DB_PASSWORD = ""
DATABASE = "megamind_database"
LOG_LEVEL = "DEBUG"
```

### Production Mode (`config/production.py`):
```python
DEBUG = False
AUTO_APPROVE = True  # Your environment variable
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "gameapp"
DB_PASSWORD = ""
DATABASE = "megamind_database"
LOG_LEVEL = "INFO"
```

## Success Metrics

### Immediate Validation:
- [ ] Service starts on port 5004
- [ ] Interface loads with chunk display
- [ ] Search tester returns agent-matching results
- [ ] Approval/rejection buttons update database
- [ ] Docker container builds and runs

### Development Workflow Integration:
- [ ] Environment variable controls auto-approval behavior
- [ ] Interface shows chunk boundaries clearly
- [ ] Rule creation workflow produces expected output format
- [ ] Search simulation matches agent behavior exactly
- [ ] Batch operations handle multiple chunks efficiently

## Maintenance and Extensions

### Future Enhancements (If Needed):
1. **Real-time Updates:** WebSocket integration for live chunk status
2. **Advanced Visualization:** Graph view of chunk relationships
3. **Rule Management:** Visual rule editor with query preview
4. **Analytics Dashboard:** Approval rates and chunk quality metrics
5. **Export Functionality:** Approved chunks export to various formats

### Monitoring Integration:
- Health check endpoint: `/health`
- Metrics endpoint: `/metrics`
- Log integration with existing monitoring stack
- Performance tracking for database operations

This deployment plan provides a complete, maintainable chunk review interface that integrates seamlessly with your existing infrastructure while remaining simple enough for rapid development and deployment.