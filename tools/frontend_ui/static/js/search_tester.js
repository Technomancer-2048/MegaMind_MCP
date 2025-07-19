/**
 * Live search tester with relevance scoring and agent behavior simulation
 * Provides real-time search functionality with performance metrics
 */

class SearchTester {
    constructor() {
        this.searchHistory = new Map();
        this.searchCache = new Map();
        this.searchStats = {
            totalSearches: 0,
            averageResponseTime: 0,
            cacheHits: 0
        };
        this.debounceTimer = null;
        this.currentSearch = null;
        
        this.initializeSearchTester();
    }
    
    initializeSearchTester() {
        this.bindEvents();
        this.setupSearchInterface();
        this.loadSearchHistory();
    }
    
    bindEvents() {
        const searchInput = document.getElementById('searchQuery');
        const testButton = document.getElementById('testSearch');
        
        if (searchInput) {
            // Real-time search with debouncing
            searchInput.addEventListener('input', (e) => {
                this.debouncedSearch(e.target.value);
            });
            
            // Enter key support
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.performSearch(e.target.value);
                }
            });
            
            // Search suggestions
            searchInput.addEventListener('focus', () => {
                this.showSearchSuggestions();
            });
        }
        
        if (testButton) {
            testButton.addEventListener('click', () => {
                const query = searchInput?.value || '';
                this.performSearch(query);
            });
        }
        
        // Advanced search controls
        this.bindAdvancedSearchControls();
    }
    
    bindAdvancedSearchControls() {
        // Search type selector
        const searchTypeSelect = document.getElementById('searchType');
        if (searchTypeSelect) {
            searchTypeSelect.addEventListener('change', (e) => {
                this.searchType = e.target.value;
                this.updateSearchInterface();
            });
        }
        
        // Realm selector
        const realmSelect = document.getElementById('realmFilter');
        if (realmSelect) {
            realmSelect.addEventListener('change', (e) => {
                this.currentRealm = e.target.value;
            });
        }
        
        // Live search toggle
        const liveSearchToggle = document.getElementById('liveSearchToggle');
        if (liveSearchToggle) {
            liveSearchToggle.addEventListener('change', (e) => {
                this.liveSearchEnabled = e.target.checked;
            });
        }
    }
    
    setupSearchInterface() {
        const searchContainer = document.getElementById('searchTester') || 
                              document.querySelector('.search-tester');
        
        if (!searchContainer) return;
        
        // Enhanced search interface
        const enhancedInterface = `
            <div class="search-interface">
                <div class="search-controls-advanced">
                    <div class="search-input-group">
                        <input type="text" id="searchQuery" placeholder="Enter search query..." 
                               class="form-control search-input" autocomplete="off">
                        <button id="testSearch" class="btn btn-info search-btn">
                            <span class="search-icon">🔍</span>
                            Search
                        </button>
                    </div>
                    
                    <div class="search-options">
                        <div class="search-option">
                            <label for="searchType">Search Type:</label>
                            <select id="searchType" class="form-control">
                                <option value="simulate">Agent Simulation</option>
                                <option value="fulltext">Full Text</option>
                                <option value="semantic">Semantic</option>
                                <option value="realm">Realm Specific</option>
                            </select>
                        </div>
                        
                        <div class="search-option">
                            <label for="realmFilter">Realm:</label>
                            <select id="realmFilter" class="form-control">
                                <option value="">All Realms</option>
                                <option value="PROJECT">Project</option>
                                <option value="GLOBAL">Global</option>
                            </select>
                        </div>
                        
                        <div class="search-option">
                            <label>
                                <input type="checkbox" id="liveSearchToggle" checked>
                                Live Search
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="search-stats" id="searchStats">
                    <div class="stat-item">
                        <span class="stat-label">Total Searches:</span>
                        <span class="stat-value" id="totalSearches">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Response:</span>
                        <span class="stat-value" id="avgResponse">0ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Cache Hits:</span>
                        <span class="stat-value" id="cacheHits">0</span>
                    </div>
                </div>
                
                <div class="search-suggestions" id="searchSuggestions" style="display: none;">
                    <h4>Search Suggestions</h4>
                    <div class="suggestions-list" id="suggestionsList"></div>
                </div>
                
                <div class="search-results" id="searchResults"></div>
                
                <div class="search-history" id="searchHistory">
                    <h4>Recent Searches</h4>
                    <div class="history-list" id="historyList"></div>
                </div>
            </div>
        `;
        
        searchContainer.innerHTML = enhancedInterface;
        
        // Initialize default values
        this.searchType = 'simulate';
        this.currentRealm = '';
        this.liveSearchEnabled = true;
        
        // Re-bind events after interface update
        this.bindEvents();
    }
    
    debouncedSearch(query) {
        if (!this.liveSearchEnabled || !query.trim()) return;
        
        // Clear previous timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        // Set new timer
        this.debounceTimer = setTimeout(() => {
            this.performSearch(query, true);
        }, 300); // 300ms delay
    }
    
    async performSearch(query, isLiveSearch = false) {
        if (!query.trim()) {
            this.displaySearchResults([]);
            return;
        }
        
        // Check cache first
        const cacheKey = this.getCacheKey(query);
        if (this.searchCache.has(cacheKey)) {
            this.searchStats.cacheHits++;
            this.displaySearchResults(this.searchCache.get(cacheKey), true);
            this.updateSearchStats();
            return;
        }
        
        // Show loading state
        this.showSearchLoading();
        
        const startTime = performance.now();
        
        try {
            // Cancel previous search if still running
            if (this.currentSearch) {
                this.currentSearch.abort();
            }
            
            // Create new search request
            this.currentSearch = new AbortController();
            
            const searchEndpoint = this.getSearchEndpoint();
            const searchBody = this.buildSearchBody(query);
            
            const response = await fetch(searchEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(searchBody),
                signal: this.currentSearch.signal
            });
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }
            
            const data = await response.json();
            const results = data.success ? data.results : [];
            
            // Calculate response time
            const responseTime = performance.now() - startTime;
            
            // Update stats
            this.searchStats.totalSearches++;
            this.searchStats.averageResponseTime = 
                (this.searchStats.averageResponseTime + responseTime) / 2;
            
            // Cache results
            this.searchCache.set(cacheKey, results);
            
            // Store in history
            this.addToSearchHistory(query, results.length, responseTime);
            
            // Display results
            this.displaySearchResults(results, false, responseTime);
            
            // Update UI
            this.updateSearchStats();
            this.updateSearchHistory();
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.log('Search aborted');
                return;
            }
            
            console.error('Search error:', error);
            this.displaySearchError(error.message);
        } finally {
            this.hideSearchLoading();
            this.currentSearch = null;
        }
    }
    
    getSearchEndpoint() {
        switch (this.searchType) {
            case 'simulate':
                return '/api/search/simulate';
            case 'fulltext':
                return '/api/search/simulate';
            case 'semantic':
                return '/api/search/simulate';
            case 'realm':
                return '/api/search/realm';
            default:
                return '/api/search/simulate';
        }
    }
    
    buildSearchBody(query) {
        const body = {
            query: query,
            limit: 10
        };
        
        if (this.searchType === 'realm' && this.currentRealm) {
            body.realm_id = this.currentRealm;
        }
        
        if (this.searchType === 'semantic') {
            body.search_type = 'semantic';
        }
        
        return body;
    }
    
    getCacheKey(query) {
        return `${this.searchType}:${this.currentRealm}:${query.toLowerCase()}`;
    }
    
    showSearchLoading() {
        const resultsContainer = document.getElementById('searchResults');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div class="search-loading">
                    <div class="loading-spinner"></div>
                    <p>Searching...</p>
                </div>
            `;
        }
    }
    
    hideSearchLoading() {
        const loadingElement = document.querySelector('.search-loading');
        if (loadingElement) {
            loadingElement.remove();
        }
    }
    
    displaySearchResults(results, fromCache = false, responseTime = 0) {
        const container = document.getElementById('searchResults');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (results.length === 0) {
            container.innerHTML = `
                <div class="no-results">
                    <p>No matching chunks found</p>
                    <div class="search-suggestions">
                        <p>Try:</p>
                        <ul>
                            <li>Using different keywords</li>
                            <li>Checking spelling</li>
                            <li>Using broader search terms</li>
                            <li>Changing search type</li>
                        </ul>
                    </div>
                </div>
            `;
            return;
        }
        
        // Results header
        const headerHtml = `
            <div class="results-header">
                <h4>Search Results (${results.length} found)</h4>
                <div class="results-meta">
                    ${fromCache ? '<span class="cache-indicator">From Cache</span>' : ''}
                    ${responseTime > 0 ? `<span class="response-time">${responseTime.toFixed(0)}ms</span>` : ''}
                </div>
            </div>
        `;
        
        container.innerHTML = headerHtml;
        
        // Results list
        const resultsList = document.createElement('div');
        resultsList.className = 'results-list';
        
        results.forEach((result, index) => {
            const resultElement = this.createResultElement(result, index);
            resultsList.appendChild(resultElement);
        });
        
        container.appendChild(resultsList);
        
        // Add result actions
        this.addResultActions(container, results);
    }
    
    createResultElement(result, index) {
        const resultElement = document.createElement('div');
        resultElement.className = 'search-result';
        resultElement.dataset.chunkId = result.chunk_id;
        
        // Calculate relevance color
        const relevanceColor = this.getRelevanceColor(result.relevance_score);
        
        resultElement.innerHTML = `
            <div class="result-header">
                <div class="result-info">
                    <strong class="result-title">Chunk ${result.chunk_id}</strong>
                    <span class="result-type">${result.chunk_type || 'unknown'}</span>
                </div>
                <div class="result-metrics">
                    <span class="relevance-score" style="background-color: ${relevanceColor}">
                        ${result.relevance_score.toFixed(3)}
                    </span>
                    <span class="result-position">#${index + 1}</span>
                </div>
            </div>
            
            <div class="result-meta">
                <span class="result-source">
                    <strong>Source:</strong> ${result.source_document}
                </span>
                <span class="result-section">
                    ${result.section_path ? `<strong>Section:</strong> ${result.section_path}` : ''}
                </span>
                <span class="result-status status-${result.approval_status}">
                    ${result.approval_status}
                </span>
            </div>
            
            <div class="result-content">
                ${this.highlightSearchTerms(result.content_preview, this.getLastSearchQuery())}
            </div>
            
            <div class="result-stats">
                <span>Lines: ${result.line_count || 0}</span>
                <span>Tokens: ${result.token_count || 0}</span>
                <span>Access: ${result.access_count || 0}</span>
            </div>
            
            <div class="result-actions">
                <button class="btn btn-sm btn-outline view-chunk" data-chunk-id="${result.chunk_id}">
                    View Details
                </button>
                <button class="btn btn-sm btn-outline copy-chunk-id" data-chunk-id="${result.chunk_id}">
                    Copy ID
                </button>
                <button class="btn btn-sm btn-outline highlight-chunk" data-chunk-id="${result.chunk_id}">
                    Highlight
                </button>
            </div>
        `;
        
        // Bind result actions
        this.bindResultActions(resultElement);
        
        return resultElement;
    }
    
    bindResultActions(resultElement) {
        const viewBtn = resultElement.querySelector('.view-chunk');
        const copyBtn = resultElement.querySelector('.copy-chunk-id');
        const highlightBtn = resultElement.querySelector('.highlight-chunk');
        
        if (viewBtn) {
            viewBtn.addEventListener('click', (e) => {
                const chunkId = e.target.dataset.chunkId;
                this.viewChunkDetails(chunkId);
            });
        }
        
        if (copyBtn) {
            copyBtn.addEventListener('click', (e) => {
                const chunkId = e.target.dataset.chunkId;
                this.copyChunkId(chunkId);
            });
        }
        
        if (highlightBtn) {
            highlightBtn.addEventListener('click', (e) => {
                const chunkId = e.target.dataset.chunkId;
                this.highlightChunkInList(chunkId);
            });
        }
    }
    
    getRelevanceColor(score) {
        // Color scale from red (low) to green (high)
        const red = Math.max(0, Math.min(255, 255 - (score * 255)));
        const green = Math.max(0, Math.min(255, score * 255));
        return `rgb(${red}, ${green}, 0)`;
    }
    
    highlightSearchTerms(content, query) {
        if (!content || !query) return content;
        
        const terms = query.split(/\s+/).filter(term => term.length > 2);
        let highlighted = content;
        
        terms.forEach(term => {
            const regex = new RegExp(`(${term})`, 'gi');
            highlighted = highlighted.replace(regex, '<mark class="search-highlight">$1</mark>');
        });
        
        return highlighted;
    }
    
    getLastSearchQuery() {
        const searchInput = document.getElementById('searchQuery');
        return searchInput ? searchInput.value : '';
    }
    
    addResultActions(container, results) {
        const actionsHtml = `
            <div class="results-actions">
                <button class="btn btn-sm btn-outline" id="selectAllResults">
                    Select All Results
                </button>
                <button class="btn btn-sm btn-outline" id="exportResults">
                    Export Results
                </button>
                <button class="btn btn-sm btn-outline" id="clearResults">
                    Clear Results
                </button>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', actionsHtml);
        
        // Bind actions
        document.getElementById('selectAllResults')?.addEventListener('click', () => {
            this.selectAllResults(results);
        });
        
        document.getElementById('exportResults')?.addEventListener('click', () => {
            this.exportResults(results);
        });
        
        document.getElementById('clearResults')?.addEventListener('click', () => {
            this.clearResults();
        });
    }
    
    displaySearchError(message) {
        const container = document.getElementById('searchResults');
        if (!container) return;
        
        container.innerHTML = `
            <div class="search-error">
                <h4>Search Error</h4>
                <p>${message}</p>
                <button class="btn btn-sm btn-outline" onclick="location.reload()">
                    Retry
                </button>
            </div>
        `;
    }
    
    // Search history management
    addToSearchHistory(query, resultCount, responseTime) {
        const historyEntry = {
            query: query,
            timestamp: new Date().toISOString(),
            resultCount: resultCount,
            responseTime: responseTime,
            searchType: this.searchType,
            realm: this.currentRealm
        };
        
        this.searchHistory.set(query, historyEntry);
        
        // Keep only last 10 searches
        if (this.searchHistory.size > 10) {
            const firstKey = this.searchHistory.keys().next().value;
            this.searchHistory.delete(firstKey);
        }
    }
    
    updateSearchHistory() {
        const historyContainer = document.getElementById('historyList');
        if (!historyContainer) return;
        
        historyContainer.innerHTML = '';
        
        const entries = Array.from(this.searchHistory.values()).reverse();
        
        entries.forEach(entry => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <div class="history-query" data-query="${entry.query}">
                    "${entry.query}"
                </div>
                <div class="history-meta">
                    <span>${entry.resultCount} results</span>
                    <span>${entry.responseTime.toFixed(0)}ms</span>
                    <span>${new Date(entry.timestamp).toLocaleTimeString()}</span>
                </div>
            `;
            
            // Make clickable
            historyItem.addEventListener('click', () => {
                document.getElementById('searchQuery').value = entry.query;
                this.performSearch(entry.query);
            });
            
            historyContainer.appendChild(historyItem);
        });
    }
    
    loadSearchHistory() {
        // Load from localStorage if available
        const stored = localStorage.getItem('searchHistory');
        if (stored) {
            try {
                const history = JSON.parse(stored);
                this.searchHistory = new Map(history);
            } catch (error) {
                console.error('Error loading search history:', error);
            }
        }
    }
    
    saveSearchHistory() {
        try {
            const history = Array.from(this.searchHistory.entries());
            localStorage.setItem('searchHistory', JSON.stringify(history));
        } catch (error) {
            console.error('Error saving search history:', error);
        }
    }
    
    // Search suggestions
    showSearchSuggestions() {
        const suggestionsContainer = document.getElementById('searchSuggestions');
        if (!suggestionsContainer) return;
        
        const suggestions = this.generateSearchSuggestions();
        
        if (suggestions.length === 0) {
            suggestionsContainer.style.display = 'none';
            return;
        }
        
        const suggestionsList = document.getElementById('suggestionsList');
        suggestionsList.innerHTML = '';
        
        suggestions.forEach(suggestion => {
            const suggestionItem = document.createElement('div');
            suggestionItem.className = 'suggestion-item';
            suggestionItem.textContent = suggestion;
            suggestionItem.addEventListener('click', () => {
                document.getElementById('searchQuery').value = suggestion;
                this.performSearch(suggestion);
                suggestionsContainer.style.display = 'none';
            });
            
            suggestionsList.appendChild(suggestionItem);
        });
        
        suggestionsContainer.style.display = 'block';
    }
    
    generateSearchSuggestions() {
        const suggestions = [
            'database schema',
            'API endpoints',
            'error handling',
            'authentication',
            'configuration',
            'testing',
            'deployment',
            'security',
            'performance',
            'documentation'
        ];
        
        // Add from search history
        const historyQueries = Array.from(this.searchHistory.keys());
        suggestions.push(...historyQueries);
        
        return [...new Set(suggestions)].slice(0, 8);
    }
    
    // Utility methods
    updateSearchStats() {
        document.getElementById('totalSearches').textContent = this.searchStats.totalSearches;
        document.getElementById('avgResponse').textContent = 
            `${this.searchStats.averageResponseTime.toFixed(0)}ms`;
        document.getElementById('cacheHits').textContent = this.searchStats.cacheHits;
    }
    
    updateSearchInterface() {
        // Update interface based on search type
        const realmOption = document.querySelector('.search-option').parentElement;
        if (this.searchType === 'realm') {
            realmOption.style.display = 'block';
        } else {
            realmOption.style.display = 'none';
        }
    }
    
    // Action handlers
    async viewChunkDetails(chunkId) {
        // Dispatch event for main app to handle
        const event = new CustomEvent('viewChunkDetails', {
            detail: { chunkId: chunkId }
        });
        document.dispatchEvent(event);
    }
    
    copyChunkId(chunkId) {
        navigator.clipboard.writeText(chunkId).then(() => {
            window.Utils?.showNotification(`Copied chunk ID: ${chunkId}`, 'success');
        });
    }
    
    highlightChunkInList(chunkId) {
        // Dispatch event for chunk highlighter
        const event = new CustomEvent('highlightChunk', {
            detail: { chunkId: chunkId }
        });
        document.dispatchEvent(event);
    }
    
    selectAllResults(results) {
        const chunkIds = results.map(result => result.chunk_id);
        const event = new CustomEvent('selectChunks', {
            detail: { chunkIds: chunkIds }
        });
        document.dispatchEvent(event);
    }
    
    exportResults(results) {
        const data = {
            timestamp: new Date().toISOString(),
            query: this.getLastSearchQuery(),
            searchType: this.searchType,
            realm: this.currentRealm,
            results: results
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `search_results_${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    clearResults() {
        const container = document.getElementById('searchResults');
        if (container) {
            container.innerHTML = '';
        }
    }
    
    // Cleanup
    destroy() {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        if (this.currentSearch) {
            this.currentSearch.abort();
        }
        
        this.saveSearchHistory();
        this.searchHistory.clear();
        this.searchCache.clear();
    }
}

// Initialize search tester when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.searchTester = new SearchTester();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SearchTester;
}