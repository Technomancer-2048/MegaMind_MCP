/**
 * Chunk boundary visualization and highlighting functionality
 * Manages chunk selection, highlighting, and boundary visualization
 */

class ChunkHighlighter {
    constructor() {
        this.selectedChunks = new Set();
        this.allChunks = new Map();
        this.highlightMode = 'boundaries'; // 'boundaries', 'content', 'relationships'
        this.initializeHighlighter();
    }
    
    initializeHighlighter() {
        this.bindEvents();
        this.setupHighlightControls();
    }
    
    bindEvents() {
        // Listen for chunk render events
        document.addEventListener('chunksRendered', (e) => {
            this.processRenderedChunks(e.detail.chunks);
        });
        
        // Listen for chunk selection changes
        document.addEventListener('chunkSelectionChanged', (e) => {
            this.updateHighlighting();
        });
        
        // Highlight mode changes
        document.addEventListener('highlightModeChanged', (e) => {
            this.highlightMode = e.detail.mode;
            this.updateHighlighting();
        });
    }
    
    setupHighlightControls() {
        const controlsHtml = `
            <div class="highlight-controls">
                <label for="highlightMode">Highlight Mode:</label>
                <select id="highlightMode" class="form-control">
                    <option value="boundaries">Chunk Boundaries</option>
                    <option value="content">Content Quality</option>
                    <option value="relationships">Relationships</option>
                </select>
                <button id="toggleHighlights" class="btn btn-sm btn-outline">
                    Toggle Highlights
                </button>
            </div>
        `;
        
        // Add controls to chunk header if it exists
        const chunkHeader = document.querySelector('.chunk-header');
        if (chunkHeader) {
            const controlsContainer = document.createElement('div');
            controlsContainer.innerHTML = controlsHtml;
            chunkHeader.appendChild(controlsContainer);
            
            // Bind control events
            const modeSelect = document.getElementById('highlightMode');
            modeSelect.addEventListener('change', (e) => {
                this.highlightMode = e.target.value;
                this.updateHighlighting();
            });
            
            const toggleBtn = document.getElementById('toggleHighlights');
            toggleBtn.addEventListener('click', () => {
                this.toggleHighlights();
            });
        }
    }
    
    processRenderedChunks(chunks) {
        chunks.forEach(chunk => {
            this.allChunks.set(chunk.chunk_id, chunk);
            this.addChunkBoundaryVisualization(chunk);
        });
        
        this.updateHighlighting();
    }
    
    addChunkBoundaryVisualization(chunk) {
        const chunkElement = document.querySelector(`[data-chunk-id="${chunk.chunk_id}"]`);
        if (!chunkElement) return;
        
        // Add boundary markers
        this.addBoundaryMarkers(chunkElement, chunk);
        
        // Add hover effects
        this.addHoverEffects(chunkElement, chunk);
        
        // Add complexity indicators
        this.addComplexityIndicators(chunkElement, chunk);
    }
    
    addBoundaryMarkers(chunkElement, chunk) {
        // Add visual boundary indicators
        const boundaryMarker = document.createElement('div');
        boundaryMarker.className = 'chunk-boundary-marker';
        boundaryMarker.innerHTML = `
            <div class="boundary-top"></div>
            <div class="boundary-info">
                <span class="boundary-label">Chunk ${chunk.chunk_id}</span>
                <span class="boundary-stats">${chunk.line_count || 0} lines, ${chunk.token_count || 0} tokens</span>
            </div>
            <div class="boundary-bottom"></div>
        `;
        
        chunkElement.insertBefore(boundaryMarker, chunkElement.firstChild);
    }
    
    addHoverEffects(chunkElement, chunk) {
        chunkElement.addEventListener('mouseenter', () => {
            this.showChunkPreview(chunk);
            this.highlightRelatedChunks(chunk);
        });
        
        chunkElement.addEventListener('mouseleave', () => {
            this.hideChunkPreview();
            this.clearRelatedHighlights();
        });
    }
    
    addComplexityIndicators(chunkElement, chunk) {
        const complexityScore = chunk.complexity_score || 0;
        const complexityClass = this.getComplexityClass(complexityScore);
        
        chunkElement.classList.add(`complexity-${complexityClass}`);
        
        // Add complexity visualization
        const complexityIndicator = document.createElement('div');
        complexityIndicator.className = 'complexity-indicator';
        complexityIndicator.innerHTML = `
            <div class="complexity-bar">
                <div class="complexity-fill" style="width: ${Math.min(complexityScore * 100, 100)}%"></div>
            </div>
            <span class="complexity-label">${complexityScore.toFixed(2)}</span>
        `;
        
        const chunkHeader = chunkElement.querySelector('.chunk-header');
        if (chunkHeader) {
            chunkHeader.appendChild(complexityIndicator);
        }
    }
    
    getComplexityClass(score) {
        if (score < 0.3) return 'low';
        if (score < 0.7) return 'medium';
        return 'high';
    }
    
    showChunkPreview(chunk) {
        const preview = document.getElementById('chunkPreview') || this.createPreviewElement();
        
        preview.innerHTML = `
            <div class="preview-header">
                <h4>Chunk ${chunk.chunk_id}</h4>
                <button class="preview-close" onclick="this.parentElement.parentElement.style.display='none'">&times;</button>
            </div>
            <div class="preview-content">
                <div class="preview-meta">
                    <span><strong>Source:</strong> ${chunk.source_document}</span>
                    <span><strong>Type:</strong> ${chunk.chunk_type}</span>
                    <span><strong>Status:</strong> ${chunk.approval_status}</span>
                </div>
                <div class="preview-text">
                    ${this.formatChunkContent(chunk.content)}
                </div>
            </div>
        `;
        
        preview.style.display = 'block';
        this.positionPreview(preview);
    }
    
    createPreviewElement() {
        const preview = document.createElement('div');
        preview.id = 'chunkPreview';
        preview.className = 'chunk-preview';
        document.body.appendChild(preview);
        return preview;
    }
    
    formatChunkContent(content) {
        if (!content) return '<em>No content</em>';
        
        // Highlight important sections
        let formatted = content;
        
        // Highlight code blocks
        formatted = formatted.replace(/```[\s\S]*?```/g, (match) => {
            return `<div class="code-block">${match}</div>`;
        });
        
        // Highlight headers
        formatted = formatted.replace(/^#+\s+(.+)$/gm, (match, header) => {
            return `<div class="header-highlight">${match}</div>`;
        });
        
        // Highlight important keywords
        const keywords = ['IMPORTANT', 'NOTE', 'WARNING', 'ERROR', 'TODO'];
        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
            formatted = formatted.replace(regex, `<span class="keyword-highlight">${keyword}</span>`);
        });
        
        return `<pre class="content-preview">${formatted}</pre>`;
    }
    
    positionPreview(preview) {
        const rect = event.target.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        let left = rect.right + 10;
        let top = rect.top;
        
        // Adjust if preview would go off screen
        if (left + 400 > viewportWidth) {
            left = rect.left - 410;
        }
        
        if (top + 300 > viewportHeight) {
            top = viewportHeight - 320;
        }
        
        preview.style.left = `${left}px`;
        preview.style.top = `${top}px`;
    }
    
    hideChunkPreview() {
        const preview = document.getElementById('chunkPreview');
        if (preview) {
            preview.style.display = 'none';
        }
    }
    
    highlightRelatedChunks(chunk) {
        // Find and highlight related chunks
        this.findRelatedChunks(chunk.chunk_id).then(related => {
            related.forEach(relatedChunk => {
                const element = document.querySelector(`[data-chunk-id="${relatedChunk.chunk_id}"]`);
                if (element) {
                    element.classList.add('related-highlight');
                }
            });
        });
    }
    
    async findRelatedChunks(chunkId) {
        try {
            const response = await fetch(`/api/chunks/${chunkId}/relationships`);
            const data = await response.json();
            return data.success ? data.relationships : [];
        } catch (error) {
            console.error('Error finding related chunks:', error);
            return [];
        }
    }
    
    clearRelatedHighlights() {
        document.querySelectorAll('.related-highlight').forEach(element => {
            element.classList.remove('related-highlight');
        });
    }
    
    updateHighlighting() {
        const chunks = document.querySelectorAll('.chunk-item');
        
        chunks.forEach(chunkElement => {
            const chunkId = chunkElement.dataset.chunkId;
            const chunk = this.allChunks.get(chunkId);
            
            if (!chunk) return;
            
            // Clear existing highlights
            chunkElement.classList.remove('highlight-boundaries', 'highlight-content', 'highlight-relationships');
            
            // Apply new highlighting based on mode
            switch (this.highlightMode) {
                case 'boundaries':
                    this.applyBoundaryHighlighting(chunkElement, chunk);
                    break;
                case 'content':
                    this.applyContentHighlighting(chunkElement, chunk);
                    break;
                case 'relationships':
                    this.applyRelationshipHighlighting(chunkElement, chunk);
                    break;
            }
        });
    }
    
    applyBoundaryHighlighting(chunkElement, chunk) {
        chunkElement.classList.add('highlight-boundaries');
        
        // Add boundary strength indicators
        const boundaryStrength = this.calculateBoundaryStrength(chunk);
        chunkElement.classList.add(`boundary-strength-${boundaryStrength}`);
    }
    
    applyContentHighlighting(chunkElement, chunk) {
        chunkElement.classList.add('highlight-content');
        
        // Highlight based on content quality metrics
        const qualityScore = this.calculateContentQuality(chunk);
        chunkElement.classList.add(`content-quality-${qualityScore}`);
    }
    
    applyRelationshipHighlighting(chunkElement, chunk) {
        chunkElement.classList.add('highlight-relationships');
        
        // Highlight based on relationship density
        const relationshipDensity = this.calculateRelationshipDensity(chunk);
        chunkElement.classList.add(`relationship-density-${relationshipDensity}`);
    }
    
    calculateBoundaryStrength(chunk) {
        // Calculate how well-defined the chunk boundaries are
        const factors = [
            chunk.line_count > 10 ? 1 : 0,
            chunk.token_count > 50 ? 1 : 0,
            chunk.section_path ? 1 : 0,
            chunk.chunk_type !== 'fragment' ? 1 : 0
        ];
        
        const strength = factors.reduce((sum, factor) => sum + factor, 0);
        
        if (strength >= 3) return 'strong';
        if (strength >= 2) return 'medium';
        return 'weak';
    }
    
    calculateContentQuality(chunk) {
        // Calculate content quality based on various factors
        const content = chunk.content || '';
        const factors = [
            content.length > 200 ? 1 : 0,
            content.includes('```') ? 1 : 0, // Has code blocks
            /^#+\s+/.test(content) ? 1 : 0, // Has headers
            content.split('\n').length > 5 ? 1 : 0, // Multi-line
            !/TODO|FIXME|XXX/.test(content) ? 1 : 0 // No TODO items
        ];
        
        const quality = factors.reduce((sum, factor) => sum + factor, 0);
        
        if (quality >= 4) return 'high';
        if (quality >= 2) return 'medium';
        return 'low';
    }
    
    calculateRelationshipDensity(chunk) {
        // This would need to be calculated based on actual relationships
        // For now, use a simple heuristic
        const accessCount = chunk.access_count || 0;
        
        if (accessCount > 10) return 'high';
        if (accessCount > 3) return 'medium';
        return 'low';
    }
    
    toggleHighlights() {
        const chunks = document.querySelectorAll('.chunk-item');
        const isHighlighted = chunks[0]?.classList.contains('highlights-enabled');
        
        chunks.forEach(chunk => {
            if (isHighlighted) {
                chunk.classList.remove('highlights-enabled');
            } else {
                chunk.classList.add('highlights-enabled');
            }
        });
    }
    
    // Selection management
    selectChunk(chunkId) {
        this.selectedChunks.add(chunkId);
        this.updateChunkSelection(chunkId, true);
        this.dispatchSelectionEvent();
    }
    
    deselectChunk(chunkId) {
        this.selectedChunks.delete(chunkId);
        this.updateChunkSelection(chunkId, false);
        this.dispatchSelectionEvent();
    }
    
    updateChunkSelection(chunkId, selected) {
        const chunkElement = document.querySelector(`[data-chunk-id="${chunkId}"]`);
        if (chunkElement) {
            const checkbox = chunkElement.querySelector('.chunk-checkbox');
            if (checkbox) {
                checkbox.checked = selected;
            }
            
            if (selected) {
                chunkElement.classList.add('selected');
            } else {
                chunkElement.classList.remove('selected');
            }
        }
    }
    
    dispatchSelectionEvent() {
        const event = new CustomEvent('chunkSelectionChanged', {
            detail: {
                selectedChunks: Array.from(this.selectedChunks),
                count: this.selectedChunks.size
            }
        });
        document.dispatchEvent(event);
    }
    
    // Advanced highlighting features
    highlightSearchTerms(terms) {
        const chunks = document.querySelectorAll('.chunk-content');
        chunks.forEach(chunkContent => {
            let content = chunkContent.textContent;
            
            terms.forEach(term => {
                const regex = new RegExp(`(${term})`, 'gi');
                content = content.replace(regex, '<mark class="search-highlight">$1</mark>');
            });
            
            chunkContent.innerHTML = content;
        });
    }
    
    clearSearchHighlights() {
        const highlights = document.querySelectorAll('.search-highlight');
        highlights.forEach(highlight => {
            highlight.outerHTML = highlight.textContent;
        });
    }
    
    // Animation helpers
    animateChunkSelection(chunkId) {
        const chunkElement = document.querySelector(`[data-chunk-id="${chunkId}"]`);
        if (chunkElement) {
            chunkElement.classList.add('selection-animation');
            setTimeout(() => {
                chunkElement.classList.remove('selection-animation');
            }, 500);
        }
    }
    
    // Cleanup
    destroy() {
        this.selectedChunks.clear();
        this.allChunks.clear();
        
        // Remove event listeners
        document.removeEventListener('chunksRendered', this.processRenderedChunks);
        document.removeEventListener('chunkSelectionChanged', this.updateHighlighting);
        
        // Remove preview element
        const preview = document.getElementById('chunkPreview');
        if (preview) {
            preview.remove();
        }
    }
}

// Initialize chunk highlighter when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chunkHighlighter = new ChunkHighlighter();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ChunkHighlighter;
}