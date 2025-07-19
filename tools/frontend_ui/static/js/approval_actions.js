/**
 * Approval and rejection workflows with batch operations
 * Handles chunk approval, rejection, and bulk operations with audit trails
 */

class ApprovalActions {
    constructor() {
        this.selectedChunks = new Set();
        this.approvalHistory = new Map();
        this.rejectionReasons = new Map();
        this.bulkOperationInProgress = false;
        this.approvalStats = {
            totalApprovals: 0,
            totalRejections: 0,
            averageApprovalTime: 0
        };
        
        this.initializeApprovalActions();
    }
    
    initializeApprovalActions() {
        this.bindEvents();
        this.loadApprovalHistory();
        this.setupApprovalInterface();
    }
    
    bindEvents() {
        // Listen for chunk selection changes
        document.addEventListener('chunkSelectionChanged', (e) => {
            this.selectedChunks = new Set(e.detail.selectedChunks);
            this.updateApprovalUI();
        });
        
        // Individual chunk actions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('approve-single')) {
                this.approveSingleChunk(e.target.dataset.chunkId);
            } else if (e.target.classList.contains('reject-single')) {
                this.rejectSingleChunk(e.target.dataset.chunkId);
            }
        });
        
        // Bulk actions
        document.getElementById('approveSelected')?.addEventListener('click', () => {
            this.showBulkApprovalDialog();
        });
        
        document.getElementById('rejectSelected')?.addEventListener('click', () => {
            this.showBulkRejectionDialog();
        });
        
        // Approval dialog handlers
        this.bindApprovalDialogEvents();
        
        // Rejection dialog handlers
        this.bindRejectionDialogEvents();
    }
    
    bindApprovalDialogEvents() {
        document.getElementById('confirmBulkApproval')?.addEventListener('click', () => {
            this.confirmBulkApproval();
        });
        
        document.getElementById('cancelBulkApproval')?.addEventListener('click', () => {
            this.hideBulkApprovalDialog();
        });
    }
    
    bindRejectionDialogEvents() {
        document.getElementById('confirmReject')?.addEventListener('click', () => {
            this.confirmBulkRejection();
        });
        
        document.getElementById('cancelReject')?.addEventListener('click', () => {
            this.hideBulkRejectionDialog();
        });
        
        // Rejection reason templates
        document.getElementById('rejectionTemplates')?.addEventListener('change', (e) => {
            const template = e.target.value;
            if (template) {
                document.getElementById('rejectionReason').value = template;
            }
        });
    }
    
    setupApprovalInterface() {
        this.createApprovalDialogs();
        this.createRejectionReasonTemplates();
        this.updateApprovalStats();
    }
    
    createApprovalDialogs() {
        // Bulk approval dialog
        const bulkApprovalDialog = `
            <div class="modal approval-dialog" id="bulkApprovalDialog" style="display: none;">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>Bulk Approval</h3>
                        <button class="modal-close" id="cancelBulkApproval">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="approval-summary">
                            <p>You are about to approve <strong id="bulkApprovalCount">0</strong> chunks.</p>
                            <div class="approval-preview" id="bulkApprovalPreview"></div>
                        </div>
                        
                        <div class="approval-options">
                            <div class="approval-option">
                                <label for="approvalNotes">Approval Notes (optional):</label>
                                <textarea id="approvalNotes" placeholder="Add notes about this approval..." rows="3"></textarea>
                            </div>
                            
                            <div class="approval-option">
                                <label for="approverName">Approved By:</label>
                                <input type="text" id="approverName" value="frontend_user" class="form-control">
                            </div>
                            
                            <div class="approval-option">
                                <label>
                                    <input type="checkbox" id="sendNotification" checked>
                                    Send approval notifications
                                </label>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-success" id="confirmBulkApproval">
                            Approve All
                        </button>
                        <button class="btn btn-secondary" id="cancelBulkApproval">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', bulkApprovalDialog);
    }
    
    createRejectionReasonTemplates() {
        const templates = [
            { value: '', label: 'Select a template...' },
            { value: 'Content is outdated and needs revision', label: 'Outdated Content' },
            { value: 'Technical information is incorrect', label: 'Technical Accuracy' },
            { value: 'Does not meet style guide requirements', label: 'Style Compliance' },
            { value: 'Missing required information or incomplete', label: 'Incomplete Content' },
            { value: 'Contains sensitive information that should not be included', label: 'Sensitive Information' },
            { value: 'Duplicate content already exists elsewhere', label: 'Duplicate Content' },
            { value: 'Code examples are broken or non-functional', label: 'Broken Code' },
            { value: 'Content quality is below standards', label: 'Quality Issues' }
        ];
        
        const rejectionPanel = document.getElementById('rejectionPanel');
        if (rejectionPanel) {
            const templateSelect = document.createElement('select');
            templateSelect.id = 'rejectionTemplates';
            templateSelect.className = 'form-control';
            templateSelect.innerHTML = templates.map(template => 
                `<option value="${template.value}">${template.label}</option>`
            ).join('');
            
            // Insert before rejection reason textarea
            const reasonTextarea = document.getElementById('rejectionReason');
            if (reasonTextarea) {
                const label = document.createElement('label');
                label.textContent = 'Rejection Reason Templates:';
                label.style.display = 'block';
                label.style.marginBottom = '5px';
                
                reasonTextarea.parentNode.insertBefore(label, reasonTextarea);
                reasonTextarea.parentNode.insertBefore(templateSelect, reasonTextarea);
            }
        }
    }
    
    updateApprovalUI() {
        const count = this.selectedChunks.size;
        const approveBtn = document.getElementById('approveSelected');
        const rejectBtn = document.getElementById('rejectSelected');
        const selectedCountSpan = document.getElementById('selectedCount');
        
        if (selectedCountSpan) {
            selectedCountSpan.textContent = count;
        }
        
        if (approveBtn && rejectBtn) {
            if (count > 0) {
                approveBtn.disabled = false;
                rejectBtn.disabled = false;
                approveBtn.textContent = `Approve Selected (${count})`;
                rejectBtn.textContent = `Reject Selected (${count})`;
            } else {
                approveBtn.disabled = true;
                rejectBtn.disabled = true;
                approveBtn.textContent = 'Approve Selected (0)';
                rejectBtn.textContent = 'Reject Selected (0)';
            }
        }
    }
    
    async approveSingleChunk(chunkId) {
        if (!chunkId) return;
        
        const startTime = Date.now();
        
        try {
            const response = await this.performApproval([chunkId], 'frontend_user');
            
            if (response.success) {
                const approvalTime = Date.now() - startTime;
                this.recordApprovalAction(chunkId, 'approved', approvalTime);
                
                window.Utils?.showNotification('Chunk approved successfully', 'success');
                
                // Update chunk UI
                this.updateChunkStatus(chunkId, 'approved');
                
                // Refresh data
                this.dispatchRefreshEvent();
                
            } else {
                throw new Error(response.error || 'Failed to approve chunk');
            }
        } catch (error) {
            console.error('Error approving chunk:', error);
            window.Utils?.showNotification('Failed to approve chunk', 'error');
        }
    }
    
    async rejectSingleChunk(chunkId) {
        if (!chunkId) return;
        
        const reason = await this.promptForRejectionReason();
        if (!reason) return;
        
        const startTime = Date.now();
        
        try {
            const response = await this.performRejection([chunkId], reason, 'frontend_user');
            
            if (response.success) {
                const rejectionTime = Date.now() - startTime;
                this.recordApprovalAction(chunkId, 'rejected', rejectionTime, reason);
                
                window.Utils?.showNotification('Chunk rejected successfully', 'success');
                
                // Update chunk UI
                this.updateChunkStatus(chunkId, 'rejected');
                
                // Refresh data
                this.dispatchRefreshEvent();
                
            } else {
                throw new Error(response.error || 'Failed to reject chunk');
            }
        } catch (error) {
            console.error('Error rejecting chunk:', error);
            window.Utils?.showNotification('Failed to reject chunk', 'error');
        }
    }
    
    showBulkApprovalDialog() {
        if (this.selectedChunks.size === 0) {
            window.Utils?.showNotification('No chunks selected', 'error');
            return;
        }
        
        const dialog = document.getElementById('bulkApprovalDialog');
        const countSpan = document.getElementById('bulkApprovalCount');
        const previewDiv = document.getElementById('bulkApprovalPreview');
        
        if (!dialog || !countSpan || !previewDiv) return;
        
        countSpan.textContent = this.selectedChunks.size;
        
        // Create preview of selected chunks
        previewDiv.innerHTML = '';
        const chunkList = Array.from(this.selectedChunks).slice(0, 5);
        
        chunkList.forEach(chunkId => {
            const chunkElement = document.querySelector(`[data-chunk-id="${chunkId}"]`);
            if (chunkElement) {
                const chunkTitle = chunkElement.querySelector('.chunk-id')?.textContent || chunkId;
                const chunkSource = chunkElement.querySelector('.chunk-source')?.textContent || 'Unknown';
                
                const previewItem = document.createElement('div');
                previewItem.className = 'approval-preview-item';
                previewItem.innerHTML = `
                    <strong>${chunkTitle}</strong> - ${chunkSource}
                `;
                previewDiv.appendChild(previewItem);
            }
        });
        
        if (this.selectedChunks.size > 5) {
            const moreItem = document.createElement('div');
            moreItem.className = 'approval-preview-more';
            moreItem.textContent = `... and ${this.selectedChunks.size - 5} more`;
            previewDiv.appendChild(moreItem);
        }
        
        dialog.style.display = 'block';
    }
    
    hideBulkApprovalDialog() {
        const dialog = document.getElementById('bulkApprovalDialog');
        if (dialog) {
            dialog.style.display = 'none';
        }
    }
    
    async confirmBulkApproval() {
        if (this.bulkOperationInProgress) return;
        
        this.bulkOperationInProgress = true;
        
        const approverName = document.getElementById('approverName')?.value || 'frontend_user';
        const approvalNotes = document.getElementById('approvalNotes')?.value || '';
        const chunkIds = Array.from(this.selectedChunks);
        
        try {
            // Show progress
            this.showBulkProgress('Approving chunks...');
            
            const response = await this.performApproval(chunkIds, approverName, approvalNotes);
            
            if (response.success) {
                // Record approval actions
                chunkIds.forEach(chunkId => {
                    this.recordApprovalAction(chunkId, 'approved', 0, approvalNotes);
                });
                
                window.Utils?.showNotification(
                    `Successfully approved ${response.approved_count} chunks`, 
                    'success'
                );
                
                // Update UI
                this.updateMultipleChunkStatus(chunkIds, 'approved');
                this.clearSelection();
                this.hideBulkApprovalDialog();
                this.dispatchRefreshEvent();
                
            } else {
                throw new Error(response.error || 'Failed to approve chunks');
            }
        } catch (error) {
            console.error('Error in bulk approval:', error);
            window.Utils?.showNotification('Failed to approve chunks', 'error');
        } finally {
            this.bulkOperationInProgress = false;
            this.hideBulkProgress();
        }
    }
    
    showBulkRejectionDialog() {
        if (this.selectedChunks.size === 0) {
            window.Utils?.showNotification('No chunks selected', 'error');
            return;
        }
        
        const rejectionPanel = document.getElementById('rejectionPanel');
        if (rejectionPanel) {
            rejectionPanel.style.display = 'block';
            
            // Update rejection count
            const countSpan = rejectionPanel.querySelector('.rejection-count');
            if (countSpan) {
                countSpan.textContent = this.selectedChunks.size;
            }
            
            // Focus on reason textarea
            const reasonTextarea = document.getElementById('rejectionReason');
            if (reasonTextarea) {
                reasonTextarea.focus();
            }
        }
    }
    
    hideBulkRejectionDialog() {
        const rejectionPanel = document.getElementById('rejectionPanel');
        if (rejectionPanel) {
            rejectionPanel.style.display = 'none';
            
            // Clear reason
            const reasonTextarea = document.getElementById('rejectionReason');
            if (reasonTextarea) {
                reasonTextarea.value = '';
            }
        }
    }
    
    async confirmBulkRejection() {
        if (this.bulkOperationInProgress) return;
        
        const rejectionReason = document.getElementById('rejectionReason')?.value?.trim();
        if (!rejectionReason) {
            window.Utils?.showNotification('Please provide a rejection reason', 'error');
            return;
        }
        
        this.bulkOperationInProgress = true;
        
        const chunkIds = Array.from(this.selectedChunks);
        
        try {
            // Show progress
            this.showBulkProgress('Rejecting chunks...');
            
            const response = await this.performRejection(chunkIds, rejectionReason, 'frontend_user');
            
            if (response.success) {
                // Record rejection actions
                chunkIds.forEach(chunkId => {
                    this.recordApprovalAction(chunkId, 'rejected', 0, rejectionReason);
                });
                
                window.Utils?.showNotification(
                    `Successfully rejected ${response.rejected_count} chunks`, 
                    'success'
                );
                
                // Update UI
                this.updateMultipleChunkStatus(chunkIds, 'rejected');
                this.clearSelection();
                this.hideBulkRejectionDialog();
                this.dispatchRefreshEvent();
                
            } else {
                throw new Error(response.error || 'Failed to reject chunks');
            }
        } catch (error) {
            console.error('Error in bulk rejection:', error);
            window.Utils?.showNotification('Failed to reject chunks', 'error');
        } finally {
            this.bulkOperationInProgress = false;
            this.hideBulkProgress();
        }
    }
    
    async performApproval(chunkIds, approvedBy, approvalNotes = '') {
        const response = await fetch('/api/chunks/approve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                chunk_ids: chunkIds,
                approved_by: approvedBy,
                approval_notes: approvalNotes
            })
        });
        
        return await response.json();
    }
    
    async performRejection(chunkIds, rejectionReason, rejectedBy) {
        const response = await fetch('/api/chunks/reject', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                chunk_ids: chunkIds,
                rejection_reason: rejectionReason,
                rejected_by: rejectedBy
            })
        });
        
        return await response.json();
    }
    
    recordApprovalAction(chunkId, action, timeMs, reason = '') {
        const record = {
            chunkId: chunkId,
            action: action,
            timestamp: new Date().toISOString(),
            timeMs: timeMs,
            reason: reason
        };
        
        this.approvalHistory.set(`${chunkId}_${Date.now()}`, record);
        
        // Update stats
        if (action === 'approved') {
            this.approvalStats.totalApprovals++;
        } else if (action === 'rejected') {
            this.approvalStats.totalRejections++;
        }
        
        if (timeMs > 0) {
            this.approvalStats.averageApprovalTime = 
                (this.approvalStats.averageApprovalTime + timeMs) / 2;
        }
        
        this.updateApprovalStats();
        this.saveApprovalHistory();
    }
    
    updateChunkStatus(chunkId, status) {
        const chunkElement = document.querySelector(`[data-chunk-id="${chunkId}"]`);
        if (chunkElement) {
            // Update visual indicators
            chunkElement.classList.remove('status-pending', 'status-approved', 'status-rejected');
            chunkElement.classList.add(`status-${status}`);
            
            // Update status displays
            const statusElements = chunkElement.querySelectorAll('.status');
            statusElements.forEach(element => {
                element.textContent = status;
                element.className = `status status-${status}`;
            });
            
            // Add approved/rejected timestamp
            const timestamp = new Date().toLocaleString();
            const timestampElement = chunkElement.querySelector('.status-timestamp');
            if (timestampElement) {
                timestampElement.textContent = timestamp;
            }
        }
    }
    
    updateMultipleChunkStatus(chunkIds, status) {
        chunkIds.forEach(chunkId => {
            this.updateChunkStatus(chunkId, status);
        });
    }
    
    showBulkProgress(message) {
        const progressDialog = document.createElement('div');
        progressDialog.id = 'bulkProgressDialog';
        progressDialog.className = 'modal progress-dialog';
        progressDialog.innerHTML = `
            <div class="modal-content">
                <div class="modal-body">
                    <div class="progress-content">
                        <div class="progress-spinner"></div>
                        <p>${message}</p>
                        <div class="progress-bar">
                            <div class="progress-fill"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(progressDialog);
        progressDialog.style.display = 'block';
    }
    
    hideBulkProgress() {
        const progressDialog = document.getElementById('bulkProgressDialog');
        if (progressDialog) {
            progressDialog.remove();
        }
    }
    
    async promptForRejectionReason() {
        return new Promise((resolve) => {
            const reason = prompt('Enter rejection reason:');
            resolve(reason ? reason.trim() : null);
        });
    }
    
    clearSelection() {
        this.selectedChunks.clear();
        
        // Dispatch event to update UI
        const event = new CustomEvent('chunkSelectionChanged', {
            detail: {
                selectedChunks: [],
                count: 0
            }
        });
        document.dispatchEvent(event);
    }
    
    dispatchRefreshEvent() {
        const event = new CustomEvent('refreshRequired', {
            detail: { reason: 'approval_action' }
        });
        document.dispatchEvent(event);
    }
    
    // Statistics and history management
    updateApprovalStats() {
        const statsContainer = document.getElementById('approvalStats');
        if (!statsContainer) return;
        
        statsContainer.innerHTML = `
            <div class="approval-stats">
                <div class="stat-item">
                    <span class="stat-label">Total Approvals:</span>
                    <span class="stat-value">${this.approvalStats.totalApprovals}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Total Rejections:</span>
                    <span class="stat-value">${this.approvalStats.totalRejections}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Avg Time:</span>
                    <span class="stat-value">${this.approvalStats.averageApprovalTime.toFixed(0)}ms</span>
                </div>
            </div>
        `;
    }
    
    loadApprovalHistory() {
        const stored = localStorage.getItem('approvalHistory');
        if (stored) {
            try {
                const history = JSON.parse(stored);
                this.approvalHistory = new Map(history);
                
                // Recalculate stats
                this.recalculateStats();
            } catch (error) {
                console.error('Error loading approval history:', error);
            }
        }
    }
    
    saveApprovalHistory() {
        try {
            const history = Array.from(this.approvalHistory.entries());
            localStorage.setItem('approvalHistory', JSON.stringify(history));
        } catch (error) {
            console.error('Error saving approval history:', error);
        }
    }
    
    recalculateStats() {
        let totalApprovals = 0;
        let totalRejections = 0;
        let totalTime = 0;
        let timeCount = 0;
        
        this.approvalHistory.forEach(record => {
            if (record.action === 'approved') {
                totalApprovals++;
            } else if (record.action === 'rejected') {
                totalRejections++;
            }
            
            if (record.timeMs > 0) {
                totalTime += record.timeMs;
                timeCount++;
            }
        });
        
        this.approvalStats.totalApprovals = totalApprovals;
        this.approvalStats.totalRejections = totalRejections;
        this.approvalStats.averageApprovalTime = timeCount > 0 ? totalTime / timeCount : 0;
    }
    
    // Export functionality
    exportApprovalHistory() {
        const history = Array.from(this.approvalHistory.values());
        const data = {
            timestamp: new Date().toISOString(),
            stats: this.approvalStats,
            history: history
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `approval_history_${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    // Cleanup
    destroy() {
        this.saveApprovalHistory();
        this.approvalHistory.clear();
        this.selectedChunks.clear();
    }
}

// Initialize approval actions when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.approvalActions = new ApprovalActions();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ApprovalActions;
}