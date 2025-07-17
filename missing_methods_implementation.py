    def knowledge_ingest_document_dual_realm(self, document_path: str, 
                                           processing_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ingest document with knowledge processing"""
        try:
            import os
            
            if not os.path.exists(document_path):
                return {
                    "success": False,
                    "error": f"Document not found: {document_path}"
                }
            
            # Read document content
            with open(document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            document_name = os.path.basename(document_path)
            
            # Analyze document first
            analysis = self.content_analyze_document_dual_realm(
                content, document_name, processing_options.get('session_id', '') if processing_options else ''
            )
            
            # Create chunks based on analysis
            if analysis['success']:
                suggested_chunks = analysis['analysis']['suggested_chunks']
                chunk_size = len(content) // max(1, suggested_chunks)
                
                created_chunks = []
                for i in range(0, len(content), chunk_size):
                    chunk_content = content[i:i + chunk_size]
                    if len(chunk_content.strip()) > 50:  # Skip tiny chunks
                        chunk_result = self.create_chunk_dual_realm(
                            content=chunk_content,
                            source_document=document_name,
                            section_path=f"/ingested/chunk_{i//chunk_size + 1}",
                            session_id=processing_options.get('session_id', '') if processing_options else ''
                        )
                        if chunk_result['success']:
                            created_chunks.append(chunk_result['chunk_id'])
                
                return {
                    "success": True,
                    "document_path": document_path,
                    "document_name": document_name,
                    "chunks_created": len(created_chunks),
                    "chunk_ids": created_chunks,
                    "analysis": analysis['analysis'],
                    "ingested_at": datetime.now().isoformat()
                }
            else:
                return analysis
                
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_path": document_path
            }
    
    def knowledge_discover_relationships_dual_realm(self, chunk_ids: List[str], 
                                                   discovery_method: str = "semantic") -> Dict[str, Any]:
        """Discover relationships between chunks"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            discovered_relationships = []
            
            # Get chunk contents
            if len(chunk_ids) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 chunks for relationship discovery"
                }
            
            chunk_placeholders = ",".join(["%s"] * len(chunk_ids))
            chunk_query = f"""
            SELECT chunk_id, content, source_document, section_path
            FROM megamind_chunks
            WHERE chunk_id IN ({chunk_placeholders})
            """
            cursor.execute(chunk_query, chunk_ids)
            chunks = cursor.fetchall()
            
            # Simple relationship discovery based on content similarity
            for i, chunk1 in enumerate(chunks):
                for j, chunk2 in enumerate(chunks[i+1:], i+1):
                    # Calculate basic similarity
                    content1_words = set(chunk1['content'].lower().split())
                    content2_words = set(chunk2['content'].lower().split())
                    
                    if content1_words and content2_words:
                        similarity = len(content1_words & content2_words) / len(content1_words | content2_words)
                        
                        if similarity > 0.1:  # Threshold for relationship
                            relationship_type = "semantic_similarity" if discovery_method == "semantic" else "content_related"
                            
                            # Create relationship if it doesn't exist
                            add_result = self.add_relationship_dual_realm(
                                chunk1['chunk_id'], 
                                chunk2['chunk_id'], 
                                relationship_type,
                                ""  # session_id
                            )
                            
                            if add_result['success']:
                                discovered_relationships.append({
                                    "chunk_id_1": chunk1['chunk_id'],
                                    "chunk_id_2": chunk2['chunk_id'],
                                    "relationship_type": relationship_type,
                                    "strength_score": similarity,
                                    "discovery_method": discovery_method
                                })
            
            return {
                "success": True,
                "discovery_method": discovery_method,
                "chunk_ids": chunk_ids,
                "relationships_discovered": len(discovered_relationships),
                "relationships": discovered_relationships,
                "discovered_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Relationship discovery failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunk_ids": chunk_ids
            }
        finally:
            if connection:
                connection.close()
    
    def knowledge_optimize_retrieval_dual_realm(self, target_queries: List[str], 
                                              optimization_strategy: str = "performance") -> Dict[str, Any]:
        """Optimize retrieval performance for target queries"""
        try:
            optimization_results = []
            
            for query in target_queries:
                # Test current retrieval performance
                start_time = datetime.now()
                results = self.search_chunks_dual_realm(query, 10, "hybrid")
                end_time = datetime.now()
                
                retrieval_time = (end_time - start_time).total_seconds()
                
                optimization_results.append({
                    "query": query,
                    "current_performance": {
                        "retrieval_time_seconds": retrieval_time,
                        "results_count": len(results),
                        "avg_relevance_score": sum(r.get('access_count', 0) for r in results) / max(1, len(results))
                    },
                    "optimization_suggestions": [
                        "Consider adding more specific embeddings",
                        "Optimize database indexes for frequent queries",
                        "Cache frequently accessed chunks"
                    ]
                })
            
            return {
                "success": True,
                "optimization_strategy": optimization_strategy,
                "queries_analyzed": len(target_queries),
                "optimization_results": optimization_results,
                "overall_performance": {
                    "avg_retrieval_time": sum(r["current_performance"]["retrieval_time_seconds"] for r in optimization_results) / len(optimization_results),
                    "total_results": sum(r["current_performance"]["results_count"] for r in optimization_results)
                },
                "optimized_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Retrieval optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "target_queries": target_queries
            }

    # ========================================
    # 🚀 PROMOTION CLASS - Missing Methods (6)
    # ========================================
    
    def create_promotion_request_dual_realm(self, chunk_id: str, target_realm: str, 
                                           justification: str, session_id: str = "") -> Dict[str, Any]:
        """Create promotion request with dual-realm support"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Verify chunk exists
            chunk_query = "SELECT chunk_id, realm_id FROM megamind_chunks WHERE chunk_id = %s"
            cursor.execute(chunk_query, (chunk_id,))
            chunk = cursor.fetchone()
            
            if not chunk:
                return {
                    "success": False,
                    "error": f"Chunk not found: {chunk_id}"
                }
            
            # Generate promotion ID
            promotion_id = f"promotion_{uuid.uuid4().hex[:8]}"
            
            # Insert promotion request
            insert_query = """
            INSERT INTO megamind_promotion_queue 
            (promotion_id, chunk_id, source_realm, target_realm, justification, status, created_by, created_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            now = datetime.now()
            cursor.execute(insert_query, (
                promotion_id, chunk_id, chunk['realm_id'], target_realm, 
                justification, 'pending', session_id, now
            ))
            
            connection.commit()
            
            return {
                "success": True,
                "promotion_id": promotion_id,
                "chunk_id": chunk_id,
                "source_realm": chunk['realm_id'],
                "target_realm": target_realm,
                "status": "pending",
                "created_at": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create promotion request: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunk_id": chunk_id
            }
        finally:
            if connection:
                connection.close()
    
    def get_promotion_requests_dual_realm(self, filter_status: str = "", 
                                        filter_realm: str = "", 
                                        limit: int = 20) -> List[Dict[str, Any]]:
        """Get promotion requests with filtering"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Build query with filters
            query = """
            SELECT p.promotion_id, p.chunk_id, p.source_realm, p.target_realm,
                   p.justification, p.status, p.created_by, p.created_date,
                   c.content, c.source_document, c.section_path
            FROM megamind_promotion_queue p
            JOIN megamind_chunks c ON p.chunk_id = c.chunk_id
            WHERE 1=1
            """
            params = []
            
            if filter_status:
                query += " AND p.status = %s"
                params.append(filter_status)
            
            if filter_realm:
                query += " AND p.target_realm = %s"
                params.append(filter_realm)
            
            query += " ORDER BY p.created_date DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            requests = cursor.fetchall()
            
            # Add metadata
            for request in requests:
                request['days_pending'] = (datetime.now() - request['created_date']).days
            
            return requests
            
        except Exception as e:
            logger.error(f"Failed to get promotion requests: {e}")
            return []
        finally:
            if connection:
                connection.close()
    
    def get_promotion_queue_summary_dual_realm(self, filter_realm: str = "") -> Dict[str, Any]:
        """Get promotion queue summary statistics"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Base query
            base_query = "SELECT status, COUNT(*) as count FROM megamind_promotion_queue"
            params = []
            
            if filter_realm:
                base_query += " WHERE target_realm = %s"
                params.append(filter_realm)
            
            base_query += " GROUP BY status"
            
            cursor.execute(base_query, params)
            status_counts = cursor.fetchall()
            
            summary = {
                "total_pending": 0,
                "total_approved": 0,
                "total_rejected": 0,
                "total_requests": 0,
                "status_breakdown": status_counts,
                "filter_realm": filter_realm or "all"
            }
            
            for status_count in status_counts:
                if status_count['status'] == 'pending':
                    summary['total_pending'] = status_count['count']
                elif status_count['status'] == 'approved':
                    summary['total_approved'] = status_count['count']
                elif status_count['status'] == 'rejected':
                    summary['total_rejected'] = status_count['count']
                
                summary['total_requests'] += status_count['count']
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get promotion queue summary: {e}")
            return {
                "success": False,
                "error": str(e),
                "filter_realm": filter_realm
            }
        finally:
            if connection:
                connection.close()
    
    def approve_promotion_request_dual_realm(self, promotion_id: str, 
                                           approval_reason: str, 
                                           session_id: str = "") -> Dict[str, Any]:
        """Approve promotion request"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Update promotion status
            update_query = """
            UPDATE megamind_promotion_queue 
            SET status = 'approved', updated_date = %s 
            WHERE promotion_id = %s AND status = 'pending'
            """
            
            now = datetime.now()
            cursor.execute(update_query, (now, promotion_id))
            
            if cursor.rowcount == 0:
                return {
                    "success": False,
                    "error": "Promotion request not found or already processed"
                }
            
            # Add to history
            history_query = """
            INSERT INTO megamind_promotion_history 
            (promotion_id, action, action_reason, action_by, action_date)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(history_query, (
                promotion_id, 'approved', approval_reason, session_id, now
            ))
            
            connection.commit()
            
            return {
                "success": True,
                "promotion_id": promotion_id,
                "action": "approved",
                "reason": approval_reason,
                "approved_at": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to approve promotion: {e}")
            return {
                "success": False,
                "error": str(e),
                "promotion_id": promotion_id
            }
        finally:
            if connection:
                connection.close()
    
    def reject_promotion_request_dual_realm(self, promotion_id: str, 
                                          rejection_reason: str, 
                                          session_id: str = "") -> Dict[str, Any]:
        """Reject promotion request"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Update promotion status
            update_query = """
            UPDATE megamind_promotion_queue 
            SET status = 'rejected', updated_date = %s 
            WHERE promotion_id = %s AND status = 'pending'
            """
            
            now = datetime.now()
            cursor.execute(update_query, (now, promotion_id))
            
            if cursor.rowcount == 0:
                return {
                    "success": False,
                    "error": "Promotion request not found or already processed"
                }
            
            # Add to history
            history_query = """
            INSERT INTO megamind_promotion_history 
            (promotion_id, action, action_reason, action_by, action_date)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(history_query, (
                promotion_id, 'rejected', rejection_reason, session_id, now
            ))
            
            connection.commit()
            
            return {
                "success": True,
                "promotion_id": promotion_id,
                "action": "rejected",
                "reason": rejection_reason,
                "rejected_at": now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to reject promotion: {e}")
            return {
                "success": False,
                "error": str(e),
                "promotion_id": promotion_id
            }
        finally:
            if connection:
                connection.close()
    
    def get_promotion_impact_dual_realm(self, promotion_id: str) -> Dict[str, Any]:
        """Analyze promotion impact"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get promotion details
            promo_query = """
            SELECT p.chunk_id, p.source_realm, p.target_realm, 
                   c.content, c.source_document
            FROM megamind_promotion_queue p
            JOIN megamind_chunks c ON p.chunk_id = c.chunk_id
            WHERE p.promotion_id = %s
            """
            cursor.execute(promo_query, (promotion_id,))
            promo = cursor.fetchone()
            
            if not promo:
                return {
                    "success": False,
                    "error": "Promotion not found"
                }
            
            # Analyze relationships
            rel_query = """
            SELECT COUNT(*) as relationship_count
            FROM megamind_chunk_relationships
            WHERE chunk_id_1 = %s OR chunk_id_2 = %s
            """
            cursor.execute(rel_query, (promo['chunk_id'], promo['chunk_id']))
            rel_count = cursor.fetchone()['relationship_count']
            
            # Check for similar content in target realm
            similar_query = """
            SELECT COUNT(*) as similar_count
            FROM megamind_chunks
            WHERE realm_id = %s AND source_document = %s
            """
            cursor.execute(similar_query, (promo['target_realm'], promo['source_document']))
            similar_count = cursor.fetchone()['similar_count']
            
            return {
                "success": True,
                "promotion_id": promotion_id,
                "impact_analysis": {
                    "relationship_count": rel_count,
                    "similar_content_in_target": similar_count,
                    "conflict_probability": "low" if similar_count == 0 else "medium",
                    "estimated_integration_effort": "low" if rel_count < 5 else "medium"
                },
                "confidence_score": 0.8 if similar_count == 0 else 0.6,
                "analyzed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze promotion impact: {e}")
            return {
                "success": False,
                "error": str(e),
                "promotion_id": promotion_id
            }
        finally:
            if connection:
                connection.close()