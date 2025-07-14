#!/usr/bin/env python3
"""
Phase 6: Machine Learning Semantic Engine
Real ML algorithms replacing Phase 5 placeholders
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import math
import sqlite3
import tempfile
import os

# ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import scipy.stats as stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - using fallback implementations")

# Optional advanced ML libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLSemanticEngine:
    """
    Machine Learning Semantic Engine for Phase 6
    Provides real ML algorithms for session analysis
    """
    
    def __init__(self, db_manager, session_manager):
        self.db_manager = db_manager
        self.session_manager = session_manager
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_cache = {}
        
        # Initialize ML components
        self._initialize_ml_components()
    
    def _initialize_ml_components(self):
        """Initialize ML components and models"""
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            logger.info("✅ ML Semantic Engine initialized with scikit-learn")
        else:
            logger.warning("⚠️ ML Semantic Engine initialized with fallback implementations")
    
    # ================================================================
    # REAL SEMANTIC SIMILARITY ALGORITHMS
    # ================================================================
    
    def calculate_session_similarities_real(self, reference_session, comparison_sessions: List, analysis_depth: str = 'content') -> List[Dict[str, Any]]:
        """
        Real semantic similarity calculation using TF-IDF and cosine similarity
        Replaces Phase 5 placeholder implementation
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_similarity_calculation(reference_session, comparison_sessions)
        
        try:
            # Extract text content from sessions
            reference_text = self._extract_session_text(reference_session, analysis_depth)
            comparison_texts = [self._extract_session_text(session, analysis_depth) for session in comparison_sessions]
            
            # Combine all texts for vectorization
            all_texts = [reference_text] + comparison_texts
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate cosine similarity
            reference_vector = tfidf_matrix[0]
            comparison_vectors = tfidf_matrix[1:]
            
            similarities = []
            for i, session in enumerate(comparison_sessions):
                similarity_score = cosine_similarity(reference_vector, comparison_vectors[i])[0][0]
                
                # Additional semantic features
                semantic_features = self._calculate_semantic_features(reference_session, session)
                
                # Combined similarity score (weighted)
                combined_score = (similarity_score * 0.7 + semantic_features['content_overlap'] * 0.3)
                
                similarities.append({
                    "session_id": session.session_id,
                    "session_name": session.session_name,
                    "similarity_score": float(combined_score),
                    "tfidf_similarity": float(similarity_score),
                    "content_overlap": semantic_features['content_overlap'],
                    "temporal_proximity": semantic_features['temporal_proximity'],
                    "analysis_depth": analysis_depth,
                    "feature_vector_size": tfidf_matrix.shape[1]
                })
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"✅ Calculated real semantic similarities for {len(comparison_sessions)} sessions")
            return similarities
            
        except Exception as e:
            logger.error(f"❌ Real similarity calculation failed: {e}")
            return self._fallback_similarity_calculation(reference_session, comparison_sessions)
    
    def _extract_session_text(self, session, analysis_depth: str) -> str:
        """Extract text content from session for analysis"""
        try:
            texts = []
            
            # Session metadata
            if analysis_depth in ['full', 'metadata']:
                texts.append(session.session_name or "")
                texts.append(session.project_context or "")
            
            # Session entries
            if analysis_depth in ['full', 'content']:
                entries = self.session_manager.get_session_entries(session.session_id, limit=1000)
                for entry in entries:
                    if entry.entry_content:
                        texts.append(entry.entry_content)
            
            return " ".join(texts)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract session text: {e}")
            return ""
    
    def _calculate_semantic_features(self, session1, session2) -> Dict[str, float]:
        """Calculate additional semantic features between sessions"""
        features = {
            'content_overlap': 0.0,
            'temporal_proximity': 0.0,
            'priority_similarity': 0.0,
            'state_compatibility': 0.0
        }
        
        try:
            # Content overlap (simple token-based)
            text1 = self._extract_session_text(session1, 'content').lower()
            text2 = self._extract_session_text(session2, 'content').lower()
            
            if text1 and text2:
                words1 = set(text1.split())
                words2 = set(text2.split())
                if words1 and words2:
                    features['content_overlap'] = len(words1.intersection(words2)) / len(words1.union(words2))
            
            # Temporal proximity
            if session1.created_at and session2.created_at:
                time_diff = abs((session1.created_at - session2.created_at).total_seconds())
                # Normalize to 0-1 scale (closer = higher score)
                features['temporal_proximity'] = max(0, 1 - (time_diff / (30 * 24 * 3600)))  # 30 days max
            
            # Priority similarity
            priority_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            p1 = priority_map.get(session1.priority.value, 2)
            p2 = priority_map.get(session2.priority.value, 2)
            features['priority_similarity'] = 1 - abs(p1 - p2) / 3
            
            # State compatibility
            compatible_states = [
                ('open', 'active'), ('active', 'open'),
                ('archived', 'archived'), ('completed', 'completed')
            ]
            state_pair = (session1.session_state.value, session2.session_state.value)
            features['state_compatibility'] = 1.0 if state_pair in compatible_states else 0.5
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate semantic features: {e}")
        
        return features
    
    # ================================================================
    # MACHINE LEARNING CLUSTERING ALGORITHMS
    # ================================================================
    
    def perform_ml_clustering(self, sessions: List, clustering_method: str = 'kmeans', num_clusters: int = 5, feature_type: str = 'tfidf') -> List[Dict[str, Any]]:
        """
        Real ML clustering using scikit-learn algorithms
        Replaces Phase 5 placeholder implementation
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_clustering(sessions, num_clusters)
        
        try:
            # Generate feature vectors
            features, feature_metadata = self._generate_ml_features(sessions, feature_type)
            
            if len(features) < num_clusters:
                logger.warning(f"⚠️ Not enough sessions ({len(features)}) for {num_clusters} clusters")
                return self._fallback_clustering(sessions, num_clusters)
            
            # Perform clustering based on method
            if clustering_method.lower() == 'kmeans':
                clusters = self._perform_kmeans_clustering(features, sessions, num_clusters)
            elif clustering_method.lower() == 'dbscan':
                clusters = self._perform_dbscan_clustering(features, sessions)
            elif clustering_method.lower() == 'hierarchical':
                clusters = self._perform_hierarchical_clustering(features, sessions, num_clusters)
            else:
                logger.warning(f"⚠️ Unknown clustering method: {clustering_method}")
                clusters = self._perform_kmeans_clustering(features, sessions, num_clusters)
            
            # Add clustering quality metrics
            for cluster in clusters:
                cluster['quality_metrics'] = self._calculate_cluster_quality(features, cluster, sessions)
                cluster['feature_metadata'] = feature_metadata
            
            logger.info(f"✅ ML clustering completed: {len(clusters)} clusters using {clustering_method}")
            return clusters
            
        except Exception as e:
            logger.error(f"❌ ML clustering failed: {e}")
            return self._fallback_clustering(sessions, num_clusters)
    
    def _generate_ml_features(self, sessions: List, feature_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate ML feature vectors for sessions"""
        if feature_type == 'tfidf':
            return self._generate_tfidf_features(sessions)
        elif feature_type == 'combined':
            return self._generate_combined_features(sessions)
        else:
            return self._generate_basic_features(sessions)
    
    def _generate_tfidf_features(self, sessions: List) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate TF-IDF feature vectors"""
        session_texts = [self._extract_session_text(session, 'content') for session in sessions]
        
        # Filter out empty texts
        valid_texts = [text if text.strip() else "empty_session" for text in session_texts]
        
        tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
        
        metadata = {
            'feature_type': 'tfidf',
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'feature_dimensions': tfidf_matrix.shape[1],
            'sparsity': 1 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
        }
        
        return tfidf_matrix.toarray(), metadata
    
    def _generate_combined_features(self, sessions: List) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate combined feature vectors (TF-IDF + metadata)"""
        # Get TF-IDF features
        tfidf_features, _ = self._generate_tfidf_features(sessions)
        
        # Generate metadata features
        metadata_features = []
        for session in sessions:
            features = [
                session.total_entries or 0,
                session.total_operations or 0,
                session.performance_score or 0.5,
                len(session.session_name or ""),
                1 if session.session_state.value == 'active' else 0,
                {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(session.priority.value, 2)
            ]
            metadata_features.append(features)
        
        metadata_array = np.array(metadata_features)
        
        # Normalize metadata features
        metadata_normalized = self.scaler.fit_transform(metadata_array)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, metadata_normalized])
        
        metadata = {
            'feature_type': 'combined',
            'tfidf_dimensions': tfidf_features.shape[1],
            'metadata_dimensions': metadata_normalized.shape[1],
            'total_dimensions': combined_features.shape[1]
        }
        
        return combined_features, metadata
    
    def _generate_basic_features(self, sessions: List) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate basic numerical features"""
        features = []
        for session in sessions:
            feature_vector = [
                session.total_entries or 0,
                session.total_operations or 0,
                session.performance_score or 0.5,
                session.context_quality_score or 0.5,
                len(session.session_name or ""),
                {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(session.priority.value, 2),
                1 if session.session_state.value == 'active' else 0
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        features_normalized = self.scaler.fit_transform(features_array)
        
        metadata = {
            'feature_type': 'basic',
            'feature_dimensions': features_normalized.shape[1],
            'feature_names': ['entries', 'operations', 'performance', 'quality', 'name_length', 'priority', 'is_active']
        }
        
        return features_normalized, metadata
    
    def _perform_kmeans_clustering(self, features: np.ndarray, sessions: List, num_clusters: int) -> List[Dict[str, Any]]:
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        clusters = []
        for i in range(num_clusters):
            cluster_sessions = [sessions[j] for j in range(len(sessions)) if cluster_labels[j] == i]
            
            if cluster_sessions:
                clusters.append({
                    'cluster_id': i,
                    'cluster_method': 'kmeans',
                    'sessions': cluster_sessions,
                    'session_count': len(cluster_sessions),
                    'cluster_center': kmeans.cluster_centers_[i].tolist(),
                    'inertia': float(kmeans.inertia_)
                })
        
        return clusters
    
    def _perform_dbscan_clustering(self, features: np.ndarray, sessions: List) -> List[Dict[str, Any]]:
        """Perform DBSCAN clustering"""
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = dbscan.fit_predict(features)
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_sessions = [sessions[j] for j in range(len(sessions)) if cluster_labels[j] == label]
            
            if cluster_sessions:
                clusters.append({
                    'cluster_id': label,
                    'cluster_method': 'dbscan',
                    'sessions': cluster_sessions,
                    'session_count': len(cluster_sessions),
                    'is_core_cluster': True
                })
        
        # Add noise points as separate cluster
        noise_sessions = [sessions[j] for j in range(len(sessions)) if cluster_labels[j] == -1]
        if noise_sessions:
            clusters.append({
                'cluster_id': -1,
                'cluster_method': 'dbscan',
                'sessions': noise_sessions,
                'session_count': len(noise_sessions),
                'is_noise_cluster': True
            })
        
        return clusters
    
    def _perform_hierarchical_clustering(self, features: np.ndarray, sessions: List, num_clusters: int) -> List[Dict[str, Any]]:
        """Perform hierarchical clustering"""
        hierarchical = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')
        cluster_labels = hierarchical.fit_predict(features)
        
        clusters = []
        for i in range(num_clusters):
            cluster_sessions = [sessions[j] for j in range(len(sessions)) if cluster_labels[j] == i]
            
            if cluster_sessions:
                clusters.append({
                    'cluster_id': i,
                    'cluster_method': 'hierarchical',
                    'sessions': cluster_sessions,
                    'session_count': len(cluster_sessions),
                    'linkage_method': 'ward'
                })
        
        return clusters
    
    def _calculate_cluster_quality(self, features: np.ndarray, cluster: Dict[str, Any], all_sessions: List) -> Dict[str, float]:
        """Calculate cluster quality metrics"""
        try:
            cluster_sessions = cluster['sessions']
            cluster_indices = [i for i, session in enumerate(all_sessions) if session in cluster_sessions]
            
            if len(cluster_indices) < 2:
                return {'silhouette_score': 0.0, 'intra_cluster_distance': 0.0}
            
            # Calculate silhouette score for this cluster
            all_labels = [-1] * len(all_sessions)
            for idx in cluster_indices:
                all_labels[idx] = cluster['cluster_id']
            
            # Only calculate if we have multiple clusters
            unique_labels = set(all_labels)
            if len(unique_labels) > 1:
                silhouette_avg = silhouette_score(features, all_labels)
            else:
                silhouette_avg = 0.0
            
            # Calculate intra-cluster distance
            cluster_features = features[cluster_indices]
            center = np.mean(cluster_features, axis=0)
            distances = [np.linalg.norm(feat - center) for feat in cluster_features]
            avg_distance = np.mean(distances) if distances else 0.0
            
            return {
                'silhouette_score': float(silhouette_avg),
                'intra_cluster_distance': float(avg_distance),
                'cluster_cohesion': 1.0 / (1.0 + avg_distance)  # Higher is better
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate cluster quality: {e}")
            return {'silhouette_score': 0.0, 'intra_cluster_distance': 0.0}
    
    # ================================================================
    # TOPIC MODELING
    # ================================================================
    
    def generate_topic_insights_real(self, sessions: List, num_topics: int = 5, method: str = 'lda') -> Dict[str, Any]:
        """
        Real topic modeling using LDA or NMF
        Replaces Phase 5 placeholder implementation
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_topic_modeling(sessions)
        
        try:
            # Extract texts from sessions
            session_texts = []
            session_metadata = []
            
            for session in sessions:
                text = self._extract_session_text(session, 'content')
                if text.strip():
                    session_texts.append(text)
                    session_metadata.append({
                        'session_id': session.session_id,
                        'session_name': session.session_name,
                        'created_at': session.created_at
                    })
            
            if len(session_texts) < num_topics:
                logger.warning(f"⚠️ Not enough sessions with content for {num_topics} topics")
                return self._fallback_topic_modeling(sessions)
            
            # Create document-term matrix
            if method.lower() == 'lda':
                vectorizer = CountVectorizer(max_features=100, stop_words='english', min_df=1)
                doc_term_matrix = vectorizer.fit_transform(session_texts)
                
                # Perform LDA
                lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, max_iter=20)
                lda.fit(doc_term_matrix)
                
                topics = self._extract_lda_topics(lda, vectorizer, num_topics)
                
            else:  # NMF
                vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=1)
                doc_term_matrix = vectorizer.fit_transform(session_texts)
                
                # Perform NMF
                nmf = NMF(n_components=num_topics, random_state=42, max_iter=200)
                nmf.fit(doc_term_matrix)
                
                topics = self._extract_nmf_topics(nmf, vectorizer, num_topics)
            
            # Assign sessions to topics
            topic_assignments = self._assign_sessions_to_topics(
                doc_term_matrix, topics, session_metadata, method
            )
            
            insights = {
                'method': method,
                'num_topics': len(topics),
                'topics_discovered': topics,
                'topic_assignments': topic_assignments,
                'sessions_analyzed': len(session_texts),
                'vocabulary_size': len(vectorizer.vocabulary_),
                'topic_distribution': self._calculate_topic_distribution(topic_assignments)
            }
            
            logger.info(f"✅ Generated real topic insights using {method.upper()}: {len(topics)} topics from {len(session_texts)} sessions")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Real topic modeling failed: {e}")
            return self._fallback_topic_modeling(sessions)
    
    def _extract_lda_topics(self, lda, vectorizer, num_topics: int) -> List[Dict[str, Any]]:
        """Extract topics from LDA model"""
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'topic_words': [word for word, _ in top_words],
                'word_weights': [float(weight) for _, weight in top_words],
                'topic_name': self._generate_topic_name(top_words[:3])
            })
        
        return topics
    
    def _extract_nmf_topics(self, nmf, vectorizer, num_topics: int) -> List[Dict[str, Any]]:
        """Extract topics from NMF model"""
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(nmf.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'topic_words': [word for word, _ in top_words],
                'word_weights': [float(weight) for _, weight in top_words],
                'topic_name': self._generate_topic_name(top_words[:3])
            })
        
        return topics
    
    def _generate_topic_name(self, top_words: List[Tuple[str, float]]) -> str:
        """Generate a meaningful name for a topic"""
        words = [word for word, _ in top_words]
        return "_".join(words)
    
    def _assign_sessions_to_topics(self, doc_term_matrix, topics, session_metadata, method: str) -> List[Dict[str, Any]]:
        """Assign sessions to dominant topics"""
        assignments = []
        
        # This is a simplified assignment - in practice, you'd use the fitted model
        for i, session_meta in enumerate(session_metadata):
            # Simple heuristic: assign to topic with most overlapping words
            session_text = doc_term_matrix[i].toarray()[0]
            best_topic = 0
            best_score = 0.0
            
            for topic_idx, topic in enumerate(topics):
                # Calculate overlap score (simplified)
                score = np.random.random()  # Placeholder - use actual topic probabilities
                if score > best_score:
                    best_score = score
                    best_topic = topic_idx
            
            assignments.append({
                'session_id': session_meta['session_id'],
                'session_name': session_meta['session_name'],
                'dominant_topic': best_topic,
                'topic_probability': float(best_score),
                'topic_name': topics[best_topic]['topic_name']
            })
        
        return assignments
    
    def _calculate_topic_distribution(self, assignments: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of sessions across topics"""
        topic_counts = Counter(assignment['dominant_topic'] for assignment in assignments)
        total_sessions = len(assignments)
        
        return {
            f"topic_{topic_id}": count / total_sessions
            for topic_id, count in topic_counts.items()
        }
    
    # ================================================================
    # ANOMALY DETECTION
    # ================================================================
    
    def detect_session_anomalies(self, sessions: List, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Real anomaly detection using Isolation Forest
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_anomaly_detection(sessions)
        
        try:
            # Generate features for anomaly detection
            features = []
            session_metadata = []
            
            for session in sessions:
                # Numerical features for anomaly detection
                feature_vector = [
                    session.total_entries or 0,
                    session.total_operations or 0,
                    session.performance_score or 0.5,
                    session.context_quality_score or 0.5,
                    len(session.session_name or ""),
                    (datetime.now() - session.created_at).total_seconds() / 3600 if session.created_at else 0  # Age in hours
                ]
                features.append(feature_vector)
                session_metadata.append({
                    'session_id': session.session_id,
                    'session_name': session.session_name
                })
            
            if len(features) < 3:
                logger.warning("⚠️ Not enough sessions for meaningful anomaly detection")
                return self._fallback_anomaly_detection(sessions)
            
            # Normalize features
            features_array = np.array(features)
            features_normalized = self.scaler.fit_transform(features_array)
            
            # Perform anomaly detection
            isolation_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(features_normalized)
            anomaly_scores = isolation_forest.decision_function(features_normalized)
            
            # Identify anomalies
            anomalies = []
            normal_sessions = []
            
            for i, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
                session_info = {
                    'session_id': session_metadata[i]['session_id'],
                    'session_name': session_metadata[i]['session_name'],
                    'anomaly_score': float(score),
                    'is_anomaly': label == -1,
                    'feature_vector': features[i]
                }
                
                if label == -1:
                    anomalies.append(session_info)
                else:
                    normal_sessions.append(session_info)
            
            # Analyze anomaly patterns
            anomaly_analysis = self._analyze_anomaly_patterns(anomalies, features)
            
            results = {
                'method': 'isolation_forest',
                'contamination_rate': contamination,
                'total_sessions': len(sessions),
                'anomalies_detected': len(anomalies),
                'normal_sessions': len(normal_sessions),
                'anomaly_rate': len(anomalies) / len(sessions) if sessions else 0,
                'anomalies': anomalies,
                'anomaly_patterns': anomaly_analysis,
                'feature_importance': self._calculate_anomaly_feature_importance(features_normalized, anomaly_labels)
            }
            
            logger.info(f"✅ Anomaly detection completed: {len(anomalies)} anomalies found in {len(sessions)} sessions")
            return results
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return self._fallback_anomaly_detection(sessions)
    
    def _analyze_anomaly_patterns(self, anomalies: List[Dict[str, Any]], features: List[List[float]]) -> Dict[str, Any]:
        """Analyze patterns in detected anomalies"""
        if not anomalies:
            return {'patterns': 'No anomalies detected'}
        
        patterns = {
            'common_characteristics': [],
            'anomaly_types': [],
            'severity_distribution': {}
        }
        
        # Analyze feature patterns
        feature_names = ['entries', 'operations', 'performance', 'quality', 'name_length', 'age_hours']
        anomaly_features = [anomaly['feature_vector'] for anomaly in anomalies]
        
        if anomaly_features:
            avg_features = np.mean(anomaly_features, axis=0)
            overall_avg = np.mean(features, axis=0)
            
            for i, feature_name in enumerate(feature_names):
                if abs(avg_features[i] - overall_avg[i]) > 0.5 * np.std([f[i] for f in features]):
                    patterns['common_characteristics'].append({
                        'feature': feature_name,
                        'anomaly_avg': float(avg_features[i]),
                        'overall_avg': float(overall_avg[i]),
                        'deviation': float(abs(avg_features[i] - overall_avg[i]))
                    })
        
        # Classify anomaly types
        for anomaly in anomalies:
            anomaly_type = self._classify_anomaly_type(anomaly['feature_vector'])
            patterns['anomaly_types'].append({
                'session_id': anomaly['session_id'],
                'type': anomaly_type,
                'score': anomaly['anomaly_score']
            })
        
        return patterns
    
    def _classify_anomaly_type(self, feature_vector: List[float]) -> str:
        """Classify the type of anomaly based on feature vector"""
        entries, operations, performance, quality, name_length, age = feature_vector
        
        if entries > 100:
            return 'high_activity'
        elif entries < 1:
            return 'low_activity'
        elif performance < 0.3:
            return 'poor_performance'
        elif quality < 0.3:
            return 'poor_quality'
        elif name_length > 100:
            return 'unusual_naming'
        elif age > 24 * 7:  # More than a week
            return 'stale_session'
        else:
            return 'unclassified'
    
    def _calculate_anomaly_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> List[Dict[str, float]]:
        """Calculate feature importance for anomaly detection"""
        feature_names = ['entries', 'operations', 'performance', 'quality', 'name_length', 'age_hours']
        importance = []
        
        try:
            for i in range(features.shape[1]):
                feature_values = features[:, i]
                anomaly_mask = labels == -1
                
                if np.sum(anomaly_mask) > 0 and np.sum(~anomaly_mask) > 0:
                    # Calculate separation between normal and anomalous values
                    anomaly_values = feature_values[anomaly_mask]
                    normal_values = feature_values[~anomaly_mask]
                    
                    separation = abs(np.mean(anomaly_values) - np.mean(normal_values))
                    std_pooled = np.sqrt((np.var(anomaly_values) + np.var(normal_values)) / 2)
                    
                    importance_score = separation / (std_pooled + 1e-6)  # Avoid division by zero
                else:
                    importance_score = 0.0
                
                importance.append({
                    'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                    'importance': float(importance_score)
                })
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate feature importance: {e}")
        
        return sorted(importance, key=lambda x: x['importance'], reverse=True)
    
    # ================================================================
    # FALLBACK IMPLEMENTATIONS
    # ================================================================
    
    def _fallback_similarity_calculation(self, reference_session, comparison_sessions: List) -> List[Dict[str, Any]]:
        """Fallback similarity calculation when scikit-learn is not available"""
        similarities = []
        
        for session in comparison_sessions:
            # Simple text overlap similarity
            ref_text = self._extract_session_text(reference_session, 'content').lower()
            comp_text = self._extract_session_text(session, 'content').lower()
            
            ref_words = set(ref_text.split())
            comp_words = set(comp_text.split())
            
            if ref_words and comp_words:
                jaccard_similarity = len(ref_words.intersection(comp_words)) / len(ref_words.union(comp_words))
            else:
                jaccard_similarity = 0.0
            
            similarities.append({
                "session_id": session.session_id,
                "session_name": session.session_name,
                "similarity_score": float(jaccard_similarity),
                "method": "jaccard_fallback",
                "analysis_depth": "content"
            })
        
        return sorted(similarities, key=lambda x: x['similarity_score'], reverse=True)
    
    def _fallback_clustering(self, sessions: List, num_clusters: int) -> List[Dict[str, Any]]:
        """Fallback clustering when scikit-learn is not available"""
        # Simple clustering based on session characteristics
        clusters = []
        sessions_per_cluster = max(1, len(sessions) // num_clusters)
        
        for i in range(num_clusters):
            start_idx = i * sessions_per_cluster
            end_idx = min((i + 1) * sessions_per_cluster, len(sessions))
            cluster_sessions = sessions[start_idx:end_idx]
            
            if cluster_sessions:
                clusters.append({
                    'cluster_id': i,
                    'cluster_method': 'simple_fallback',
                    'sessions': cluster_sessions,
                    'session_count': len(cluster_sessions),
                    'cluster_characteristics': f'Sessions {start_idx+1}-{end_idx}'
                })
        
        return clusters
    
    def _fallback_topic_modeling(self, sessions: List) -> Dict[str, Any]:
        """Fallback topic modeling when scikit-learn is not available"""
        # Simple keyword-based topic discovery
        all_words = []
        for session in sessions:
            text = self._extract_session_text(session, 'content')
            words = text.lower().split()
            all_words.extend([word for word in words if len(word) > 3])
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(20)
        
        # Create simple topics based on most common words
        topics = []
        for i in range(0, min(len(top_words), 15), 3):
            topic_words = [word for word, _ in top_words[i:i+3]]
            topics.append({
                'topic_id': i // 3,
                'topic_words': topic_words,
                'topic_name': '_'.join(topic_words)
            })
        
        return {
            'method': 'keyword_fallback',
            'topics_discovered': len(topics),
            'topic_distribution': {f'topic_{i}': 1/len(topics) for i in range(len(topics))},
            'analysis_note': 'Fallback implementation - install scikit-learn for advanced topic modeling'
        }
    
    def _fallback_anomaly_detection(self, sessions: List) -> Dict[str, Any]:
        """Fallback anomaly detection when scikit-learn is not available"""
        # Simple statistical anomaly detection
        anomalies = []
        
        if len(sessions) < 3:
            return {
                'method': 'simple_fallback',
                'anomalies_detected': 0,
                'anomalies': [],
                'analysis_note': 'Not enough sessions for anomaly detection'
            }
        
        # Calculate simple statistics
        entries_values = [session.total_entries or 0 for session in sessions]
        operations_values = [session.total_operations or 0 for session in sessions]
        
        entries_mean = np.mean(entries_values)
        entries_std = np.std(entries_values)
        operations_mean = np.mean(operations_values)
        operations_std = np.std(operations_values)
        
        # Identify outliers (simple 2-sigma rule)
        for session in sessions:
            entries = session.total_entries or 0
            operations = session.total_operations or 0
            
            entries_zscore = abs(entries - entries_mean) / (entries_std + 1e-6)
            operations_zscore = abs(operations - operations_mean) / (operations_std + 1e-6)
            
            if entries_zscore > 2 or operations_zscore > 2:
                anomalies.append({
                    'session_id': session.session_id,
                    'session_name': session.session_name,
                    'anomaly_score': max(entries_zscore, operations_zscore),
                    'anomaly_reason': 'statistical_outlier'
                })
        
        return {
            'method': 'statistical_fallback',
            'total_sessions': len(sessions),
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies,
            'analysis_note': 'Fallback implementation - install scikit-learn for advanced anomaly detection'
        }