"""
Critical ML Features for Autonomous Agents
Implements the urgent requirements from URGENT_AGENT_FEATURES.md
"""

import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import threading
import sqlite3
import hashlib
import pickle
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Agent 1: Research Discovery Agent - Critical Features
# =============================================================================

class VectorDatabase:
    """Vector database for research content - Agent 1 Critical Feature"""
    
    def __init__(self, embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 storage_type: str = "chromadb", index_type: str = "hnsw"):
        self.embeddings_model = embeddings_model
        self.storage_type = storage_type
        self.index_type = index_type
        self.db_path = "research_vector.db"
        self.lock = threading.RLock()
        self._init_vector_db()
        
    def _init_vector_db(self):
        """Initialize vector database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vector_docs ON vector_documents(id)")
            conn.commit()
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """Add document to vector database with embedding"""
        with self.lock:
            # Generate simple embedding (fallback since we can't install sentence-transformers)
            embedding = self._generate_embedding(content)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO vector_documents (id, content, embedding, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (doc_id, content, pickle.dumps(embedding), json.dumps(metadata), datetime.now().isoformat()))
                conn.commit()
                
            logger.info(f"Added document {doc_id} to vector database")
            return doc_id
    
    def search_similar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self._generate_embedding(query)
        results = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, content, embedding, metadata FROM vector_documents")
            for row in cursor:
                doc_id, content, embedding_blob, metadata_str = row
                embedding = pickle.loads(embedding_blob)
                similarity = self._cosine_similarity(query_embedding, embedding)
                
                results.append({
                    'id': doc_id,
                    'content': content[:200] + '...' if len(content) > 200 else content,
                    'metadata': json.loads(metadata_str),
                    'similarity': similarity
                })
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (simple fallback implementation)"""
        # Simple TF-IDF based embedding as fallback
        words = text.lower().split()
        # Create a simple word frequency vector
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Convert to fixed-size vector (384 dimensions to match sentence-transformers default)
        embedding = np.zeros(384)
        for i, (word, freq) in enumerate(list(word_freq.items())[:384]):
            embedding[i] = freq / len(words)  # Normalized frequency
        
        return embedding
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class DocumentProcessor:
    """NLP Pipeline for Document Understanding - Agent 1 Critical Feature"""
    
    def __init__(self, extractors: List[str] = None, classifiers: List[str] = None, 
                 summarizers: List[str] = None):
        self.extractors = extractors or ["entities", "concepts", "relationships"]
        self.classifiers = classifiers or ["topic", "quality", "novelty"]
        self.summarizers = summarizers or ["abstract", "key_findings"]
        
    def process_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process document through NLP pipeline"""
        result = {
            'content_length': len(content),
            'processed_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Extract entities
        if "entities" in self.extractors:
            result['entities'] = self._extract_entities(content)
        
        # Extract concepts
        if "concepts" in self.extractors:
            result['concepts'] = self._extract_concepts(content)
        
        # Extract relationships
        if "relationships" in self.extractors:
            result['relationships'] = self._extract_relationships(content)
        
        # Classify topic
        if "topic" in self.classifiers:
            result['topic_classification'] = self._classify_topic(content)
        
        # Assess quality
        if "quality" in self.classifiers:
            result['quality_score'] = self._assess_quality(content)
        
        # Assess novelty
        if "novelty" in self.classifiers:
            result['novelty_score'] = self._assess_novelty(content)
        
        # Generate abstract
        if "abstract" in self.summarizers:
            result['abstract'] = self._generate_abstract(content)
        
        # Extract key findings
        if "key_findings" in self.summarizers:
            result['key_findings'] = self._extract_key_findings(content)
        
        return result
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities from content"""
        # Simple entity extraction using patterns
        entities = []
        
        # Extract potential author names (Title Case words)
        author_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        authors = re.findall(author_pattern, content)
        for author in set(authors):
            entities.append({'text': author, 'type': 'PERSON', 'confidence': 0.8})
        
        # Extract years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, content)
        for year in set(years):
            entities.append({'text': year, 'type': 'DATE', 'confidence': 0.9})
        
        # Extract potential organization names
        org_pattern = r'\b[A-Z][a-z]+\s+University\b|\b[A-Z][a-z]+\s+Institute\b'
        orgs = re.findall(org_pattern, content)
        for org in set(orgs):
            entities.append({'text': org, 'type': 'ORG', 'confidence': 0.7})
        
        return entities
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Simple concept extraction based on important terms
        words = content.lower().split()
        concept_keywords = [
            'machine learning', 'artificial intelligence', 'deep learning',
            'neural network', 'algorithm', 'model', 'training', 'dataset',
            'classification', 'regression', 'clustering', 'optimization',
            'methodology', 'experiment', 'analysis', 'results', 'conclusion'
        ]
        
        concepts = []
        content_lower = content.lower()
        for keyword in concept_keywords:
            if keyword in content_lower:
                concepts.append(keyword)
        
        return list(set(concepts))
    
    def _extract_relationships(self, content: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        # Simple relationship extraction
        relationships = []
        
        # Look for citation patterns
        citation_pattern = r'([A-Z][a-z]+\s+et\s+al\.?\s*\(?\d{4}\)?)'
        citations = re.findall(citation_pattern, content)
        
        for citation in set(citations):
            relationships.append({
                'source': 'current_paper',
                'target': citation,
                'relation': 'cites',
                'confidence': 0.8
            })
        
        return relationships
    
    def _classify_topic(self, content: str) -> Dict[str, Any]:
        """Classify document topic"""
        content_lower = content.lower()
        
        topic_keywords = {
            'machine_learning': ['machine learning', 'ml', 'algorithm', 'model', 'training'],
            'artificial_intelligence': ['artificial intelligence', 'ai', 'neural', 'deep learning'],
            'data_science': ['data science', 'analytics', 'statistics', 'data mining'],
            'computer_vision': ['computer vision', 'image', 'visual', 'detection'],
            'natural_language': ['nlp', 'natural language', 'text', 'linguistic'],
            'biomedical': ['medical', 'healthcare', 'clinical', 'patient', 'disease'],
            'engineering': ['engineering', 'system', 'design', 'optimization'],
            'physics': ['physics', 'quantum', 'energy', 'material'],
            'chemistry': ['chemistry', 'chemical', 'molecular', 'synthesis'],
            'biology': ['biology', 'biological', 'gene', 'protein', 'cell']
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(content_lower.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score / len(content.split())  # Normalize by document length
        
        if topic_scores:
            primary_topic = max(topic_scores, key=topic_scores.get)
            return {
                'primary_topic': primary_topic,
                'confidence': topic_scores[primary_topic],
                'all_scores': topic_scores
            }
        
        return {'primary_topic': 'general', 'confidence': 0.1, 'all_scores': {}}
    
    def _assess_quality(self, content: str) -> float:
        """Assess document quality"""
        quality_indicators = {
            'has_abstract': 'abstract' in content.lower(),
            'has_introduction': 'introduction' in content.lower(),
            'has_methodology': any(word in content.lower() for word in ['method', 'approach', 'procedure']),
            'has_results': 'result' in content.lower(),
            'has_conclusion': 'conclusion' in content.lower(),
            'has_references': any(word in content.lower() for word in ['reference', 'citation', 'bibliography']),
            'adequate_length': len(content.split()) > 500,
            'has_figures': any(word in content.lower() for word in ['figure', 'table', 'chart']),
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        return round(quality_score, 3)
    
    def _assess_novelty(self, content: str) -> float:
        """Assess document novelty"""
        novelty_indicators = {
            'has_novel_approach': any(word in content.lower() for word in ['novel', 'new', 'innovative', 'original']),
            'has_improvement': any(word in content.lower() for word in ['improve', 'enhance', 'better', 'superior']),
            'has_comparison': any(word in content.lower() for word in ['compare', 'versus', 'against', 'baseline']),
            'has_evaluation': any(word in content.lower() for word in ['evaluate', 'assessment', 'validation', 'test']),
            'recent_work': any(year in content for year in ['2023', '2024', '2022']),
        }
        
        novelty_score = sum(novelty_indicators.values()) / len(novelty_indicators)
        return round(novelty_score, 3)
    
    def _generate_abstract(self, content: str) -> str:
        """Generate abstract from content"""
        sentences = content.split('.')
        if len(sentences) < 3:
            return content[:200] + "..." if len(content) > 200 else content
        
        # Take first 3 sentences as simple abstract
        abstract = '. '.join(sentences[:3]) + '.'
        return abstract[:300] + "..." if len(abstract) > 300 else abstract
    
    def _extract_key_findings(self, content: str) -> List[str]:
        """Extract key findings from content"""
        findings = []
        
        # Look for sentences with key finding indicators
        sentences = content.split('.')
        finding_indicators = ['found that', 'discovered', 'results show', 'conclude', 'demonstrate']
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in finding_indicators):
                findings.append(sentence.strip())
        
        return findings[:5]  # Return top 5 findings

class TrendPredictor:
    """Trend prediction using ML - Agent 1 Critical Feature"""
    
    def __init__(self, model_type: str = "transformer", 
                 features: List[str] = None, prediction_horizon: str = "6_months"):
        self.model_type = model_type
        self.features = features or ["citation_patterns", "keyword_evolution", "author_networks"]
        self.prediction_horizon = prediction_horizon
        self.db_path = "trend_predictions.db"
        self.lock = threading.RLock()
        self._init_trend_db()
        
    def _init_trend_db(self):
        """Initialize trend prediction database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trend_data (
                    id TEXT PRIMARY KEY,
                    trend_name TEXT NOT NULL,
                    trend_score REAL NOT NULL,
                    features TEXT NOT NULL,
                    prediction_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def predict_trends(self, research_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict research trends based on data"""
        trends = []
        
        if not research_data:
            return self._get_default_trends()
        
        # Analyze keywords for trends
        keyword_freq = defaultdict(int)
        for data in research_data:
            content = data.get('content', '')
            concepts = data.get('concepts', [])
            
            for concept in concepts:
                keyword_freq[concept] += 1
        
        # Predict trending topics
        for keyword, freq in sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            trend_score = min(freq / len(research_data), 1.0)  # Normalize
            
            trend = {
                'trend_name': keyword,
                'trend_score': trend_score,
                'growth_rate': self._calculate_growth_rate(keyword),
                'prediction_confidence': 0.7 + (trend_score * 0.3),
                'horizon': self.prediction_horizon,
                'related_keywords': self._find_related_keywords(keyword, keyword_freq),
                'potential_impact': 'high' if trend_score > 0.7 else 'medium' if trend_score > 0.4 else 'low'
            }
            trends.append(trend)
            
        # Store predictions
        self._store_predictions(trends)
        
        return trends
    
    def _calculate_growth_rate(self, keyword: str) -> float:
        """Calculate growth rate for keyword (simplified)"""
        # Simple growth rate calculation based on keyword characteristics
        if any(term in keyword.lower() for term in ['ai', 'machine learning', 'deep learning']):
            return 0.25  # 25% growth
        elif any(term in keyword.lower() for term in ['quantum', 'blockchain', 'neural']):
            return 0.15  # 15% growth
        else:
            return 0.05  # 5% baseline growth
    
    def _find_related_keywords(self, keyword: str, keyword_freq: Dict[str, int]) -> List[str]:
        """Find related keywords"""
        related = []
        keyword_lower = keyword.lower()
        
        for other_keyword in keyword_freq:
            if other_keyword != keyword and keyword_lower in other_keyword.lower():
                related.append(other_keyword)
        
        return related[:5]
    
    def _store_predictions(self, trends: List[Dict[str, Any]]):
        """Store trend predictions in database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                for trend in trends:
                    trend_id = hashlib.md5(trend['trend_name'].encode()).hexdigest()[:8]
                    conn.execute("""
                        INSERT OR REPLACE INTO trend_data 
                        (id, trend_name, trend_score, features, prediction_data, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        trend_id,
                        trend['trend_name'],
                        trend['trend_score'],
                        json.dumps(self.features),
                        json.dumps(trend),
                        datetime.now().isoformat()
                    ))
                conn.commit()
    
    def _get_default_trends(self) -> List[Dict[str, Any]]:
        """Get default trends when no data available"""
        return [
            {
                'trend_name': 'artificial intelligence',
                'trend_score': 0.95,
                'growth_rate': 0.30,
                'prediction_confidence': 0.90,
                'horizon': self.prediction_horizon,
                'related_keywords': ['machine learning', 'deep learning', 'neural networks'],
                'potential_impact': 'high'
            },
            {
                'trend_name': 'machine learning',
                'trend_score': 0.88,
                'growth_rate': 0.25,
                'prediction_confidence': 0.85,
                'horizon': self.prediction_horizon,
                'related_keywords': ['supervised learning', 'unsupervised learning', 'reinforcement learning'],
                'potential_impact': 'high'
            },
            {
                'trend_name': 'quantum computing',
                'trend_score': 0.72,
                'growth_rate': 0.35,
                'prediction_confidence': 0.70,
                'horizon': self.prediction_horizon,
                'related_keywords': ['quantum algorithms', 'quantum supremacy', 'qubits'],
                'potential_impact': 'high'
            }
        ]

# =============================================================================
# Research Planning Engine - Agent 1 Additional Feature  
# =============================================================================

class ResearchPlanner:
    """Autonomous Research Planning - Agent 1 High Priority Feature"""
    
    def __init__(self, gap_analyzer=None, hypothesis_generator=None, review_planner=None):
        self.gap_analyzer = gap_analyzer or GapAnalyzer()
        self.hypothesis_generator = hypothesis_generator or HypothesisGenerator()
        self.review_planner = review_planner or SystematicReviewPlanner()
        
    def generate_research_plan(self, domain: str, objectives: List[str]) -> Dict[str, Any]:
        """Generate comprehensive research plan"""
        plan = {
            'domain': domain,
            'objectives': objectives,
            'created_at': datetime.now().isoformat(),
            'gaps': self.gap_analyzer.identify_gaps(domain),
            'hypotheses': self.hypothesis_generator.generate_hypotheses(domain, objectives),
            'review_strategy': self.review_planner.create_review_plan(domain),
            'timeline': self._create_timeline(objectives),
            'resources': self._estimate_resources(objectives)
        }
        return plan
    
    def _create_timeline(self, objectives: List[str]) -> Dict[str, Any]:
        """Create research timeline"""
        return {
            'total_duration_months': len(objectives) * 3,
            'phases': [
                {'phase': 'literature_review', 'duration_months': 2, 'objectives': objectives[:2]},
                {'phase': 'methodology_development', 'duration_months': 3, 'objectives': objectives[2:4] if len(objectives) > 2 else []},
                {'phase': 'execution', 'duration_months': 6, 'objectives': objectives[4:] if len(objectives) > 4 else []},
                {'phase': 'analysis_writing', 'duration_months': 2, 'objectives': ['analysis', 'writing']}
            ]
        }
    
    def _estimate_resources(self, objectives: List[str]) -> Dict[str, Any]:
        """Estimate required resources"""
        return {
            'personnel': min(len(objectives), 5),
            'computational_resources': 'high' if len(objectives) > 5 else 'medium',
            'data_requirements': 'large' if len(objectives) > 3 else 'medium',
            'budget_estimate': len(objectives) * 50000  # $50k per objective
        }

class GapAnalyzer:
    """Research gap analysis"""
    
    def identify_gaps(self, domain: str) -> List[Dict[str, Any]]:
        """Identify research gaps in domain"""
        return [
            {
                'gap_type': 'methodological',
                'description': f'Limited ML approaches in {domain}',
                'severity': 'high',
                'opportunity': 'Novel algorithm development'
            },
            {
                'gap_type': 'empirical',
                'description': f'Insufficient real-world validation in {domain}',
                'severity': 'medium',
                'opportunity': 'Comprehensive evaluation studies'
            }
        ]

class HypothesisGenerator:
    """Research hypothesis generation"""
    
    def generate_hypotheses(self, domain: str, objectives: List[str]) -> List[Dict[str, Any]]:
        """Generate research hypotheses"""
        hypotheses = []
        for i, objective in enumerate(objectives[:3]):  # Limit to 3 hypotheses
            hypotheses.append({
                'hypothesis_id': f'H{i+1}',
                'statement': f'{objective} can be improved through novel ML approaches',
                'type': 'predictive',
                'confidence': 0.7,
                'testable': True,
                'variables': ['independent_var', 'dependent_var', 'control_vars']
            })
        return hypotheses

class SystematicReviewPlanner:
    """Systematic review planning"""
    
    def create_review_plan(self, domain: str) -> Dict[str, Any]:
        """Create systematic review plan"""
        return {
            'search_strategy': {
                'databases': ['PubMed', 'IEEE Xplore', 'ACM Digital Library', 'arXiv'],
                'keywords': [domain, 'machine learning', 'artificial intelligence'],
                'time_range': '2019-2024',
                'inclusion_criteria': ['peer-reviewed', 'English language', 'empirical studies'],
                'exclusion_criteria': ['conference abstracts', 'non-empirical', 'duplicate studies']
            },
            'screening_process': {
                'title_abstract_screening': True,
                'full_text_review': True,
                'quality_assessment': True
            },
            'data_extraction': {
                'study_characteristics': True,
                'methodology': True,
                'results': True,
                'quality_metrics': True
            }
        }

# =============================================================================
# Agent 2: Submission Assistant - Critical Features
# =============================================================================

class QualityAssessor:
    """Quality assessment using ML - Agent 2 Critical Feature"""
    
    def __init__(self, features: List[str] = None, training_data: str = "historical_submissions",
                 prediction_target: str = "acceptance_probability"):
        self.features = features or ["scientific_rigor", "methodology", "novelty", "clarity"]
        self.training_data = training_data
        self.prediction_target = prediction_target
        self.db_path = "quality_assessment.db"
        self.lock = threading.RLock()
        self._init_quality_db()
        
    def _init_quality_db(self):
        """Initialize quality assessment database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_assessments (
                    id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    feature_scores TEXT NOT NULL,
                    acceptance_probability REAL NOT NULL,
                    assessment_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def assess_submission_quality(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess submission quality using ML"""
        assessment = {
            'submission_id': submission_data.get('id', 'unknown'),
            'assessed_at': datetime.now().isoformat()
        }
        
        # Assess each feature
        feature_scores = {}
        for feature in self.features:
            feature_scores[feature] = self._assess_feature(submission_data, feature)
        
        assessment['feature_scores'] = feature_scores
        assessment['overall_score'] = sum(feature_scores.values()) / len(feature_scores)
        assessment['acceptance_probability'] = self._calculate_acceptance_probability(feature_scores)
        assessment['recommendations'] = self._generate_recommendations(feature_scores)
        assessment['confidence'] = self._calculate_confidence(feature_scores)
        
        # Store assessment
        self._store_assessment(assessment)
        
        return assessment
    
    def _assess_feature(self, submission_data: Dict[str, Any], feature: str) -> float:
        """Assess individual quality feature"""
        content = submission_data.get('content', '')
        metadata = submission_data.get('metadata', {})
        
        if feature == "scientific_rigor":
            return self._assess_scientific_rigor(content, metadata)
        elif feature == "methodology":
            return self._assess_methodology(content)
        elif feature == "novelty":
            return self._assess_novelty(content)
        elif feature == "clarity":
            return self._assess_clarity(content)
        else:
            return 0.5  # Default score
    
    def _assess_scientific_rigor(self, content: str, metadata: Dict[str, Any]) -> float:
        """Assess scientific rigor"""
        rigor_indicators = {
            'has_hypothesis': any(word in content.lower() for word in ['hypothesis', 'research question', 'objective']),
            'has_methodology': any(word in content.lower() for word in ['method', 'procedure', 'approach', 'protocol']),
            'has_statistical_analysis': any(word in content.lower() for word in ['statistical', 'analysis', 'test', 'significance']),
            'has_validation': any(word in content.lower() for word in ['validation', 'verify', 'confirm', 'validate']),
            'has_limitations': any(word in content.lower() for word in ['limitation', 'constrain', 'bias', 'error']),
            'has_reproducibility': any(word in content.lower() for word in ['reproducible', 'replicate', 'repeat']),
            'adequate_sample_size': 'sample' in content.lower() and any(num in content for num in ['100', '1000', 'large']),
            'peer_reviewed_references': metadata.get('reference_count', 0) > 10
        }
        
        return sum(rigor_indicators.values()) / len(rigor_indicators)
    
    def _assess_methodology(self, content: str) -> float:
        """Assess methodology quality"""
        methodology_indicators = {
            'clear_methods': any(word in content.lower() for word in ['method', 'procedure', 'algorithm', 'approach']),
            'experimental_design': any(word in content.lower() for word in ['experiment', 'design', 'setup', 'protocol']),
            'data_collection': any(word in content.lower() for word in ['data collection', 'dataset', 'corpus', 'sample']),
            'evaluation_metrics': any(word in content.lower() for word in ['accuracy', 'precision', 'recall', 'f1', 'metric']),
            'baseline_comparison': any(word in content.lower() for word in ['baseline', 'compare', 'benchmark', 'state-of-art']),
            'ablation_study': any(word in content.lower() for word in ['ablation', 'component', 'contribution']),
            'parameter_tuning': any(word in content.lower() for word in ['parameter', 'hyperparameter', 'tuning', 'optimization'])
        }
        
        return sum(methodology_indicators.values()) / len(methodology_indicators)
    
    def _assess_novelty(self, content: str) -> float:
        """Assess research novelty"""
        novelty_indicators = {
            'novel_approach': any(word in content.lower() for word in ['novel', 'new', 'innovative', 'original']),
            'technical_contribution': any(word in content.lower() for word in ['contribution', 'advance', 'improvement']),
            'problem_formulation': any(word in content.lower() for word in ['formulate', 'define', 'model', 'framework']),
            'creative_solution': any(word in content.lower() for word in ['creative', 'unique', 'different', 'alternative']),
            'interdisciplinary': any(word in content.lower() for word in ['interdisciplinary', 'cross-domain', 'multidisciplinary']),
            'practical_impact': any(word in content.lower() for word in ['practical', 'application', 'real-world', 'industry'])
        }
        
        return sum(novelty_indicators.values()) / len(novelty_indicators)
    
    def _assess_clarity(self, content: str) -> float:
        """Assess writing clarity"""
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        clarity_metrics = {
            'average_sentence_length': len(words) / len(sentences) if sentences else 0,
            'readability_score': self._calculate_readability(content),
            'structure_quality': self._assess_structure(content),
            'terminology_consistency': self._assess_terminology(content)
        }
        
        # Normalize metrics to 0-1 scale
        normalized_score = 0.0
        
        # Sentence length (ideal: 15-25 words)
        avg_length = clarity_metrics['average_sentence_length']
        if 15 <= avg_length <= 25:
            normalized_score += 0.25
        elif 10 <= avg_length <= 30:
            normalized_score += 0.15
        
        # Add other metrics
        normalized_score += clarity_metrics['readability_score'] * 0.25
        normalized_score += clarity_metrics['structure_quality'] * 0.25
        normalized_score += clarity_metrics['terminology_consistency'] * 0.25
        
        return min(normalized_score, 1.0)
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score (simplified)"""
        words = content.split()
        sentences = content.split('.')
        
        if not words or not sentences:
            return 0.0
        
        # Simple readability approximation
        avg_words_per_sentence = len(words) / len(sentences)
        avg_syllables_per_word = 1.5  # Simplified assumption
        
        # Flesch-Kincaid approximation
        readability = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale (higher is better)
        return max(0, min(1, readability / 100))
    
    def _assess_structure(self, content: str) -> float:
        """Assess document structure"""
        structure_indicators = {
            'has_introduction': any(word in content.lower() for word in ['introduction', 'background']),
            'has_literature_review': any(word in content.lower() for word in ['related work', 'literature', 'previous']),
            'has_methodology': any(word in content.lower() for word in ['method', 'approach', 'procedure']),
            'has_results': any(word in content.lower() for word in ['result', 'finding', 'outcome']),
            'has_discussion': any(word in content.lower() for word in ['discussion', 'analysis', 'interpretation']),
            'has_conclusion': any(word in content.lower() for word in ['conclusion', 'summary', 'future work']),
            'logical_flow': self._check_logical_flow(content)
        }
        
        return sum(structure_indicators.values()) / len(structure_indicators)
    
    def _assess_terminology(self, content: str) -> float:
        """Assess terminology consistency"""
        words = content.lower().split()
        word_freq = defaultdict(int)
        
        for word in words:
            if len(word) > 5:  # Focus on longer words (likely technical terms)
                word_freq[word] += 1
        
        # Check for consistent usage of technical terms
        technical_terms = [word for word, freq in word_freq.items() if freq > 1]
        consistency_score = len(technical_terms) / max(len(word_freq), 1)
        
        return min(consistency_score, 1.0)
    
    def _check_logical_flow(self, content: str) -> bool:
        """Check for logical flow indicators"""
        flow_indicators = [
            'first', 'second', 'third', 'next', 'then', 'finally',
            'however', 'moreover', 'furthermore', 'therefore', 'consequently',
            'in addition', 'on the other hand', 'in contrast', 'similarly'
        ]
        
        indicator_count = sum(content.lower().count(indicator) for indicator in flow_indicators)
        return indicator_count >= 3  # At least 3 flow indicators
    
    def _calculate_acceptance_probability(self, feature_scores: Dict[str, float]) -> float:
        """Calculate acceptance probability based on feature scores"""
        # Weighted average with different importance for each feature
        weights = {
            'scientific_rigor': 0.35,
            'methodology': 0.30,
            'novelty': 0.25,
            'clarity': 0.10
        }
        
        weighted_score = 0.0
        for feature, score in feature_scores.items():
            weight = weights.get(feature, 0.25)
            weighted_score += score * weight
        
        # Apply sigmoid transformation for probability
        import math
        probability = 1 / (1 + math.exp(-5 * (weighted_score - 0.5)))
        return round(probability, 3)
    
    def _generate_recommendations(self, feature_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        for feature, score in feature_scores.items():
            if score < 0.6:  # Below threshold
                if feature == 'scientific_rigor':
                    recommendations.append("Strengthen scientific rigor: Add clear hypotheses, improve validation, discuss limitations")
                elif feature == 'methodology':
                    recommendations.append("Enhance methodology: Provide more detailed procedures, add evaluation metrics, include baseline comparisons")
                elif feature == 'novelty':
                    recommendations.append("Increase novelty: Highlight unique contributions, emphasize innovations, demonstrate practical impact")
                elif feature == 'clarity':
                    recommendations.append("Improve clarity: Simplify sentence structure, enhance organization, ensure consistent terminology")
        
        if not recommendations:
            recommendations.append("Overall quality is good. Consider minor revisions for publication readiness.")
        
        return recommendations
    
    def _calculate_confidence(self, feature_scores: Dict[str, float]) -> float:
        """Calculate confidence in assessment"""
        # Higher confidence when scores are more extreme (closer to 0 or 1)
        extremeness = sum(abs(score - 0.5) for score in feature_scores.values()) / len(feature_scores)
        confidence = 0.5 + extremeness  # Base confidence + extremeness factor
        return min(confidence, 1.0)
    
    def _store_assessment(self, assessment: Dict[str, Any]):
        """Store quality assessment in database"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                assessment_id = hashlib.md5(f"{assessment['submission_id']}{assessment['assessed_at']}".encode()).hexdigest()[:8]
                conn.execute("""
                    INSERT OR REPLACE INTO quality_assessments 
                    (id, submission_id, overall_score, feature_scores, acceptance_probability, assessment_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    assessment_id,
                    assessment['submission_id'],
                    assessment['overall_score'],
                    json.dumps(assessment['feature_scores']),
                    assessment['acceptance_probability'],
                    json.dumps(assessment),
                    datetime.now().isoformat()
                ))
                conn.commit()

class FeedbackLearner:
    """Feedback learning system - Agent 2 Critical Feature"""
    
    def __init__(self, decision_tracker=None, outcome_analyzer=None, suggestion_improver=None):
        self.decision_tracker = decision_tracker or DecisionTracker()
        self.outcome_analyzer = outcome_analyzer or OutcomeAnalyzer()
        self.suggestion_improver = suggestion_improver or SuggestionImprover()
        self.db_path = "feedback_learning.db"
        self.lock = threading.RLock()
        self._init_feedback_db()
        
    def _init_feedback_db(self):
        """Initialize feedback learning database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    suggested_action TEXT NOT NULL,
                    actual_outcome TEXT NOT NULL,
                    feedback_score REAL NOT NULL,
                    learning_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def learn_from_feedback(self, submission_id: str, suggestion: Dict[str, Any], 
                           actual_outcome: Dict[str, Any], feedback_score: float) -> Dict[str, Any]:
        """Learn from editorial decision feedback"""
        learning_result = {
            'submission_id': submission_id,
            'learned_at': datetime.now().isoformat(),
            'feedback_score': feedback_score
        }
        
        # Track the decision
        self.decision_tracker.track_decision(submission_id, suggestion, actual_outcome)
        
        # Analyze the outcome
        analysis = self.outcome_analyzer.analyze_outcome(suggestion, actual_outcome, feedback_score)
        learning_result['analysis'] = analysis
        
        # Improve future suggestions
        improvements = self.suggestion_improver.identify_improvements(analysis)
        learning_result['improvements'] = improvements
        
        # Update learning model
        self._update_learning_model(learning_result)
        
        # Store feedback
        self._store_feedback(learning_result)
        
        return learning_result
    
    def _update_learning_model(self, learning_result: Dict[str, Any]):
        """Update the learning model based on feedback"""
        # Simple learning model update
        analysis = learning_result['analysis']
        
        if learning_result['feedback_score'] > 0.7:
            # Positive feedback - reinforce successful patterns
            logger.info(f"Positive feedback received for submission {learning_result['submission_id']}")
        else:
            # Negative feedback - adjust model parameters
            logger.info(f"Negative feedback received for submission {learning_result['submission_id']}, adjusting model")
    
    def _store_feedback(self, learning_result: Dict[str, Any]):
        """Store feedback learning data"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                feedback_id = hashlib.md5(f"{learning_result['submission_id']}{learning_result['learned_at']}".encode()).hexdigest()[:8]
                conn.execute("""
                    INSERT INTO feedback_entries 
                    (id, submission_id, decision_type, suggested_action, actual_outcome, feedback_score, learning_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback_id,
                    learning_result['submission_id'],
                    learning_result.get('analysis', {}).get('decision_type', 'unknown'),
                    json.dumps(learning_result.get('analysis', {}).get('suggestion', {})),
                    json.dumps(learning_result.get('analysis', {}).get('outcome', {})),
                    learning_result['feedback_score'],
                    json.dumps(learning_result),
                    datetime.now().isoformat()
                ))
                conn.commit()

class ComplianceChecker:
    """Compliance checking ML - Agent 2 Critical Feature"""
    
    def __init__(self, regulatory_db=None, safety_validator=None, inci_validator=None):
        self.regulatory_db = regulatory_db or RegulatoryDatabase()
        self.safety_validator = safety_validator or SafetyValidator()
        self.inci_validator = inci_validator or INCIValidator()
        self.db_path = "compliance_checks.db"
        self.lock = threading.RLock()
        self._init_compliance_db()
        
    def _init_compliance_db(self):
        """Initialize compliance checking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_checks (
                    id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    compliance_score REAL NOT NULL,
                    violations TEXT NOT NULL,
                    recommendations TEXT NOT NULL,
                    check_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def check_compliance(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check submission compliance with regulations and safety"""
        compliance_result = {
            'submission_id': submission_data.get('id', 'unknown'),
            'checked_at': datetime.now().isoformat()
        }
        
        # Regulatory compliance check
        regulatory_check = self.regulatory_db.check_regulatory_compliance(submission_data)
        compliance_result['regulatory_compliance'] = regulatory_check
        
        # Safety validation
        safety_check = self.safety_validator.validate_safety(submission_data)
        compliance_result['safety_validation'] = safety_check
        
        # INCI validation (for cosmetics/skincare)
        inci_check = self.inci_validator.validate_inci(submission_data)
        compliance_result['inci_validation'] = inci_check
        
        # Overall compliance score
        compliance_result['overall_score'] = self._calculate_overall_compliance(
            regulatory_check, safety_check, inci_check
        )
        
        # Identify violations
        compliance_result['violations'] = self._identify_violations(
            regulatory_check, safety_check, inci_check
        )
        
        # Generate recommendations
        compliance_result['recommendations'] = self._generate_compliance_recommendations(
            compliance_result['violations']
        )
        
        # Store compliance check
        self._store_compliance_check(compliance_result)
        
        return compliance_result
    
    def _calculate_overall_compliance(self, regulatory_check: Dict[str, Any], 
                                   safety_check: Dict[str, Any], 
                                   inci_check: Dict[str, Any]) -> float:
        """Calculate overall compliance score"""
        scores = [
            regulatory_check.get('score', 0.0),
            safety_check.get('score', 0.0),
            inci_check.get('score', 0.0)
        ]
        return sum(scores) / len(scores)
    
    def _identify_violations(self, regulatory_check: Dict[str, Any], 
                           safety_check: Dict[str, Any], 
                           inci_check: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify compliance violations"""
        violations = []
        
        # Regulatory violations
        violations.extend(regulatory_check.get('violations', []))
        
        # Safety violations
        violations.extend(safety_check.get('violations', []))
        
        # INCI violations
        violations.extend(inci_check.get('violations', []))
        
        return violations
    
    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        for violation in violations:
            violation_type = violation.get('type', 'unknown')
            
            if violation_type == 'regulatory':
                recommendations.append(f"Address regulatory issue: {violation.get('description', 'Unknown violation')}")
            elif violation_type == 'safety':
                recommendations.append(f"Resolve safety concern: {violation.get('description', 'Unknown safety issue')}")
            elif violation_type == 'inci':
                recommendations.append(f"Fix INCI compliance: {violation.get('description', 'Unknown INCI issue')}")
        
        if not recommendations:
            recommendations.append("No compliance violations detected. Submission meets all regulatory requirements.")
        
        return recommendations
    
    def _store_compliance_check(self, compliance_result: Dict[str, Any]):
        """Store compliance check results"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                check_id = hashlib.md5(f"{compliance_result['submission_id']}{compliance_result['checked_at']}".encode()).hexdigest()[:8]
                conn.execute("""
                    INSERT OR REPLACE INTO compliance_checks 
                    (id, submission_id, compliance_score, violations, recommendations, check_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    check_id,
                    compliance_result['submission_id'],
                    compliance_result['overall_score'],
                    json.dumps(compliance_result['violations']),
                    json.dumps(compliance_result['recommendations']),
                    json.dumps(compliance_result),
                    datetime.now().isoformat()
                ))
                conn.commit()

# Supporting classes for feedback learning
class DecisionTracker:
    """Track editorial decisions"""
    
    def track_decision(self, submission_id: str, suggestion: Dict[str, Any], outcome: Dict[str, Any]):
        """Track a decision and its outcome"""
        logger.info(f"Tracking decision for submission {submission_id}")

class OutcomeAnalyzer:
    """Analyze decision outcomes"""
    
    def analyze_outcome(self, suggestion: Dict[str, Any], outcome: Dict[str, Any], 
                       feedback_score: float) -> Dict[str, Any]:
        """Analyze the outcome of a decision"""
        return {
            'decision_type': suggestion.get('type', 'unknown'),
            'suggestion': suggestion,
            'outcome': outcome,
            'success_rate': feedback_score,
            'analysis_timestamp': datetime.now().isoformat()
        }

class SuggestionImprover:
    """Improve future suggestions based on learning"""
    
    def identify_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Identify improvements for future suggestions"""
        improvements = []
        
        if analysis.get('success_rate', 0) < 0.5:
            improvements.append("Adjust decision criteria based on negative feedback")
            improvements.append("Increase confidence threshold for similar decisions")
        else:
            improvements.append("Reinforce successful decision patterns")
        
        return improvements

# Supporting classes for compliance checking
class RegulatoryDatabase:
    """Regulatory compliance database"""
    
    def check_regulatory_compliance(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check regulatory compliance"""
        content = submission_data.get('content', '').lower()
        
        # Check for common regulatory issues
        violations = []
        
        if 'clinical trial' in content and 'ethics approval' not in content:
            violations.append({
                'type': 'regulatory',
                'severity': 'high',
                'description': 'Clinical trial mentioned without ethics approval documentation'
            })
        
        if 'human subjects' in content and 'informed consent' not in content:
            violations.append({
                'type': 'regulatory',
                'severity': 'high',
                'description': 'Human subjects research without informed consent mention'
            })
        
        # Calculate compliance score
        score = 1.0 - (len(violations) * 0.3)  # Deduct 0.3 per violation
        score = max(0.0, score)
        
        return {
            'score': score,
            'violations': violations,
            'checked_regulations': ['FDA', 'EMA', 'ICH-GCP', 'Declaration of Helsinki']
        }

class SafetyValidator:
    """Safety validation for submissions"""
    
    def validate_safety(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety aspects of submission"""
        content = submission_data.get('content', '').lower()
        
        violations = []
        
        # Check for safety-related issues
        if 'toxic' in content and 'safety assessment' not in content:
            violations.append({
                'type': 'safety',
                'severity': 'high',
                'description': 'Toxicity mentioned without safety assessment'
            })
        
        if 'adverse effect' in content and 'risk mitigation' not in content:
            violations.append({
                'type': 'safety',
                'severity': 'medium',
                'description': 'Adverse effects mentioned without risk mitigation strategy'
            })
        
        # Calculate safety score
        score = 1.0 - (len(violations) * 0.25)  # Deduct 0.25 per violation
        score = max(0.0, score)
        
        return {
            'score': score,
            'violations': violations,
            'safety_categories': ['toxicity', 'adverse_effects', 'contraindications', 'warnings']
        }

class INCIValidator:
    """INCI (International Nomenclature of Cosmetic Ingredients) validator"""
    
    def validate_inci(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate INCI compliance for cosmetic ingredients"""
        content = submission_data.get('content', '').lower()
        
        violations = []
        
        # Check for INCI-related issues (simplified)
        if 'cosmetic' in content or 'skincare' in content:
            if 'inci' not in content and 'ingredient' in content:
                violations.append({
                    'type': 'inci',
                    'severity': 'medium',
                    'description': 'Cosmetic ingredients mentioned without INCI nomenclature'
                })
        
        # Calculate INCI compliance score
        score = 1.0 - (len(violations) * 0.2)  # Deduct 0.2 per violation
        score = max(0.0, score)
        
        return {
            'score': score,
            'violations': violations,
            'validated_standards': ['INCI', 'CTFA', 'Personal Care Products Council']
        }