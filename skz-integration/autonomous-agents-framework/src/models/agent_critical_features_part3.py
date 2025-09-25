"""
Critical ML Features for Autonomous Agents - Part 3
Implements the urgent requirements from URGENT_AGENT_FEATURES.md
Agents 4-7 Critical Features
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
# Agent 4: Peer Review Coordination - Critical Features
# =============================================================================

class ReviewerMatcher:
    """Reviewer matching ML - Agent 4 Critical Feature"""
    
    def __init__(self, expertise_analyzer=None, workload_optimizer=None, quality_predictor=None):
        self.expertise_analyzer = expertise_analyzer or ExpertiseAnalyzer()
        self.workload_optimizer = workload_optimizer or WorkloadOptimizer()
        self.quality_predictor = quality_predictor or ReviewQualityPredictor()
        self.db_path = "reviewer_matching.db"
        self.lock = threading.RLock()
        self._init_reviewer_db()
        
    def _init_reviewer_db(self):
        """Initialize reviewer matching database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviewer_matches (
                    id TEXT PRIMARY KEY,
                    submission_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    match_score REAL NOT NULL,
                    expertise_match REAL NOT NULL,
                    workload_score REAL NOT NULL,
                    quality_prediction REAL NOT NULL,
                    match_data TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def match_reviewers(self, submission_data: Dict[str, Any], 
                       available_reviewers: List[Dict[str, Any]], 
                       num_reviewers: int = 3) -> List[Dict[str, Any]]:
        """Match optimal reviewers to submission using ML"""
        matching_result = {
            'submission_id': submission_data.get('id', 'unknown'),
            'matched_at': datetime.now().isoformat(),
            'requested_reviewers': num_reviewers
        }
        
        reviewer_scores = []
        
        for reviewer in available_reviewers:
            # Analyze expertise match
            expertise_match = self.expertise_analyzer.calculate_expertise_match(
                submission_data, reviewer
            )
            
            # Check workload optimization
            workload_score = self.workload_optimizer.calculate_workload_score(reviewer)
            
            # Predict review quality
            quality_prediction = self.quality_predictor.predict_review_quality(
                submission_data, reviewer
            )
            
            # Calculate overall match score
            overall_score = self._calculate_match_score(
                expertise_match, workload_score, quality_prediction
            )
            
            reviewer_score = {
                'reviewer_id': reviewer.get('id', 'unknown'),
                'reviewer_name': reviewer.get('name', 'Unknown'),
                'match_score': overall_score,
                'expertise_match': expertise_match,
                'workload_score': workload_score,
                'quality_prediction': quality_prediction['predicted_score'],
                'specializations': reviewer.get('specializations', []),
                'availability': reviewer.get('availability', 'unknown'),
                'recent_performance': reviewer.get('recent_performance', 0.8)
            }
            
            reviewer_scores.append(reviewer_score)
        
        # Sort by match score and select top reviewers
        reviewer_scores.sort(key=lambda x: x['match_score'], reverse=True)
        selected_reviewers = reviewer_scores[:num_reviewers]
        
        matching_result['selected_reviewers'] = selected_reviewers
        matching_result['all_candidate_scores'] = reviewer_scores
        matching_result['selection_criteria'] = self._generate_selection_criteria()
        
        # Store matching results
        self._store_matching_results(matching_result)
        
        return selected_reviewers
    
    def _calculate_match_score(self, expertise_match: float, workload_score: float, 
                             quality_prediction: Dict[str, Any]) -> float:
        """Calculate overall reviewer match score"""
        # Weighted combination of factors
        weights = {
            'expertise': 0.4,
            'workload': 0.3,
            'quality': 0.3
        }
        
        quality_score = quality_prediction.get('predicted_score', 0.5)
        
        overall_score = (
            expertise_match * weights['expertise'] +
            workload_score * weights['workload'] +
            quality_score * weights['quality']
        )
        
        return round(overall_score, 3)
    
    def _generate_selection_criteria(self) -> Dict[str, Any]:
        """Generate selection criteria used"""
        return {
            'expertise_weight': 0.4,
            'workload_weight': 0.3,
            'quality_weight': 0.3,
            'minimum_expertise_threshold': 0.6,
            'maximum_workload_threshold': 0.8,
            'minimum_quality_threshold': 0.5
        }
    
    def _store_matching_results(self, matching_result: Dict[str, Any]):
        """Store reviewer matching results"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                for reviewer in matching_result['selected_reviewers']:
                    match_id = hashlib.md5(f"{matching_result['submission_id']}{reviewer['reviewer_id']}{matching_result['matched_at']}".encode()).hexdigest()[:8]
                    conn.execute("""
                        INSERT OR REPLACE INTO reviewer_matches 
                        (id, submission_id, reviewer_id, match_score, expertise_match, workload_score, quality_prediction, match_data, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        match_id,
                        matching_result['submission_id'],
                        reviewer['reviewer_id'],
                        reviewer['match_score'],
                        reviewer['expertise_match'],
                        reviewer['workload_score'],
                        reviewer['quality_prediction'],
                        json.dumps(reviewer),
                        datetime.now().isoformat()
                    ))
                conn.commit()

class ReviewQualityPredictor:
    """Review quality prediction - Agent 4 Critical Feature"""
    
    def __init__(self, reviewer_profiler=None, manuscript_analyzer=None, interaction_predictor=None):
        self.reviewer_profiler = reviewer_profiler or ReviewerProfiler()
        self.manuscript_analyzer = manuscript_analyzer or ManuscriptAnalyzer()
        self.interaction_predictor = interaction_predictor or InteractionPredictor()
        
    def predict_review_quality(self, submission_data: Dict[str, Any], 
                             reviewer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict quality of review from specific reviewer"""
        
        # Profile reviewer capabilities
        reviewer_profile = self.reviewer_profiler.create_profile(reviewer_data)
        
        # Analyze manuscript complexity
        manuscript_analysis = self.manuscript_analyzer.analyze_complexity(submission_data)
        
        # Predict reviewer-manuscript interaction
        interaction_prediction = self.interaction_predictor.predict_interaction(
            reviewer_profile, manuscript_analysis
        )
        
        # Calculate predicted quality score
        quality_prediction = self._calculate_quality_prediction(
            reviewer_profile, manuscript_analysis, interaction_prediction
        )
        
        return {
            'predicted_score': quality_prediction,
            'confidence': self._calculate_confidence(reviewer_profile, manuscript_analysis),
            'quality_factors': {
                'reviewer_expertise': reviewer_profile.get('expertise_level', 0.5),
                'manuscript_complexity': manuscript_analysis.get('complexity_score', 0.5),
                'match_quality': interaction_prediction.get('match_score', 0.5)
            },
            'predicted_review_time': self._predict_review_time(reviewer_profile, manuscript_analysis),
            'potential_issues': self._identify_potential_issues(reviewer_profile, manuscript_analysis)
        }
    
    def _calculate_quality_prediction(self, reviewer_profile: Dict[str, Any], 
                                    manuscript_analysis: Dict[str, Any], 
                                    interaction_prediction: Dict[str, Any]) -> float:
        """Calculate predicted review quality score"""
        
        # Factor weights
        weights = {
            'reviewer_expertise': 0.35,
            'reviewer_experience': 0.25,
            'manuscript_match': 0.25,
            'reviewer_reliability': 0.15
        }
        
        # Extract scores
        expertise_score = reviewer_profile.get('expertise_level', 0.5)
        experience_score = reviewer_profile.get('experience_score', 0.5)
        match_score = interaction_prediction.get('match_score', 0.5)
        reliability_score = reviewer_profile.get('reliability_score', 0.8)
        
        # Calculate weighted score
        quality_score = (
            expertise_score * weights['reviewer_expertise'] +
            experience_score * weights['reviewer_experience'] +
            match_score * weights['manuscript_match'] +
            reliability_score * weights['reviewer_reliability']
        )
        
        return round(quality_score, 3)
    
    def _calculate_confidence(self, reviewer_profile: Dict[str, Any], 
                            manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in quality prediction"""
        
        # Higher confidence when we have more data about reviewer
        reviewer_data_completeness = len([v for v in reviewer_profile.values() if v is not None]) / len(reviewer_profile)
        
        # Higher confidence for well-matched expertise
        expertise_match = reviewer_profile.get('expertise_level', 0.5)
        
        # Lower confidence for highly complex manuscripts
        complexity_penalty = 1.0 - manuscript_analysis.get('complexity_score', 0.5) * 0.3
        
        confidence = (reviewer_data_completeness * 0.4 + expertise_match * 0.4 + complexity_penalty * 0.2)
        
        return round(confidence, 3)
    
    def _predict_review_time(self, reviewer_profile: Dict[str, Any], 
                           manuscript_analysis: Dict[str, Any]) -> str:
        """Predict review completion time"""
        
        base_time = 14  # days
        
        # Adjust based on reviewer speed
        reviewer_speed = reviewer_profile.get('average_review_time', 14)
        
        # Adjust based on manuscript complexity
        complexity_multiplier = 1 + manuscript_analysis.get('complexity_score', 0.5) * 0.5
        
        # Adjust based on reviewer current workload
        workload_multiplier = 1 + reviewer_profile.get('current_workload', 0.5) * 0.3
        
        predicted_time = int(reviewer_speed * complexity_multiplier * workload_multiplier)
        
        if predicted_time <= 7:
            return "1 week"
        elif predicted_time <= 14:
            return "2 weeks"
        elif predicted_time <= 21:
            return "3 weeks"
        else:
            return "4+ weeks"
    
    def _identify_potential_issues(self, reviewer_profile: Dict[str, Any], 
                                 manuscript_analysis: Dict[str, Any]) -> List[str]:
        """Identify potential review issues"""
        issues = []
        
        # Expertise mismatch
        if reviewer_profile.get('expertise_level', 0.5) < 0.6:
            issues.append("Limited expertise in manuscript topic")
        
        # High workload
        if reviewer_profile.get('current_workload', 0.5) > 0.8:
            issues.append("Reviewer has high current workload")
        
        # Complex manuscript
        if manuscript_analysis.get('complexity_score', 0.5) > 0.8:
            issues.append("Manuscript complexity may challenge reviewer")
        
        # Reliability concerns
        if reviewer_profile.get('reliability_score', 0.8) < 0.6:
            issues.append("Reviewer has history of delayed reviews")
        
        return issues

class WorkloadOptimizer:
    """Workload optimization - Agent 4 Critical Feature"""
    
    def __init__(self, capacity_analyzer=None, assignment_optimizer=None, timeline_predictor=None):
        self.capacity_analyzer = capacity_analyzer or CapacityAnalyzer()
        self.assignment_optimizer = assignment_optimizer or AssignmentOptimizer()
        self.timeline_predictor = timeline_predictor or TimelinePredictor()
        self.db_path = "workload_optimization.db"
        self.lock = threading.RLock()
        self._init_workload_db()
        
    def _init_workload_db(self):
        """Initialize workload optimization database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workload_optimizations (
                    id TEXT PRIMARY KEY,
                    reviewer_id TEXT NOT NULL,
                    current_workload REAL NOT NULL,
                    optimal_workload REAL NOT NULL,
                    capacity_utilization REAL NOT NULL,
                    optimization_recommendations TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def optimize_workload_distribution(self, reviewers: List[Dict[str, Any]], 
                                     pending_reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize workload distribution across reviewers"""
        
        optimization_result = {
            'optimized_at': datetime.now().isoformat(),
            'total_reviewers': len(reviewers),
            'total_pending_reviews': len(pending_reviews)
        }
        
        # Analyze current capacity for each reviewer
        reviewer_capacities = []
        for reviewer in reviewers:
            capacity = self.capacity_analyzer.analyze_capacity(reviewer)
            reviewer_capacities.append({
                'reviewer_id': reviewer.get('id'),
                'current_capacity': capacity,
                'reviewer_data': reviewer
            })
        
        # Optimize assignments
        optimized_assignments = self.assignment_optimizer.optimize_assignments(
            reviewer_capacities, pending_reviews
        )
        optimization_result['optimized_assignments'] = optimized_assignments
        
        # Predict timelines
        timeline_predictions = self.timeline_predictor.predict_completion_times(
            optimized_assignments
        )
        optimization_result['timeline_predictions'] = timeline_predictions
        
        # Generate workload recommendations
        optimization_result['recommendations'] = self._generate_workload_recommendations(
            reviewer_capacities, optimized_assignments
        )
        
        # Calculate optimization metrics
        optimization_result['metrics'] = self._calculate_optimization_metrics(
            reviewer_capacities, optimized_assignments
        )
        
        # Store optimization results
        self._store_workload_optimization(optimization_result)
        
        return optimization_result
    
    def calculate_workload_score(self, reviewer_data: Dict[str, Any]) -> float:
        """Calculate workload score for reviewer (0 = overloaded, 1 = available)"""
        
        # Current active reviews
        active_reviews = reviewer_data.get('active_reviews', 0)
        max_capacity = reviewer_data.get('max_capacity', 5)  # Default max 5 reviews
        
        # Recent review completion rate
        completion_rate = reviewer_data.get('completion_rate', 0.8)
        
        # Time since last assignment
        days_since_last = reviewer_data.get('days_since_last_assignment', 7)
        
        # Calculate base workload score
        capacity_score = max(0, (max_capacity - active_reviews) / max_capacity)
        
        # Adjust for completion rate
        reliability_adjustment = completion_rate * 0.2
        
        # Adjust for recency (slightly favor reviewers who haven't had recent assignments)
        recency_adjustment = min(days_since_last / 30, 0.1)  # Max 0.1 bonus for 30+ days
        
        workload_score = capacity_score + reliability_adjustment + recency_adjustment
        
        return round(min(workload_score, 1.0), 3)
    
    def _generate_workload_recommendations(self, reviewer_capacities: List[Dict[str, Any]], 
                                         optimized_assignments: List[Dict[str, Any]]) -> List[str]:
        """Generate workload optimization recommendations"""
        recommendations = []
        
        # Check for overloaded reviewers
        overloaded_reviewers = [
            r for r in reviewer_capacities 
            if r['current_capacity']['utilization_rate'] > 0.9
        ]
        
        if overloaded_reviewers:
            recommendations.append(f"Redistribute workload from {len(overloaded_reviewers)} overloaded reviewers")
        
        # Check for underutilized reviewers
        underutilized_reviewers = [
            r for r in reviewer_capacities 
            if r['current_capacity']['utilization_rate'] < 0.3
        ]
        
        if underutilized_reviewers:
            recommendations.append(f"Increase assignments for {len(underutilized_reviewers)} underutilized reviewers")
        
        # Check for timeline issues
        urgent_reviews = [
            a for a in optimized_assignments 
            if a.get('urgency_level', 'normal') == 'urgent'
        ]
        
        if urgent_reviews:
            recommendations.append(f"Prioritize {len(urgent_reviews)} urgent reviews with fast reviewers")
        
        if not recommendations:
            recommendations.append("Workload distribution is optimal")
        
        return recommendations
    
    def _calculate_optimization_metrics(self, reviewer_capacities: List[Dict[str, Any]], 
                                      optimized_assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate workload optimization metrics"""
        
        utilization_rates = [r['current_capacity']['utilization_rate'] for r in reviewer_capacities]
        
        return {
            'average_utilization': np.mean(utilization_rates),
            'utilization_std': np.std(utilization_rates),
            'optimal_distribution_score': 1.0 - np.std(utilization_rates),  # Lower std = better distribution
            'total_assignments_optimized': len(optimized_assignments),
            'load_balancing_efficiency': self._calculate_load_balancing_efficiency(utilization_rates)
        }
    
    def _calculate_load_balancing_efficiency(self, utilization_rates: List[float]) -> float:
        """Calculate load balancing efficiency"""
        if not utilization_rates:
            return 0.0
        
        # Ideal utilization is around 0.7 (70%)
        ideal_utilization = 0.7
        deviations = [abs(rate - ideal_utilization) for rate in utilization_rates]
        average_deviation = np.mean(deviations)
        
        # Convert to efficiency score (lower deviation = higher efficiency)
        efficiency = max(0, 1.0 - average_deviation / ideal_utilization)
        
        return round(efficiency, 3)
    
    def _store_workload_optimization(self, optimization_result: Dict[str, Any]):
        """Store workload optimization results"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                for assignment in optimization_result.get('optimized_assignments', []):
                    opt_id = hashlib.md5(f"{assignment.get('reviewer_id', 'unknown')}{optimization_result['optimized_at']}".encode()).hexdigest()[:8]
                    conn.execute("""
                        INSERT OR REPLACE INTO workload_optimizations 
                        (id, reviewer_id, current_workload, optimal_workload, capacity_utilization, optimization_recommendations, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        opt_id,
                        assignment.get('reviewer_id', 'unknown'),
                        assignment.get('current_workload', 0.0),
                        assignment.get('optimal_workload', 0.0),
                        assignment.get('capacity_utilization', 0.0),
                        json.dumps(optimization_result.get('recommendations', [])),
                        datetime.now().isoformat()
                    ))
                conn.commit()

# Supporting classes for Agent 4
class ExpertiseAnalyzer:
    """Analyze reviewer expertise match"""
    
    def calculate_expertise_match(self, submission_data: Dict[str, Any], 
                                reviewer_data: Dict[str, Any]) -> float:
        """Calculate expertise match between submission and reviewer"""
        
        # Extract submission keywords/topics
        submission_content = submission_data.get('content', '').lower()
        submission_keywords = submission_data.get('keywords', [])
        
        # Extract reviewer specializations
        reviewer_specializations = reviewer_data.get('specializations', [])
        reviewer_keywords = reviewer_data.get('expertise_keywords', [])
        
        # Calculate keyword overlap
        all_submission_terms = set(submission_keywords + self._extract_key_terms(submission_content))
        all_reviewer_terms = set(reviewer_specializations + reviewer_keywords)
        
        if not all_submission_terms or not all_reviewer_terms:
            return 0.3  # Default low match if no data
        
        # Calculate Jaccard similarity
        intersection = len(all_submission_terms.intersection(all_reviewer_terms))
        union = len(all_submission_terms.union(all_reviewer_terms))
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        
        # Boost score if reviewer has high expertise in any matching area
        expertise_boost = 0.0
        for specialization in reviewer_specializations:
            if any(term in specialization.lower() for term in all_submission_terms):
                expertise_level = reviewer_data.get('expertise_levels', {}).get(specialization, 0.5)
                expertise_boost = max(expertise_boost, expertise_level * 0.3)
        
        final_score = min(jaccard_similarity + expertise_boost, 1.0)
        
        return round(final_score, 3)
    
    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content"""
        # Simple key term extraction
        important_terms = []
        
        # Look for technical terms (words with capitals in middle)
        technical_pattern = r'\b[a-z]*[A-Z][a-z]*\b'
        technical_terms = re.findall(technical_pattern, content)
        important_terms.extend(technical_terms)
        
        # Look for common research terms
        research_terms = [
            'machine learning', 'artificial intelligence', 'deep learning',
            'neural network', 'algorithm', 'model', 'analysis', 'method',
            'experiment', 'evaluation', 'performance', 'optimization'
        ]
        
        for term in research_terms:
            if term in content:
                important_terms.append(term)
        
        return list(set(important_terms))

class ReviewerProfiler:
    """Create detailed reviewer profiles"""
    
    def create_profile(self, reviewer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive reviewer profile"""
        
        profile = {
            'reviewer_id': reviewer_data.get('id', 'unknown'),
            'expertise_level': self._calculate_expertise_level(reviewer_data),
            'experience_score': self._calculate_experience_score(reviewer_data),
            'reliability_score': self._calculate_reliability_score(reviewer_data),
            'average_review_time': reviewer_data.get('average_review_time', 14),
            'current_workload': self._calculate_current_workload(reviewer_data),
            'specializations': reviewer_data.get('specializations', []),
            'recent_performance': reviewer_data.get('recent_performance', 0.8),
            'availability_status': reviewer_data.get('availability', 'available')
        }
        
        return profile
    
    def _calculate_expertise_level(self, reviewer_data: Dict[str, Any]) -> float:
        """Calculate reviewer expertise level"""
        
        # Years of experience
        years_experience = reviewer_data.get('years_experience', 5)
        experience_score = min(years_experience / 20, 1.0)  # Max at 20 years
        
        # Number of publications
        publications = reviewer_data.get('publication_count', 10)
        publication_score = min(publications / 50, 1.0)  # Max at 50 publications
        
        # H-index or similar metric
        h_index = reviewer_data.get('h_index', 5)
        h_index_score = min(h_index / 20, 1.0)  # Max at h-index of 20
        
        # Weighted combination
        expertise_level = (experience_score * 0.3 + publication_score * 0.4 + h_index_score * 0.3)
        
        return round(expertise_level, 3)
    
    def _calculate_experience_score(self, reviewer_data: Dict[str, Any]) -> float:
        """Calculate reviewer experience score"""
        
        # Number of reviews completed
        reviews_completed = reviewer_data.get('reviews_completed', 5)
        review_experience = min(reviews_completed / 20, 1.0)  # Max at 20 reviews
        
        # Quality of past reviews (if available)
        review_quality = reviewer_data.get('average_review_quality', 0.8)
        
        # Combine factors
        experience_score = (review_experience * 0.6 + review_quality * 0.4)
        
        return round(experience_score, 3)
    
    def _calculate_reliability_score(self, reviewer_data: Dict[str, Any]) -> float:
        """Calculate reviewer reliability score"""
        
        # On-time completion rate
        on_time_rate = reviewer_data.get('on_time_completion_rate', 0.8)
        
        # Response rate to review invitations
        response_rate = reviewer_data.get('invitation_response_rate', 0.7)
        
        # Recent review completion rate
        recent_completion_rate = reviewer_data.get('recent_completion_rate', 0.8)
        
        # Weighted average
        reliability_score = (on_time_rate * 0.4 + response_rate * 0.3 + recent_completion_rate * 0.3)
        
        return round(reliability_score, 3)
    
    def _calculate_current_workload(self, reviewer_data: Dict[str, Any]) -> float:
        """Calculate current workload as percentage of capacity"""
        
        active_reviews = reviewer_data.get('active_reviews', 0)
        max_capacity = reviewer_data.get('max_capacity', 5)
        
        workload_percentage = active_reviews / max_capacity if max_capacity > 0 else 0
        
        return round(min(workload_percentage, 1.0), 3)

class ManuscriptAnalyzer:
    """Analyze manuscript characteristics"""
    
    def analyze_complexity(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze manuscript complexity"""
        
        content = submission_data.get('content', '')
        
        analysis = {
            'complexity_score': self._calculate_complexity_score(content),
            'technical_depth': self._assess_technical_depth(content),
            'methodology_complexity': self._assess_methodology_complexity(content),
            'interdisciplinary_level': self._assess_interdisciplinary_level(content),
            'statistical_complexity': self._assess_statistical_complexity(content),
            'length_complexity': self._assess_length_complexity(content)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate overall manuscript complexity score"""
        
        complexity_indicators = {
            'technical_terms': len(re.findall(r'\b[A-Z][a-z]*[A-Z][a-z]*\b', content)) / max(len(content.split()), 1),
            'mathematical_content': content.lower().count('equation') + content.lower().count('formula'),
            'statistical_terms': sum(content.lower().count(term) for term in ['statistical', 'regression', 'analysis', 'correlation']),
            'methodology_sections': sum(content.lower().count(term) for term in ['method', 'procedure', 'protocol', 'algorithm']),
            'length_factor': min(len(content.split()) / 5000, 1.0)  # Normalize to 5000 words
        }
        
        # Normalize and combine indicators
        normalized_score = sum(min(score / 10, 1.0) for score in complexity_indicators.values()) / len(complexity_indicators)
        
        return round(normalized_score, 3)
    
    def _assess_technical_depth(self, content: str) -> float:
        """Assess technical depth of manuscript"""
        technical_indicators = [
            'algorithm', 'implementation', 'optimization', 'framework',
            'architecture', 'protocol', 'methodology', 'technical'
        ]
        
        technical_score = sum(content.lower().count(term) for term in technical_indicators)
        normalized_score = min(technical_score / 20, 1.0)  # Normalize
        
        return round(normalized_score, 3)
    
    def _assess_methodology_complexity(self, content: str) -> float:
        """Assess methodology complexity"""
        methodology_indicators = [
            'experimental design', 'control group', 'randomized', 'statistical analysis',
            'data collection', 'sampling', 'measurement', 'validation'
        ]
        
        methodology_score = sum(content.lower().count(term) for term in methodology_indicators)
        normalized_score = min(methodology_score / 15, 1.0)  # Normalize
        
        return round(normalized_score, 3)
    
    def _assess_interdisciplinary_level(self, content: str) -> float:
        """Assess interdisciplinary nature"""
        discipline_terms = {
            'computer_science': ['algorithm', 'software', 'programming', 'computation'],
            'mathematics': ['mathematical', 'equation', 'theorem', 'proof'],
            'biology': ['biological', 'organism', 'cell', 'genetic'],
            'physics': ['physical', 'energy', 'quantum', 'particle'],
            'chemistry': ['chemical', 'molecular', 'reaction', 'compound'],
            'psychology': ['psychological', 'behavior', 'cognitive', 'mental'],
            'engineering': ['engineering', 'design', 'system', 'optimization']
        }
        
        disciplines_found = 0
        for discipline, terms in discipline_terms.items():
            if any(term in content.lower() for term in terms):
                disciplines_found += 1
        
        interdisciplinary_score = min(disciplines_found / 3, 1.0)  # Max at 3 disciplines
        
        return round(interdisciplinary_score, 3)
    
    def _assess_statistical_complexity(self, content: str) -> float:
        """Assess statistical complexity"""
        statistical_terms = [
            'regression', 'correlation', 'significance', 'p-value',
            'confidence interval', 'hypothesis test', 'anova', 'chi-square'
        ]
        
        statistical_score = sum(content.lower().count(term) for term in statistical_terms)
        normalized_score = min(statistical_score / 10, 1.0)  # Normalize
        
        return round(normalized_score, 3)
    
    def _assess_length_complexity(self, content: str) -> float:
        """Assess complexity based on manuscript length"""
        word_count = len(content.split())
        
        if word_count < 2000:
            return 0.3  # Short, potentially less complex
        elif word_count < 5000:
            return 0.6  # Medium length
        elif word_count < 10000:
            return 0.8  # Long, potentially more complex
        else:
            return 1.0  # Very long, likely complex

class InteractionPredictor:
    """Predict reviewer-manuscript interaction quality"""
    
    def predict_interaction(self, reviewer_profile: Dict[str, Any], 
                          manuscript_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict quality of reviewer-manuscript interaction"""
        
        # Calculate match score based on various factors
        match_factors = {
            'expertise_complexity_match': self._calculate_expertise_complexity_match(
                reviewer_profile, manuscript_analysis
            ),
            'experience_depth_match': self._calculate_experience_depth_match(
                reviewer_profile, manuscript_analysis
            ),
            'workload_complexity_balance': self._calculate_workload_complexity_balance(
                reviewer_profile, manuscript_analysis
            )
        }
        
        # Overall match score
        overall_match = sum(match_factors.values()) / len(match_factors)
        
        return {
            'match_score': round(overall_match, 3),
            'match_factors': match_factors,
            'interaction_quality': 'high' if overall_match > 0.7 else 'medium' if overall_match > 0.4 else 'low',
            'potential_challenges': self._identify_interaction_challenges(reviewer_profile, manuscript_analysis)
        }
    
    def _calculate_expertise_complexity_match(self, reviewer_profile: Dict[str, Any], 
                                           manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate how well reviewer expertise matches manuscript complexity"""
        
        reviewer_expertise = reviewer_profile.get('expertise_level', 0.5)
        manuscript_complexity = manuscript_analysis.get('complexity_score', 0.5)
        
        # Ideal match: high expertise for high complexity, adequate expertise for low complexity
        if manuscript_complexity > 0.7:  # High complexity
            match_score = reviewer_expertise  # Need high expertise
        elif manuscript_complexity < 0.3:  # Low complexity
            match_score = 1.0 - abs(reviewer_expertise - 0.6)  # Moderate expertise is fine
        else:  # Medium complexity
            match_score = 1.0 - abs(reviewer_expertise - manuscript_complexity)
        
        return round(max(match_score, 0.0), 3)
    
    def _calculate_experience_depth_match(self, reviewer_profile: Dict[str, Any], 
                                        manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate experience-depth match"""
        
        reviewer_experience = reviewer_profile.get('experience_score', 0.5)
        manuscript_depth = manuscript_analysis.get('technical_depth', 0.5)
        
        # Similar logic to expertise-complexity match
        match_score = 1.0 - abs(reviewer_experience - manuscript_depth)
        
        return round(max(match_score, 0.0), 3)
    
    def _calculate_workload_complexity_balance(self, reviewer_profile: Dict[str, Any], 
                                             manuscript_analysis: Dict[str, Any]) -> float:
        """Calculate workload-complexity balance"""
        
        current_workload = reviewer_profile.get('current_workload', 0.5)
        manuscript_complexity = manuscript_analysis.get('complexity_score', 0.5)
        
        # High workload + high complexity = poor balance
        # Low workload + any complexity = good balance
        if current_workload < 0.5:
            balance_score = 1.0  # Low workload is always good
        else:
            # Higher workload requires lower complexity for good balance
            balance_score = max(0.0, 1.0 - (current_workload * manuscript_complexity))
        
        return round(balance_score, 3)
    
    def _identify_interaction_challenges(self, reviewer_profile: Dict[str, Any], 
                                       manuscript_analysis: Dict[str, Any]) -> List[str]:
        """Identify potential interaction challenges"""
        challenges = []
        
        # Expertise mismatch
        if reviewer_profile.get('expertise_level', 0.5) < manuscript_analysis.get('complexity_score', 0.5) - 0.3:
            challenges.append("Reviewer expertise may be insufficient for manuscript complexity")
        
        # High workload + complex manuscript
        if (reviewer_profile.get('current_workload', 0.5) > 0.7 and 
            manuscript_analysis.get('complexity_score', 0.5) > 0.7):
            challenges.append("High workload combined with complex manuscript may lead to delays")
        
        # Interdisciplinary challenges
        if manuscript_analysis.get('interdisciplinary_level', 0.0) > 0.7:
            challenges.append("Interdisciplinary manuscript may require broader expertise")
        
        # Statistical complexity challenges
        if manuscript_analysis.get('statistical_complexity', 0.0) > 0.8:
            challenges.append("High statistical complexity may require specialized statistical expertise")
        
        return challenges

# Additional supporting classes
class CapacityAnalyzer:
    """Analyze reviewer capacity"""
    
    def analyze_capacity(self, reviewer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reviewer's current capacity"""
        active_reviews = reviewer_data.get('active_reviews', 0)
        max_capacity = reviewer_data.get('max_capacity', 5)
        
        return {
            'current_reviews': active_reviews,
            'max_capacity': max_capacity,
            'available_slots': max(0, max_capacity - active_reviews),
            'utilization_rate': active_reviews / max_capacity if max_capacity > 0 else 0,
            'capacity_status': self._get_capacity_status(active_reviews, max_capacity)
        }
    
    def _get_capacity_status(self, active: int, max_cap: int) -> str:
        """Get capacity status description"""
        if max_cap == 0:
            return 'unavailable'
        
        utilization = active / max_cap
        if utilization >= 1.0:
            return 'at_capacity'
        elif utilization >= 0.8:
            return 'high_utilization'
        elif utilization >= 0.5:
            return 'moderate_utilization'
        else:
            return 'available'

class AssignmentOptimizer:
    """Optimize review assignments"""
    
    def optimize_assignments(self, reviewer_capacities: List[Dict[str, Any]], 
                           pending_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize review assignments"""
        assignments = []
        
        # Sort reviewers by availability
        available_reviewers = [
            r for r in reviewer_capacities 
            if r['current_capacity']['available_slots'] > 0
        ]
        available_reviewers.sort(key=lambda x: x['current_capacity']['utilization_rate'])
        
        # Assign reviews to least utilized reviewers first
        for review in pending_reviews:
            if available_reviewers:
                assigned_reviewer = available_reviewers[0]
                assignments.append({
                    'review_id': review.get('id', 'unknown'),
                    'reviewer_id': assigned_reviewer['reviewer_id'],
                    'current_workload': assigned_reviewer['current_capacity']['utilization_rate'],
                    'optimal_workload': assigned_reviewer['current_capacity']['utilization_rate'] + 0.2,
                    'capacity_utilization': assigned_reviewer['current_capacity']['utilization_rate'],
                    'assignment_priority': review.get('priority', 'normal'),
                    'urgency_level': review.get('urgency', 'normal')
                })
                
                # Update reviewer capacity
                assigned_reviewer['current_capacity']['available_slots'] -= 1
                assigned_reviewer['current_capacity']['utilization_rate'] += 0.2
                
                # Remove reviewer if at capacity
                if assigned_reviewer['current_capacity']['available_slots'] <= 0:
                    available_reviewers.remove(assigned_reviewer)
        
        return assignments

class TimelinePredictor:
    """Predict review completion timelines"""
    
    def predict_completion_times(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict completion times for assignments"""
        predictions = {}
        
        for assignment in assignments:
            review_id = assignment.get('review_id')
            workload = assignment.get('current_workload', 0.5)
            priority = assignment.get('assignment_priority', 'normal')
            
            # Base time estimates
            base_times = {
                'urgent': 7,    # days
                'high': 14,
                'normal': 21,
                'low': 28
            }
            
            base_time = base_times.get(priority, 21)
            
            # Adjust for workload
            workload_multiplier = 1 + workload * 0.5  # Higher workload = longer time
            
            predicted_time = int(base_time * workload_multiplier)
            
            predictions[review_id] = {
                'predicted_days': predicted_time,
                'confidence': 0.8 - workload * 0.2,  # Lower confidence for high workload
                'factors': {
                    'base_time': base_time,
                    'workload_adjustment': workload_multiplier,
                    'final_estimate': predicted_time
                }
            }
        
        return predictions