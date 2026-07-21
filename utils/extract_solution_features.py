"""
Extract and analyze characteristics from TSP solution files.

This module provides utilities to extract, aggregate, and analyze:
- Solution quality metrics (distance, convergence speed)
- Algorithm parameters used
- Tour structure characteristics
- Operator effectiveness per solution
- Comparative analysis across runs
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class SolutionExtractor:
    """Extract characteristics from solution JSON and log files."""
    
    @staticmethod
    def extract_solution_json(json_file: Path) -> Dict:
        """
        Extract all characteristics from a single solution JSON file.
        
        Returns:
            dict with keys:
                - route: list of city indices
                - total_distance: final tour cost
                - best_total_distance: best found during VNS
                - solver_params: VNS configuration used
                - method: initialization method (marl, nearest_neighbor, etc)
        """
        with open(json_file) as f:
            return json.load(f)
    
    @staticmethod
    def extract_operator_stats_from_log(log_file: Path) -> Dict[str, Dict]:
        """
        Parse VNS log file to extract operator effectiveness.
        
        Returns:
            dict mapping operator names to:
                - improvements: count of successful moves
                - failures: count of failed moves
                - total_reward: sum of all rewards
                - avg_reward: average reward per improvement
        """
        operator_stats = {}
        improvement_pattern = r"Improvement with (\w+) -> distance ([\d.]+) \(reward ([\d.-]+)\)"
        failure_pattern = r"No improvement with (\w+) \(penalty ([\d.-]+)\)"
        
        with open(log_file) as f:
            content = f.read()
        
        # Extract improvements
        for match in re.finditer(improvement_pattern, content):
            op_name = match.group(1)
            reward = float(match.group(3))
            if op_name not in operator_stats:
                operator_stats[op_name] = {
                    'improvements': 0,
                    'failures': 0,
                    'total_reward': 0.0,
                    'total_penalty': 0.0,
                }
            operator_stats[op_name]['improvements'] += 1
            operator_stats[op_name]['total_reward'] += reward
        
        # Extract failures
        for match in re.finditer(failure_pattern, content):
            op_name = match.group(1)
            penalty = float(match.group(2))
            if op_name not in operator_stats:
                operator_stats[op_name] = {
                    'improvements': 0,
                    'failures': 0,
                    'total_reward': 0.0,
                    'total_penalty': 0.0,
                }
            operator_stats[op_name]['failures'] += 1
            operator_stats[op_name]['total_penalty'] += penalty
        
        # Compute averages
        for op_name in operator_stats:
            stats = operator_stats[op_name]
            stats['avg_reward'] = (stats['total_reward'] / stats['improvements'] 
                                  if stats['improvements'] > 0 else 0)
            stats['success_rate'] = (stats['improvements'] / (stats['improvements'] + stats['failures']) * 100
                                    if (stats['improvements'] + stats['failures']) > 0 else 0)
        
        return operator_stats
    
    @staticmethod
    def batch_extract_solutions(solution_dir: Path) -> pd.DataFrame:
        """
        Extract characteristics from all solution JSON files in a directory.
        
        Returns:
            DataFrame with columns:
                - file: solution filename
                - total_distance: final tour cost
                - best_total_distance: best found
                - method: initialization method
                - tour_length: number of cities
                - solver_params: dict of VNS parameters
        """
        characteristics = []
        solution_files = sorted(solution_dir.glob('*_solution_*.json'))
        
        for sol_file in solution_files:
            try:
                sol = SolutionExtractor.extract_solution_json(sol_file)
                characteristics.append({
                    'file': sol_file.name,
                    'total_distance': sol.get('total_distance', np.nan),
                    'best_total_distance': sol.get('best_total_distance', np.nan),
                    'method': sol.get('solver_params', {}).get('method', 'N/A'),
                    'tour_length': len(sol.get('route', [])),
                    'best_known_distance': sol.get('solver_params', {}).get('best_known_distance', np.nan),
                })
            except Exception as e:
                print(f"Error processing {sol_file.name}: {e}")
        
        return pd.DataFrame(characteristics)
    
    @staticmethod
    def compute_solution_quality_metrics(df: pd.DataFrame) -> Dict:
        """
        Compute aggregate quality metrics from solution DataFrame.
        
        Returns:
            dict with:
                - best_distance: best solution found
                - worst_distance: worst solution found
                - mean_distance: average quality
                - std_distance: standard deviation
                - gap_to_best: percentage gap for each solution
                - convergence_rate: % solutions reaching best known
        """
        best = df['total_distance'].min()
        worst = df['total_distance'].max()
        mean = df['total_distance'].mean()
        std = df['total_distance'].std()
        best_known = df['best_known_distance'].iloc[0] if len(df) > 0 else np.nan
        
        return {
            'best_distance': best,
            'worst_distance': worst,
            'mean_distance': mean,
            'std_distance': std,
            'best_known_distance': best_known,
            'convergence_rate_pct': (len(df[df['total_distance'] <= best_known + 0.01]) / len(df) * 100) if not np.isnan(best_known) else 0,
            'gap_from_best_pct': ((mean - best) / best * 100),
        }


class SolutionComparator:
    """Compare solutions across multiple runs and configurations."""
    
    @staticmethod
    def jaccard_similarity(tour1: List[int], tour2: List[int]) -> float:
        """Compute Jaccard similarity between two tours (ignoring rotation/reflection)."""
        # Normalize tours by starting from city 0
        def normalize_tour(tour):
            idx = tour.index(0) if 0 in tour else 0
            normalized = tour[idx:] + tour[:idx]
            # Remove last element (depot repeat) for comparison
            return set(normalized[:-1])
        
        set1 = normalize_tour(tour1)
        set2 = normalize_tour(tour2)
        
        if len(set1 | set2) == 0:
            return 1.0
        return len(set1 & set2) / len(set1 | set2)
    
    @staticmethod
    def edge_overlap(tour1: List[int], tour2: List[int]) -> float:
        """Compute percentage of edges that are identical between tours."""
        edges1 = set()
        edges2 = set()
        
        # Build edge sets (undirected edges)
        for i in range(len(tour1) - 1):
            edge = tuple(sorted([tour1[i], tour1[i+1]]))
            edges1.add(edge)
        
        for i in range(len(tour2) - 1):
            edge = tuple(sorted([tour2[i], tour2[i+1]]))
            edges2.add(edge)
        
        if len(edges1 | edges2) == 0:
            return 1.0
        return len(edges1 & edges2) / len(edges1 | edges2) * 100
    
    @staticmethod
    def solution_diversity(solution_dir: Path) -> pd.DataFrame:
        """
        Analyze diversity of solutions across all runs.
        
        Returns:
            DataFrame showing pairwise similarity metrics between solutions
        """
        solution_files = sorted(solution_dir.glob('*_solution_*.json'))
        
        diversity_data = []
        solutions = []
        
        for sol_file in solution_files:
            with open(sol_file) as f:
                sol = json.load(f)
            solutions.append((sol_file.name, sol.get('route', [])))
        
        # Compute pairwise similarities
        for i, (name1, tour1) in enumerate(solutions):
            for j, (name2, tour2) in enumerate(solutions):
                if i < j:
                    jaccard = SolutionComparator.jaccard_similarity(tour1, tour2)
                    edge_overlap = SolutionComparator.edge_overlap(tour1, tour2)
                    diversity_data.append({
                        'solution_1': name1,
                        'solution_2': name2,
                        'jaccard_similarity': jaccard,
                        'edge_overlap_pct': edge_overlap,
                    })
        
        return pd.DataFrame(diversity_data)


# Example usage
if __name__ == "__main__":
    # Extract from swiss42
    sol_dir = Path("/Users/karinaassiniandreatta/Documents/01 phd/codes/tsp_rl_metaheuristic/outputs/solutions/swiss42/VNS_Solver_Q_Learnings/marl/20260224_210316")
    
    print("=" * 80)
    print("SOLUTION QUALITY ANALYSIS")
    print("=" * 80)
    
    # Batch extract all solutions
    df = SolutionExtractor.batch_extract_solutions(sol_dir)
    print(f"\nExtracted {len(df)} solutions")
    print("\nStatistics:")
    print(df[['total_distance', 'best_total_distance']].describe())
    
    # Compute quality metrics
    metrics = SolutionExtractor.compute_solution_quality_metrics(df)
    print("\nQuality Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Analyze diversity
    print("\n" + "=" * 80)
    print("SOLUTION DIVERSITY ANALYSIS")
    print("=" * 80)
    diversity_df = SolutionComparator.solution_diversity(sol_dir)
    print(f"\nMean Jaccard similarity: {diversity_df['jaccard_similarity'].mean():.4f}")
    print(f"Mean edge overlap: {diversity_df['edge_overlap_pct'].mean():.2f}%")
    print(f"Diversity (1 - similarity): {1 - diversity_df['jaccard_similarity'].mean():.4f}")
