# demo_visualization.py
import spacy
from spacy import displacy
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any
import networkx as nx
import numpy as np
import os, sys

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import existing utils
from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.quality.reqGrading import EnhancedRequirementAnalyzer
from src.utils.path_resolver import SmartPathResolver

logger = logging.getLogger(__name__)
class Demo:
    def __init__(self, 
                 requirements_file: str = "requirements.csv",
                 activities_file: str = "activities.csv",
                 ground_truth_file: Optional[str] = "manual_matches.csv"):

        # Load spaCy
        self.nlp = spacy.load("en_core_web_lg")
        self.repo_manager = RepositoryStructureManager()
        self.path_resolver = SmartPathResolver(repo_manager=self.repo_manager)
        self.matcher = AerospaceMatcher(repo_manager=self.repo_manager)
        self.analyzer = EnhancedRequirementAnalyzer()

        # --- Use repo manager visuals folder ---
        self.OUTPUT_DIR = self.repo_manager.structure['visuals_output']
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visuals output directory: {self.OUTPUT_DIR}")

        # --- Resolve file paths ---
        file_mapping = {
            'requirements': requirements_file,
            'activities': activities_file,
            'ground_truth': ground_truth_file
        }
        resolved_paths = self.path_resolver.resolve_input_files(file_mapping)
        self.requirements_file = resolved_paths['requirements']
        self.activities_file = resolved_paths['activities']
        self.ground_truth_file = resolved_paths['ground_truth']

        logger.info(f"Using resolved requirements: {self.requirements_file}")
        logger.info(f"Using resolved activities: {self.activities_file}")
        logger.info(f"Using resolved ground truth: {self.ground_truth_file}")

        # --- Load DataFrames from resolved paths ---
        self.req_df = self.matcher.load_requirements(self.requirements_file)
        self.act_df = self.matcher.load_activities(self.activities_file)

    # --- Run matching ---
    def _run_matching(self, matching_config: Dict, min_similarity: float, top_matches: int) -> pd.DataFrame:
        try:
            matches_df, _ = self.matcher.run_matching(
                requirements_file=self.requirements_file,
                activities_file=self.activities_file,
                weights=matching_config,
                min_similarity=min_similarity,
                top_n=top_matches,
                output_file="temp_matches",
                save_explanations=False
            )              
            logger.info(f"✓ Matching completed: {len(matches_df)} matches found")
            return matches_df
        except Exception as e:
            logger.error(f"❌ Matching failed: {e}")
            raise

    # --- Visualization Functions (instance methods using self.OUTPUT_DIR) ---
    def save_displacy(self, doc, name, style="dep"):
        svg = displacy.render(doc, style=style)
        path = self.OUTPUT_DIR / f"{name}_{style}.svg"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"[saved] {path}")

    def plot_match_scores(self, expl: pd.Series, req_id: str, act_name: str):
        scores = {
            "Semantic": expl.get("Semantic_Score", 0),
            "BM25": expl.get("BM25_Score", 0),
            "Domain": expl.get("Domain_Score", 0),
            "Query Expansion": expl.get("Query_Expansion_Score", 0),
        }
        plt.barh(list(scores.keys()), list(scores.values()))
        plt.title(f"Scores: {req_id} → {act_name}")
        plt.tight_layout()
        path = self.OUTPUT_DIR / f"{req_id}_{act_name}_scores.png"
        plt.savefig(path)
        plt.close()
        print(f"[saved] {path}")

    def plot_match_network(self, results_df: pd.DataFrame, threshold=0.4):
        """Plot requirement-activity match network with proper column names."""
        G = nx.Graph()
        for _, row in results_df.iterrows():
            req = row.get("Requirement Text", "REQ_UNKNOWN")
            act = row.get("Activity Name", "ACT_UNKNOWN")
            score = row.get("Combined Score", 0)
            if score >= threshold:
                G.add_edge(req, act, weight=score)

        pos = nx.spring_layout(G, k=0.5, seed=42)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_color="lightblue",
                node_size=800, font_size=8, width=2)
        plt.title("Requirement–Activity Match Network")
        path = self.OUTPUT_DIR / "match_network.png"
        plt.savefig(path)
        plt.close()
        print(f"[saved] {path}")

    def plot_quality_radar(self, metrics, req_id: str):
        """Plot radar chart from QualityMetrics dataclass."""
        labels = ["Clarity", "Completeness", "Verifiability",
                "Atomicity", "Consistency", "Semantic"]
        values = [
            getattr(metrics, "clarity_score", 0),
            getattr(metrics, "completeness_score", 0),
            getattr(metrics, "verifiability_score", 0),
            getattr(metrics, "atomicity_score", 0),
            getattr(metrics, "consistency_score", 0),
            getattr(metrics, "semantic_quality_score", 0),
        ]
        values += values[:1]  # close loop

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(subplot_kw={"polar": True})
        ax.plot(angles, values, "o-", linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(f"Requirement Quality: {req_id}")
        path = self.OUTPUT_DIR / f"{req_id}_quality_radar.png"
        plt.savefig(path)
        plt.close()
        print(f"[saved] {path}")

def main():
    # Initialize demo (paths automatically resolved, visuals use repo manager folder)
    demo = Demo(
        requirements_file="requirements.csv",
        activities_file="activities.csv",
        ground_truth_file="manual_matches.csv"
    )

    # Pick one example requirement + activity
    req_row = demo.req_df.iloc[0]
    act_row = demo.act_df.iloc[0]

    req_text = req_row["Requirement Text"]
    req_id = req_row["ID"] if "ID" in req_row else req_row.get("Requirement ID", "REQ_0")
    act_text = act_row["Activity Name"] if "Activity Name" in act_row else act_row.get("Activity", "ACT_0")

    # --- spaCy docs ---
    req_doc = demo.matcher.nlp(req_text)
    act_doc = demo.matcher.nlp(act_text)

    # --- Save displacy visuals ---
    demo.save_displacy(req_doc, req_id, style="dep")
    demo.save_displacy(req_doc, req_id, style="ent")

    # --- Run matching ---
    matching_config = {'semantic': 1, 'bm25': 1, 'domain': 1, 'query_expansion': 1}
    matches_df = demo._run_matching(matching_config, min_similarity=0.15, top_matches=5)

    # Pick first match explanation (DataFrame row)
    example_expl = matches_df.iloc[0]

    # --- Plot match score breakdown ---
    demo.plot_match_scores(example_expl, req_id, act_text)

    # --- Plot network of matches ---
    demo.plot_match_network(matches_df, threshold=0.4)

    # --- Run quality analysis ---
    issues, metrics, incose_analysis, semantic_analysis = demo.analyzer.analyze_requirement(
        req_text, req_id=req_id
    )

    print("Issues:", issues)
    print("Metrics:", metrics)
    print("INCOSE Best Pattern:", incose_analysis.best_pattern)
    print("Semantic Issues:", semantic_analysis.contextual_ambiguities)

    # --- Plot radar chart ---
    demo.plot_quality_radar(metrics, req_id)


if __name__ == "__main__":
    main()