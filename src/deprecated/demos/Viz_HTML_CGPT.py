# vizV1_html.py
"""
HTML-based Technical Journey Visualizer (16:9 slides)
Drop into the same package where vizV1.py lived. Reuses RepositoryStructureManager and SafeFileHandler.
"""

import json
import textwrap
from pathlib import Path
from datetime import datetime
import html as html_module

# re-use project utilities (same imports pattern as vizV1.py)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.repository_setup import RepositoryStructureManager
from src.utils.file_utils import SafeFileHandler
from src.matching.matcher import AerospaceMatcher
from src.quality.reqGrading import EnhancedRequirementAnalyzer

class TechnicalJourneyVisualizerHTML:
    """
    HTML/CSS-based visualizer for the technical processing journey.
    Produces 16:9 slides (default 1920x1080) as standalone HTML files.
    """
    def __init__(self, width: int = 1920, height: int = 1080):
        self.repo_manager = RepositoryStructureManager("outputs")
        self.file_handler = SafeFileHandler(self.repo_manager)
        self.matcher = AerospaceMatcher(repo_manager=self.repo_manager)
        self.quality_analyzer = EnhancedRequirementAnalyzer(repo_manager=self.repo_manager)
        self.domain = self.matcher  # domain resources available via matcher in your repo
        self.output_dir = self.repo_manager.structure['visuals_output']
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Target slide size (pixels)
        self.width = width
        self.height = height

        # Small color palette to mirror vizV1 style
        self.colors = {
            'semantic': '#3498db',
            'bm25': '#e74c3c',
            'domain': '#2ecc71',
            'query_expansion': '#f39c12',
            'good': '#27ae60',
            'warning': '#f39c12',
            'bad': '#c0392b',
            'neutral': '#95a5a6',
            'background': '#ffffff'
        }

    # -----------------------
    # Utilities
    # -----------------------
    def _read_matches_and_explanations(self):
        # load matches.csv and explanations JSON if available (same behavior as original vizV1)
        matches_path = self.repo_manager.structure['matching_results'] / "aerospace_matches.csv"
        explanations_path = self.repo_manager.structure['matching_results'] / "aerospace_matches_explanations.json"

        if not matches_path.exists():
            raise FileNotFoundError("aerospace_matches.csv not found — run matcher first.")
        import pandas as pd
        matches_df = pd.read_csv(matches_path)

        explanations = {}
        if explanations_path.exists():
            with open(explanations_path, 'r', encoding='utf-8') as f:
                ex_list = json.load(f)
                for exp in ex_list:
                    key = (exp.get('requirement_id'), exp.get('activity_name'))
                    explanations[key] = exp
        return matches_df, explanations

    def _escape(self, text: str) -> str:
        return html_module.escape(str(text) if text is not None else "")

    def _wrap_lines(self, text: str, width: int = 80):
        return textwrap.wrap(text or "", width=width)

    def _slide_style(self) -> str:
        # Inline CSS tuned for 16:9 slide; you can tweak fonts/sizes here.
        return f"""
        :root {{
            --slide-w: {self.width}px;
            --slide-h: {self.height}px;
            --padding: 32px;
            --corner: 12px;
            --muted: #6b7680;
            --bg: {self.colors['background']};
            --card-bg: #fff;
            --sans: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        html,body{{height:100%;margin:0;background:var(--bg);font-family:var(--sans);-webkit-font-smoothing:antialiased;}}
        .slide{{width:var(--slide-w);height:var(--slide-h);box-sizing:border-box;padding:var(--padding);margin:0 auto;background:linear-gradient(180deg,#fff,#f7f9fb);}}
        .title{{font-size:34px;font-weight:700;margin-bottom:14px;}}
        .subtitle{{color:var(--muted);margin-bottom:18px;font-size:14px}}
        .row{{display:flex;gap:20px;align-items:flex-start;margin-bottom:18px;}}
        .col{{flex:1;min-width:0;}}
        .card{{background:var(--card-bg);border-radius:var(--corner);box-shadow:0 6px 18px rgba(0,0,0,0.06);padding:14px;overflow:hidden;}}
        .card h3{{
            margin:0 0 8px 0;font-size:18px;
        }}
        .score-bubble{{display:inline-block;padding:8px 12px;border-radius:999px;font-weight:700;color:#fff}}
        .bar-bg{{height:12px;background:#e9eef4;border-radius:8px;overflow:hidden}}
        .bar-fill{{height:100%;border-radius:8px}}
        .meta{{font-size:12px;color:var(--muted)}}
        .small{{font-size:12px;color:var(--muted)}}
        pre.report{{white-space:pre-wrap;font-family:monospace;background:#f4f6f8;padding:12px;border-radius:8px;overflow:auto;max-height:320px}}
        .grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
        .center{{text-align:center}}
        .pill{{display:inline-block;padding:6px 10px;border-radius:999px;background:#f0f4f8;font-weight:600;font-size:12px}}
        """

    # -----------------------
    # HTML generation
    # -----------------------
    def create_processing_journey_html(self, requirement_id: str = None, label: str = None) -> str:
        """
        Create an HTML slide representing the technical journey of a requirement.
        Returns the file path saved (string).
        """
        matches_df, explanations = self._read_matches_and_explanations()

        # choose a match
        if requirement_id:
            sel = matches_df[matches_df['Requirement_ID'] == requirement_id]
            if sel.empty:
                sel = matches_df.nlargest(1, 'Combined_Score')
        else:
            sel = matches_df.nlargest(1, 'Combined_Score')

        if sel.empty:
            raise ValueError("No matches found in aerospace_matches.csv")

        row = sel.iloc[0]
        req_id = row['Requirement_ID']
        req_text = row['Requirement_Text']
        activity = row.get('Activity_Name', '')
        explanation = explanations.get((req_id, activity), {})

        # Build HTML pieces for each layer (mirrors the structure in vizV1)
        title = f"Technical Processing Journey: {req_id} (Score: {row.get('Combined_Score', 0):.3f})"
        if label:
            title = f"{label} | {title}"

        # Layer 1 - Raw Inputs
        layer1_html = f"""
        <div class="card">
          <h3>Layer 1 — Raw Inputs</h3>
          <div class="row">
            <div class="col">
              <strong>Requirement ({self._escape(req_id)})</strong>
              <p class="small">{self._escape(req_text)}</p>
              <div class="meta">Source: requirements.csv</div>
            </div>
            <div class="col">
              <strong>Activity</strong>
              <p class="small">{self._escape(activity)}</p>
              <div class="meta">Source: activities.csv</div>
            </div>
          </div>
        </div>
        """

        # Layer 2 - Preprocessing (abbrev expansion + tokens + terms)
        try:
            expanded_req = self.matcher._expand_aerospace_abbreviations(req_text)
            req_doc = self.matcher.nlp(expanded_req)
            req_terms = self.matcher._preprocess_text_aerospace(req_text)
            act_terms = self.matcher._preprocess_text_aerospace(activity)
        except Exception:
            expanded_req = req_text
            req_doc = None
            req_terms = []
            act_terms = []

        layer2_html = f"""
        <div class="card">
          <h3>Layer 2 — Preprocessing Pipeline</h3>
          <div class="row">
            <div class="col"><strong>Abbreviation Expansion</strong>
              <p class="small">{self._escape(expanded_req)}</p>
            </div>
            <div class="col"><strong>Tokenization</strong>
              <p class="small">Tokens: {len(req_doc) if req_doc is not None else 'N/A'}</p>
            </div>
            <div class="col"><strong>Term Extraction</strong>
              <p class="small">Req terms: {len(req_terms)} · Act terms: {len(act_terms)}</p>
            </div>
          </div>
        </div>
        """

        # Layer 3 — Algorithm scores (from explanations JSON if available)
        scores_obj = explanation.get('scores', {}) if explanation else {}
        def score_html(algo_key, display):
            score = scores_obj.get(algo_key, 0.0)
            color = self.colors.get(algo_key, self.colors['neutral'])
            pct = float(score)
            pct_display = f"{pct:.3f}"
            return f"""
            <div class="card center">
                <div class="pill">{display}</div>
                <div style="height:6px"></div>
                <div class="score-bubble" style="background:{color}">{pct_display}</div>
                <div style="height:8px"></div>
                <div class="bar-bg" role="progressbar" aria-valuenow="{pct*100}">
                    <div class="bar-fill" style="width:{pct*100}%;background:{color}"></div>
                </div>
            </div>
            """
        layer3_html = f"""
        <div class="card">
          <h3>Layer 3 — Algorithm Analysis</h3>
          <div class="grid-4">
            {score_html('semantic', 'Semantic')}
            {score_html('bm25', 'BM25')}
            {score_html('domain', 'Domain')}
            {score_html('query_expansion', 'Query Exp.')}
          </div>
        </div>
        """

        # Layer 4 — Score combination (display formula & combined score)
        combined_score = row.get('Combined_Score', 0.0)
        layer4_html = f"""
        <div class="card">
          <h3>Layer 4 — Score Combination</h3>
          <p class="small">Combined = weighted average of algorithm components (weights implied by config)</p>
          <div style="display:flex;align-items:center;gap:18px">
            <div style="flex:1">
              <div class="bar-bg"><div class="bar-fill" style="width:{combined_score*100}%;background:#3b82f6"></div></div>
            </div>
            <div style="width:180px;text-align:right"><strong>{combined_score:.3f}</strong></div>
          </div>
        </div>
        """

        # Layer 5 — INCOSE pattern (use analyzer on req_text)
        try:
            incose = self.quality_analyzer._analyze_incose_pattern(req_text)  # may not exist; fallback below
        except Exception:
            incose = None
        incose_html = ""
        if incose:
            best = getattr(incose, 'best_pattern', '') if not isinstance(incose, dict) else incose.get('best_pattern', '')
            comp = getattr(incose, 'compliance_score', '') if not isinstance(incose, dict) else incose.get('compliance_score', '')
            incose_html = f"""
            <div class="card">
              <h3>Layer 5 — INCOSE Pattern Analysis</h3>
              <p class="small">Best pattern: {self._escape(best)} · Compliance: {self._escape(comp)}</p>
            </div>
            """
        else:
            incose_html = """
            <div class="card">
              <h3>Layer 5 — INCOSE Pattern Analysis</h3>
              <p class="small">Pattern analysis not available (run EnhancedRequirementAnalyzer or reqGrading first)</p>
            </div>
            """

        # Layer 6 — Quality dimensions (call analyze_requirement if available)
        try:
            issues, metrics, incose_struct, semantic_struct = self.quality_analyzer.analyze_requirement(req_text)
            overall_score = (metrics.clarity_score + metrics.completeness_score + metrics.verifiability_score + metrics.atomicity_score + metrics.consistency_score) / 5
            grade = self.quality_analyzer._get_grade(overall_score)
            q_html = f"Grade: {grade} · Score: {overall_score:.0f}% · Issues: {len(issues)}"
        except Exception:
            q_html = "Quality analysis not available. Run reqGrading first."
        layer6_html = f"""
        <div class="card">
          <h3>Layer 6 — Quality Dimensions</h3>
          <p class="small">{self._escape(q_html)}</p>
        </div>
        """

        # Layer 7 — Final decision (simple rule mirrored from vizV1)
        try:
            qres = {'grade': grade, 'issues_count': len(issues)}
            if combined_score >= 0.8 and qres['grade'] in ['EXCELLENT', 'GOOD']:
                decision = "ACCEPT MATCH"
                action = "High confidence match with good quality"
                color = self.colors['good']
                symbol = "✓"
            elif combined_score >= 0.35:
                decision = "REVIEW NEEDED"
                action = "Moderate confidence - engineer review required"
                color = self.colors['warning']
                symbol = "?"
            else:
                decision = "ORPHAN"
                action = "No suitable match - write bridge requirement"
                color = self.colors['bad']
                symbol = "✗"
        except Exception:
            decision, action, color, symbol = "UNKNOWN", "Insufficient data", self.colors['neutral'], "—"

        layer7_html = f"""
        <div class="card">
          <h3>Layer 7 — Final Decision</h3>
          <div style="display:flex;align-items:center;gap:18px">
            <div style="font-size:36px;color:{color};font-weight:800">{symbol}</div>
            <div>
              <div style="font-weight:700">{self._escape(decision)}</div>
              <div class="small">{self._escape(action)}</div>
            </div>
          </div>
          <div style="margin-top:8px" class="meta">Match Score: {combined_score:.3f} · Quality: {self._escape(qres.get('grade','N/A'))} · Issues: {qres.get('issues_count','N/A')}</div>
        </div>
        """

        # Compose final HTML document
        now = datetime.now().isoformat()
        css = self._slide_style()
        html_doc = f"""<!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width,initial-scale=1"/>
          <title>{self._escape(title)}</title>
          <style>{css}</style>
        </head>
        <body>
          <div class="slide">
            <div class="title">{self._escape(title)}</div>
            <div class="subtitle">Generated: {self._escape(now)}</div>

            {layer1_html}
            <div style="height:12px"></div>
            {layer2_html}
            <div style="height:12px"></div>
            {layer3_html}
            <div style="height:12px"></div>
            <div class="row">
              <div class="col">{layer4_html}</div>
              <div class="col">{incose_html}{layer6_html}{layer7_html}</div>
            </div>

            <footer style="margin-top:18px;font-size:12px;color:#99a0a6">Source files: aerospace_matches.csv · explanations JSON · requirements.csv</footer>
          </div>
        </body>
        </html>
        """

        out_path = self.output_dir / f"{req_id}_technical_journey.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_doc)

        return str(out_path)

    def create_matching_evaluation_summary_html(self) -> str:
        """
        Read matching_evaluation_report.txt and evaluation_metrics.json (if present)
        and produce a formatted HTML summary saved to visuals output directory.
        """
        eval_txt_path = self.repo_manager.structure['evaluation_results'] / "matching_evaluation_report.txt"
        eval_json_path = self.repo_manager.structure['evaluation_results'] / "evaluation_metrics.json"

        if not eval_txt_path.exists():
            # try other names from outputs
            alt = self.repo_manager.structure['evaluation_results'] / "fixed_simple_evaluation_report.txt"
            if alt.exists():
                eval_txt_path = alt
            else:
                raise FileNotFoundError("matching_evaluation_report.txt not found in evaluation_results")

        raw = eval_txt_path.read_text(encoding='utf-8')

        metrics = {}
        if eval_json_path.exists():
            try:
                metrics = json.loads(eval_json_path.read_text(encoding='utf-8'))
            except Exception:
                metrics = {}

        # Minimal text parsing to extract top-level sections (keep paragraphs and headings)
        # We'll keep the raw report formatted in a mono block and also surface a parsed summary
        # Attempt to extract the KEY INSIGHTS block if present
        insights = []
        for line in raw.splitlines():
            if line.strip().startswith('•'):
                insights.append(line.strip().lstrip('• ').strip())

        # Build HTML
        css = self._slide_style()
        title = "Matching Evaluation Summary"
        now = datetime.now().isoformat()

        # Key metric cards
        metric_cards = []
        if metrics:
            # choose a few important keys (if present)
            picks = ['precision_at_1', 'recall_at_1', 'f1_at_1', 'precision_at_5', 'recall_at_5', 'f1_at_5', 'mrr', 'coverage']
            for k in picks:
                if k in metrics:
                    val = metrics[k]
                    metric_cards.append(f"<div class='card center'><div class='pill'>{self._escape(k)}</div><div style='height:8px'></div><div style='font-weight:800;font-size:20px'>{self._escape(round(val,3) if isinstance(val,(int,float)) else val)}</div></div>")

        metrics_html = "<div class='grid-4'>" + "".join(metric_cards) + "</div>" if metric_cards else "<div class='small'>No metrics JSON found.</div>"

        insights_html = "<ul>" + "".join(f"<li>{self._escape(i)}</li>" for i in insights[:10]) + "</ul>" if insights else "<div class='small'>No bullet insights parsed.</div>"

        full_report_html = f"<pre class='report'>{self._escape(raw)}</pre>"

        html_doc = f"""<!doctype html>
        <html lang="en">
        <head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
        <title>{self._escape(title)}</title><style>{css}</style></head>
        <body>
          <div class="slide">
            <div class="title">{self._escape(title)}</div>
            <div class="subtitle">Generated: {self._escape(now)}</div>

            <div class="card">
              <h3>Run Summary & Key Metrics</h3>
              {metrics_html}
            </div>

            <div class="card">
              <h3>Key Insights (top bullets)</h3>
              {insights_html}
            </div>

            <div class="card" style="margin-top:12px">
              <h3>Full Raw Report</h3>
              {full_report_html}
            </div>

          </div>
        </body>
        </html>
        """

        out_path = self.output_dir / "matching_evaluation_summary.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_doc)

        return str(out_path)


# Example usage (run inside your project):
# from vizV1_html import TechnicalJourneyVisualizerHTML
# viz = TechnicalJourneyVisualizerHTML()
# html_path = viz.create_processing_journey_html(requirement_id='REQ-001')
# eval_path = viz.create_matching_evaluation_summary_html()
# print("Saved journey:", html_path)
# print("Saved evaluation summary:", eval_path)
import argparse
import sys
import pandas as pd

def main():
    """
    Command-line entry point for the HTML visualizer.
    - If no --journey provided, automatically generates HIGH and LOW scoring pairs.
    - Optionally generates matching evaluation summary.
    """
    parser = argparse.ArgumentParser(
        description="HTML-based Technical Journey Visualizer (16:9 slide output)"
    )
    parser.add_argument(
        "--journey", "-j",
        type=str,
        help="Requirement ID to generate the technical journey for (e.g., REQ-001). "
             "If omitted, generates HIGH and LOW scoring examples."
    )
    parser.add_argument(
        "--label", "-l",
        type=str,
        help="Optional label to prefix the slide title (e.g., HIGH or LOW match)"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Generate HTML summary of matching evaluation report"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Slide width in pixels (default: 1920)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Slide height in pixels (default: 1080)"
    )

    args = parser.parse_args()

    viz = TechnicalJourneyVisualizerHTML(width=args.width, height=args.height)

    # Perform actions
    if not args.journey:
        # Load match data to find HIGH and LOW scoring examples
        matches_path = viz.repo_manager.structure['matching_results'] / "aerospace_matches.csv"
        if not matches_path.exists():
            print("❌ No aerospace_matches.csv found. Run matcher first.")
            sys.exit(1)

        df = pd.read_csv(matches_path)
        if df.empty or 'Combined_Score' not in df.columns:
            print("❌ aerospace_matches.csv is missing Combined_Score column.")
            sys.exit(1)

        # Highest scoring pair
        high_row = df.loc[df['Combined_Score'].idxmax()]
        low_row = df.loc[df['Combined_Score'].idxmin()]

        print(f"📊 Creating technical journey for HIGH scoring pair: {high_row['Requirement_ID']}")
        try:
            high_html = viz.create_processing_journey_html(
                requirement_id=high_row['Requirement_ID'],
                label="HIGH"
            )
            print(f"✅ HIGH journey saved to: {high_html}")
        except Exception as e:
            print(f"❌ Error creating HIGH journey: {e}")

        print(f"📊 Creating technical journey for LOW scoring pair: {low_row['Requirement_ID']}")
        try:
            low_html = viz.create_processing_journey_html(
                requirement_id=low_row['Requirement_ID'],
                label="LOW"
            )
            print(f"✅ LOW journey saved to: {low_html}")
        except Exception as e:
            print(f"❌ Error creating LOW journey: {e}")

    else:
        # Single user-specified journey
        try:
            html_path = viz.create_processing_journey_html(
                requirement_id=args.journey,
                label=args.label
            )
            print(f"✅ Technical journey HTML saved to: {html_path}")
        except Exception as e:
            print(f"❌ Error generating journey for {args.journey}: {e}")

    if args.summary:
        try:
            summary_path = viz.create_matching_evaluation_summary_html()
            print(f"✅ Evaluation summary HTML saved to: {summary_path}")
        except Exception as e:
            print(f"❌ Error generating evaluation summary: {e}")


if __name__ == "__main__":
    main()
