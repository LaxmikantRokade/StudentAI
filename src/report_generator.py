"""
StudentAI — Individual Student Report & PDF Export Module (U5)
==============================================================
Generates publication-quality, executive PDF diagnostic dossiers
for individual students using Python and Matplotlib's native vector
PDF engine.

Key Capabilities:
1. Executive Telemetry Overview:
   - Project title: StudentAI — Student Performance Prediction & Analytics System.
   - Predicted final mark (G3) on the 0-20 scale with academic tier and risk flag.
   - Population baseline and individual feature deflection (Delta).
2. Comprehensive Attribute Audit:
   - Academic history (G1, G2, study time, failures, absences).
   - Support and environment indicators (school/family support, tutoring, health, aspirations).
3. Local Explainability Visualization:
   - Visual attribution waterfall chart directly embedded into the vector PDF.
   - Top positive score boosters and top negative downward pressures with numerical mark deltas.
4. Educational Governance:
   - Personalized actionable academic advisory text.
   - Transparent non-causal disclaimer and confidential academic telemetry metadata.
5. Technical Rigor:
   - 100% offline, in-memory PDF generation using existing dependencies (zero new packages).
   - Robust string sanitization, safe character encoding, and defensive error handling.

Author: StudentAI Intelligence Engineering Team
Lab: 3rd Year B.Tech IT — Data Science & Visualization
"""

import os
import sys
import io
import datetime
import re
from typing import Dict, Any, Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

# Ensure src module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.logger import get_logger

logger = get_logger("report_generator")


def _sanitize_text(text: Any) -> str:
    """
    Cleans string input to prevent encoding issues or unsupported
    glyph rendering in Matplotlib's standard PDF engine.
    """
    if text is None:
        return "N/A"
    clean = str(text).strip()
    # Replace non-printable ASCII or extreme unicode characters that might break PDF fonts
    clean = re.sub(r"[^\x20-\x7E\u00A0-\u00FF\u0391-\u03C9\u2010-\u2022\u0394]", " ", clean)
    return clean


def generate_student_pdf_report(
    student_profile: Dict[str, Any],
    prediction_result: Dict[str, Any],
    explanation_result: Optional[Dict[str, Any]] = None,
    model_name: str = "Production Model (Full History with G1 & G2)",
    student_identifier: Optional[str] = None,
) -> bytes:
    """
    Renders an individual student performance dossier into a vector PDF document.

    Parameters:
        student_profile (dict): Input student indicators.
        prediction_result (dict): Output from predict_student_grade.
        explanation_result (dict, optional): Output from explain_prediction_tree_path.
        model_name (str): Label for the active prediction pipeline.
        student_identifier (str, optional): Custom student ID or label.

    Returns:
        bytes: Complete in-memory PDF binary stream.

    Raises:
        ValueError: If student_profile or prediction_result is invalid.
        RuntimeError: If PDF rendering fails.
    """
    if not isinstance(student_profile, dict) or not student_profile:
        logger.error("Report generation failed: empty or invalid student_profile.")
        raise ValueError("student_profile must be a non-empty dictionary.")

    if not isinstance(prediction_result, dict) or "predicted_grade" not in prediction_result:
        logger.error("Report generation failed: invalid prediction_result dictionary.")
        raise ValueError("prediction_result must contain 'predicted_grade'.")

    try:
        pdf_buffer = io.BytesIO()

        with PdfPages(pdf_buffer) as pdf:
            # Standard US Letter Portrait (8.5 x 11 inches) at 150 DPI
            fig = plt.figure(figsize=(8.5, 11), dpi=150)
            fig.patch.set_facecolor("#030712")

            # Main drawing canvas
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_facecolor("#030712")
            ax.set_xlim(0, 8.5)
            ax.set_ylim(0, 11)
            ax.axis("off")

            # -------------------------------------------------------------
            # 1. HEADER SECTION (SELENE OBSERVATORY DOSSIER)
            # -------------------------------------------------------------
            # Top accent line
            top_bar = patches.Rectangle((0.6, 10.45), 7.3, 0.04, facecolor="#38BDF8", edgecolor="none")
            ax.add_patch(top_bar)

            ax.text(
                0.6, 10.22,
                "STUDENTAI // STUDENT PERFORMANCE INTELLIGENCE PLATFORM",
                color="#38BDF8",
                fontsize=9.5,
                fontweight="bold",
                family="sans-serif",
            )
            ax.text(
                0.6, 9.88,
                "INDIVIDUAL PERFORMANCE TELEMETRY DOSSIER",
                color="#F8FAFC",
                fontsize=15.5,
                fontweight="bold",
                family="sans-serif",
            )

            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            student_label = student_identifier or student_profile.get("student_id", "INDIVIDUAL EVALUATION")
            meta_header = f"RECORD ID: {_sanitize_text(student_label)}  |  GENERATED: {current_timestamp}  |  LOCAL INFERENCE"
            ax.text(
                0.6, 9.62,
                meta_header,
                color="#94A3B8",
                fontsize=7.8,
                family="monospace",
            )

            # Divider
            div1 = patches.Rectangle((0.6, 9.48), 7.3, 0.01, facecolor="#1E293B", edgecolor="none")
            ax.add_patch(div1)

            # -------------------------------------------------------------
            # 2. KEY METRIC HUD CARDS (OUTCOME, TIER, RISK, BASELINE)
            # -------------------------------------------------------------
            pred_grade = float(prediction_result.get("predicted_grade", 0.0))
            tier_name = _sanitize_text(prediction_result.get("tier", "N/A"))
            risk_level = _sanitize_text(prediction_result.get("risk_level", "N/A"))

            # Determine colors based on performance
            if "Excellent" in tier_name or "Good" in tier_name:
                grade_color = "#38BDF8"
                tier_color = "#34D399"
                risk_color = "#34D399"
            elif "Pass" in tier_name:
                grade_color = "#FBBF24"
                tier_color = "#FBBF24"
                risk_color = "#FBBF24"
            else:
                grade_color = "#FB7185"
                tier_color = "#FB7185"
                risk_color = "#FB7185"

            # Card 1: Estimated Grade
            c1_rect = patches.FancyBboxPatch(
                (0.6, 8.42), 2.25, 0.92,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#0B1120", edgecolor="#1E293B", linewidth=1.2
            )
            ax.add_patch(c1_rect)
            ax.text(0.72, 9.18, "// ESTIMATE.FINAL_GRADE", color="#38BDF8", fontsize=7.2, family="monospace")
            ax.text(0.72, 8.78, f"{pred_grade:.1f}", color=grade_color, fontsize=19, fontweight="bold")
            ax.text(1.35, 8.80, "/ 20 marks", color="#94A3B8", fontsize=9)
            ax.text(0.72, 8.52, "Continuous G3 Forecast", color="#64748B", fontsize=7.5)

            # Card 2: Performance Tier
            c2_rect = patches.FancyBboxPatch(
                (3.12, 8.42), 2.25, 0.92,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#0B1120", edgecolor="#1E293B", linewidth=1.2
            )
            ax.add_patch(c2_rect)
            ax.text(3.24, 9.18, "// PERFORMANCE.TIER", color="#94A3B8", fontsize=7.2, family="monospace")
            tier_main = tier_name.split("(")[0].strip()
            tier_sub = "(" + tier_name.split("(")[-1].strip() if "(" in tier_name else ""
            ax.text(3.24, 8.82, tier_main, color=tier_color, fontsize=12, fontweight="bold")
            ax.text(3.24, 8.58, tier_sub, color="#CBD5E1", fontsize=7.8)

            # Card 3: Academic Risk Status
            c3_rect = patches.FancyBboxPatch(
                (5.65, 8.42), 2.25, 0.92,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#0B1120", edgecolor="#1E293B", linewidth=1.2
            )
            ax.add_patch(c3_rect)
            ax.text(5.77, 9.18, "// STATUS.ADVISORY", color="#94A3B8", fontsize=7.2, family="monospace")
            ax.text(5.77, 8.82, risk_level, color=risk_color, fontsize=13, fontweight="bold")
            ax.text(5.77, 8.58, "Formative Support Level", color="#CBD5E1", fontsize=7.8)

            # -------------------------------------------------------------
            # 3. STUDENT ATTRIBUTE AUDIT (TABULAR PROFILE)
            # -------------------------------------------------------------
            ax.text(0.6, 8.16, "STUDENT ACADEMIC & BEHAVIORAL PROFILE", color="#F8FAFC", fontsize=9.5, fontweight="bold")

            # Profile Grid Background
            grid_rect = patches.FancyBboxPatch(
                (0.6, 6.78), 7.3, 1.25,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#080E1C", edgecolor="#1E293B", linewidth=1.0
            )
            ax.add_patch(grid_rect)

            # Format study time text
            studytime_val = student_profile.get("studytime", 2)
            studytime_map = {1: "< 2 hrs/wk", 2: "2-5 hrs/wk", 3: "5-10 hrs/wk", 4: "> 10 hrs/wk"}
            studytime_str = studytime_map.get(studytime_val, f"Cat {studytime_val}")

            is_behavioral = "Behavioral" in model_name or ("G1" not in student_profile and "G2" not in student_profile)
            g1_str = "Pre-Exam" if is_behavioral else f"{student_profile.get('G1', 'N/A')}/20"
            g2_str = "Pre-Exam" if is_behavioral else f"{student_profile.get('G2', 'N/A')}/20"

            # Column 1 items (Academic History)
            ax.text(0.8, 7.78, "First Period Mark (G1):", color="#94A3B8", fontsize=7.8)
            ax.text(2.1, 7.78, g1_str, color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(0.8, 7.52, "Second Period Mark (G2):", color="#94A3B8", fontsize=7.8)
            ax.text(2.1, 7.52, g2_str, color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(0.8, 7.26, "Weekly Study Time:", color="#94A3B8", fontsize=7.8)
            ax.text(2.1, 7.26, studytime_str, color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(0.8, 7.00, "School Absences Count:", color="#94A3B8", fontsize=7.8)
            ax.text(2.1, 7.00, f"{student_profile.get('absences', 0)} days", color="#F8FAFC", fontsize=7.8, fontweight="bold")

            # Column 2 items (Background & Failures)
            ax.text(3.3, 7.78, "Past Class Failures:", color="#94A3B8", fontsize=7.8)
            ax.text(4.45, 7.78, str(student_profile.get("failures", 0)), color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(3.3, 7.52, "School Academic Support:", color="#94A3B8", fontsize=7.8)
            ax.text(4.45, 7.52, str(student_profile.get("schoolsup", "no")).upper(), color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(3.3, 7.26, "Family Academic Support:", color="#94A3B8", fontsize=7.8)
            ax.text(4.45, 7.26, str(student_profile.get("famsup", "yes")).upper(), color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(3.3, 7.00, "Extra Paid Tutoring:", color="#94A3B8", fontsize=7.8)
            ax.text(4.45, 7.00, str(student_profile.get("paid", "no")).upper(), color="#F8FAFC", fontsize=7.8, fontweight="bold")

            # Column 3 items (Lifestyle & Aspirations)
            ax.text(5.55, 7.78, "Higher Education Goal:", color="#94A3B8", fontsize=7.8)
            ax.text(6.8, 7.78, str(student_profile.get("higher", "yes")).upper(), color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(5.55, 7.52, "Home Internet Access:", color="#94A3B8", fontsize=7.8)
            ax.text(6.8, 7.52, str(student_profile.get("internet", "yes")).upper(), color="#F8FAFC", fontsize=7.8, fontweight="bold")

            health_val = student_profile.get("health", 4)
            ax.text(5.55, 7.26, "Self-Reported Health:", color="#94A3B8", fontsize=7.8)
            ax.text(6.8, 7.26, f"{health_val}/5", color="#F8FAFC", fontsize=7.8, fontweight="bold")

            ax.text(5.55, 7.00, "Evaluation Pipeline:", color="#94A3B8", fontsize=7.8)
            pipe_short = "Behavioral Mode" if is_behavioral else "Production RF"
            ax.text(6.8, 7.00, pipe_short, color="#38BDF8", fontsize=7.8, fontweight="bold")

            # -------------------------------------------------------------
            # 4. PREDICTION EXPLANATION (TREE PATH ATTRIBUTION)
            # -------------------------------------------------------------
            ax.text(0.6, 6.50, "LOCAL EXPLAINABILITY & FEATURE ATTRIBUTION (SAABAS DECOMPOSITION)", color="#F8FAFC", fontsize=9.5, fontweight="bold")

            expl_rect = patches.FancyBboxPatch(
                (0.6, 3.65), 7.3, 2.70,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#0B1120", edgecolor="#1E293B", linewidth=1.0
            )
            ax.add_patch(expl_rect)

            if explanation_result and explanation_result.get("supported", False):
                base_val = explanation_result.get("base_value", 10.34)
                delta = pred_grade - base_val
                delta_sign = "+" if delta >= 0 else ""
                delta_col = "#34D399" if delta >= 0 else "#FB7185"

                # Telemetry line
                ax.text(
                    0.8, 6.08,
                    f"POPULATION BASELINE: {base_val:.2f} / 20  |  NET FEATURE ATTRIBUTION: {delta_sign}{delta:.2f} MARKS  |  RECONSTRUCTED: {explanation_result.get('reconstructed_grade', pred_grade):.2f}",
                    color=delta_col,
                    fontsize=8.0,
                    family="monospace",
                    fontweight="bold",
                )

                # Positive and Negative Drivers
                pos_list = explanation_result.get("positive_contributions", [])[:4]
                neg_list = explanation_result.get("negative_contributions", [])[:4]

                ax.text(0.8, 5.80, "Score Lifting Factors (Positive Attributions):", color="#34D399", fontsize=8.2, fontweight="bold")
                if pos_list:
                    y_pos = 5.56
                    for item in pos_list:
                        disp = _sanitize_text(item.get("display_text", item.get("feature", "")))
                        contrib = item.get("contribution", 0.0)
                        ax.text(0.9, y_pos, f"+ {disp}", color="#E2E8F0", fontsize=7.6)
                        ax.text(3.6, y_pos, f"+{contrib:.2f} marks", color="#34D399", fontsize=7.6, fontweight="bold", ha="right")
                        y_pos -= 0.22
                else:
                    ax.text(0.9, 5.56, "No major positive deflection drivers identified.", color="#64748B", fontsize=7.6)

                ax.text(4.4, 5.80, "Downward Pressure Factors (Negative Attributions):", color="#FB7185", fontsize=8.2, fontweight="bold")
                if neg_list:
                    y_neg = 5.56
                    for item in neg_list:
                        disp = _sanitize_text(item.get("display_text", item.get("feature", "")))
                        contrib = item.get("contribution", 0.0)
                        ax.text(4.5, y_neg, f"- {disp}", color="#E2E8F0", fontsize=7.6)
                        ax.text(7.6, y_neg, f"{contrib:.2f} marks", color="#FB7185", fontsize=7.6, fontweight="bold", ha="right")
                        y_neg -= 0.22
                else:
                    ax.text(4.5, 5.56, "No downward pressure factors identified.", color="#64748B", fontsize=7.6)

                # Embedded Attribution Bar Chart (Inside the lower half of explanation block)
                chart_items = pos_list[::-1] + neg_list[::-1]
                if chart_items:
                    c_ax = fig.add_axes([0.13, 0.355, 0.74, 0.095])
                    c_ax.set_facecolor("#080E1C")
                    labels = [_sanitize_text(i.get("display_text", ""))[:28] for i in chart_items]
                    vals = [i.get("contribution", 0.0) for i in chart_items]
                    cols = ["#34D399" if v >= 0 else "#FB7185" for v in vals]

                    y_pos = range(len(labels))
                    c_ax.barh(y_pos, vals, color=cols, height=0.55, edgecolor="none")
                    c_ax.axvline(0, color="#64748B", linestyle="--", linewidth=0.8)
                    c_ax.set_yticks(y_pos)
                    c_ax.set_yticklabels(labels, fontsize=6.5, color="#94A3B8")
                    c_ax.tick_params(colors="#94A3B8", labelsize=6.5)
                    c_ax.set_xlabel("Mark Contribution Relative to Population Mean", fontsize=6.8, color="#CBD5E1")
                    for sp in c_ax.spines.values():
                        sp.set_color("#1E293B")
            else:
                ax.text(0.8, 5.2, "Feature attribution tree path decomposition not supported for current model configuration.", color="#94A3B8", fontsize=8.2)

            # -------------------------------------------------------------
            # 5. ACTIONABLE EDUCATIONAL ADVISORY
            # -------------------------------------------------------------
            ax.text(0.6, 3.40, "ACTIONABLE EDUCATIONAL GUIDANCE & RECOMMENDATIONS", color="#F8FAFC", fontsize=9.5, fontweight="bold")

            adv_rect = patches.FancyBboxPatch(
                (0.6, 2.30), 7.3, 0.95,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#0B1120", edgecolor=risk_color, linewidth=1.2
            )
            ax.add_patch(adv_rect)

            advisory_text = _sanitize_text(prediction_result.get("advisory", "Maintain consistent study habits."))
            ax.text(
                0.78, 2.95,
                "PERSONALIZED MENTORING ADVISORY:",
                color=risk_color,
                fontsize=8.0,
                fontweight="bold",
                family="monospace",
            )
            # Wrapped advisory text
            import textwrap
            wrapped_advisory = textwrap.fill(advisory_text, width=88)
            ax.text(
                0.78, 2.75,
                wrapped_advisory,
                color="#CBD5E1",
                fontsize=8.2,
                family="sans-serif",
                va="top",
            )

            # -------------------------------------------------------------
            # 6. ETHICAL MACHINE LEARNING DISCLAIMER
            # -------------------------------------------------------------
            disc_rect = patches.FancyBboxPatch(
                (0.6, 1.15), 7.3, 0.95,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#080E1C", edgecolor="#1E293B", linewidth=1.0
            )
            ax.add_patch(disc_rect)

            ax.text(0.78, 1.88, "ETHICAL MACHINE LEARNING & NON-CAUSALITY NOTICE:", color="#FBBF24", fontsize=7.8, fontweight="bold", family="monospace")
            disclaimer_body = (
                "Predictions and feature attributions are empirical statistical approximations derived from historical patterns in "
                "the UCI Student Performance benchmark. They quantify associative model mechanics, NOT guaranteed physical or educational causality. "
                "This report must never be used as a definitive outcome or sole basis for disciplinary tracking. Targeted mentoring, active "
                "classroom engagement, and continuous formative assessment significantly alter real academic achievements."
            )
            wrapped_disc = textwrap.fill(disclaimer_body, width=95)
            ax.text(0.78, 1.70, wrapped_disc, color="#94A3B8", fontsize=7.0, family="sans-serif", va="top")

            # -------------------------------------------------------------
            # 7. TELEMETRY FOOTER
            # -------------------------------------------------------------
            footer_text = (
                "STUDENTAI OBSERVATORY  |  THIRD YEAR B.TECH IT LAB  |  "
                "100% LOCAL INFERENCE  |  PROBABILISTIC ESTIMATE"
            )
            ax.text(4.25, 0.72, footer_text, color="#64748B", fontsize=6.8, ha="center", family="monospace")

            # Save page to in-memory PDF
            pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)

        pdf_bytes = pdf_buffer.getvalue()
        logger.info("Generated student PDF report successfully (%d bytes).", len(pdf_bytes))
        return pdf_bytes

    except Exception as exc:
        logger.error("Failed to generate PDF report: %s", exc, exc_info=True)
        raise RuntimeError(f"PDF report generation failed: {exc}")
