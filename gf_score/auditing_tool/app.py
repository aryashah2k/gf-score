"""
GF-Score Interactive Auditing Dashboard
=======================================
Gradio-based web interface for fairness-aware robustness auditing.
Supports both CIFAR-10 and ImageNet evaluation results.

Usage:
    python -m gf_score.auditing_tool.app
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from gf_score.config import (
    RESULTS_DIR,
    FIGURES_DIR,
    DATA_DIR,
    get_dataset_config,
)

logger = logging.getLogger("gf_score.auditing_tool.app")

# WNID-to-human-readable-name mapping for ImageNet
_wnid_to_name_cache = None


def load_wnid_to_name():
    """Load ImageNet WNID-to-human-readable-name mapping (cached)."""
    global _wnid_to_name_cache
    if _wnid_to_name_cache is not None:
        return _wnid_to_name_cache
    path = DATA_DIR / "imagenet" / "wnid_to_name.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            _wnid_to_name_cache = json.load(f)
    else:
        _wnid_to_name_cache = {}
        logger.warning(
            f"WNID-to-name mapping not found at {path}. "
            f"Run: python scripts/build_imagenet_labels.py"
        )
    return _wnid_to_name_cache


def readable_class(cls_id: str, dataset: str) -> str:
    """Return a human-readable class name.

    For CIFAR-10: returns the class name as-is (already readable).
    For ImageNet: returns 'readable_name (wnid)' if mapping available.
    """
    if dataset != "imagenet":
        return cls_id
    mapping = load_wnid_to_name()
    name = mapping.get(cls_id)
    if name:
        return f"{name} ({cls_id})"
    return cls_id


# ======================================================================
# Data loading helpers
# ======================================================================

def load_results(dataset: str = "cifar10"):
    """Load evaluation results for a given dataset."""
    suffix = f"_{dataset}" if dataset != "cifar10" else ""
    path = RESULTS_DIR / f"full_results{suffix}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_available_datasets():
    """Detect which datasets have completed evaluation results."""
    available = []
    if (RESULTS_DIR / "full_results.json").exists():
        available.append("cifar10")
    if (RESULTS_DIR / "full_results_imagenet.json").exists():
        available.append("imagenet")
    return available


def get_class_names(results):
    """Extract class names from results metadata or model data."""
    if results is None:
        return []
    metadata = results.get("metadata", {})
    class_names = metadata.get("class_names")
    if class_names:
        return class_names
    # Fallback: extract from first model's per_class_scores keys
    model_results = results.get("model_results", {})
    if model_results:
        first_model = next(iter(model_results.values()))
        return list(first_model.get("per_class_scores", {}).keys())
    return []


def get_short_names(dataset: str):
    """Get model short names for a dataset."""
    ds_cfg = get_dataset_config(dataset)
    return ds_cfg["short_names"]


def get_model_choices(results, dataset: str):
    """Get list of model display names."""
    if results is None:
        return []
    short_names = get_short_names(dataset)
    return [
        short_names.get(m, m) for m in results["model_results"].keys()
    ]


def get_model_name_from_display(display_name, results, dataset: str):
    """Reverse lookup: display name -> full model name."""
    short_names = get_short_names(dataset)
    for full_name, short_name in short_names.items():
        if short_name == display_name and full_name in results["model_results"]:
            return full_name
    # Fallback: try direct match
    if display_name in results["model_results"]:
        return display_name
    return None


def get_dataset_label(dataset: str):
    """Human-readable dataset label for display."""
    labels = {
        "cifar10": "CIFAR-10 (10 classes, L2 threat model)",
        "imagenet": "ImageNet (1000 classes, L∞ threat model)",
    }
    return labels.get(dataset, dataset)


# ======================================================================
# Analysis logic
# ======================================================================

def analyze_model(model_display_name, lambda_val, results, dataset):
    """Analyze a single model and return formatted results."""
    if results is None:
        return "No results found. Run evaluation first.", None, None

    model_name = get_model_name_from_display(model_display_name, results, dataset)
    if model_name is None:
        return f"Model '{model_display_name}' not found.", None, None

    r = results["model_results"][model_name]
    class_names = get_class_names(results)
    num_classes = len(class_names)

    # Dynamically compute FP-GREAT with user's lambda
    agg_score = r['aggregate_great_score']
    rdi = r['rdi']
    fp_great_dynamic = agg_score - lambda_val * rdi

    # Context values
    fp_at_0 = agg_score
    fp_at_1 = agg_score - rdi

    # Determine interpretation
    if lambda_val == 0:
        fp_interp = "No fairness penalty applied (= GREAT Score)"
    elif lambda_val < 0.3:
        fp_interp = "Mild fairness adjustment"
    elif lambda_val < 0.7:
        fp_interp = "Balanced robustness-fairness trade-off"
    else:
        fp_interp = "Strong fairness emphasis"

    # Dataset info
    ds_label = "CIFAR-10" if dataset == "cifar10" else "ImageNet"
    threat_model = "L2" if dataset == "cifar10" else "L∞"

    # WCR class readable name
    wcr_class_display = readable_class(r['wcr_class'], dataset)

    # Summary text
    summary = f"""## Model: {model_display_name}
**Dataset:** {ds_label} | **Threat Model:** {threat_model} | **Classes:** {num_classes}

### Aggregate Metrics
| Metric | Value | Interpretation |
|--------|-------|---------------|
| **GREAT Score** | {agg_score:.4f} | Certified robustness lower bound |
| **RDI** | {rdi:.4f} | {'Low ✅' if rdi < 0.1 else 'Moderate ⚠️' if rdi < 0.3 else 'High ❌'} disparity |
| **NRGC** | {r['nrgc']:.4f} | Robustness inequality (Gini) |
| **WCR** | {r['wcr']:.4f} | Worst-case robustness ({wcr_class_display}) |

---

### 🎛️ Fairness-Penalized Score (FP-GREAT)

**FP-GREAT = Ω̄ − λ × RDI = {agg_score:.4f} − {lambda_val:.2f} × {rdi:.4f} = {fp_great_dynamic:.4f}**

{fp_interp}

| λ Value | FP-GREAT | Meaning |
|---------|----------|---------|
| λ=0.00 | {fp_at_0:.4f} | Pure robustness (no fairness penalty) |
| **λ={lambda_val:.2f}** | **{fp_great_dynamic:.4f}** | **← Current setting** |
| λ=1.00 | {fp_at_1:.4f} | Maximum fairness penalty |

---

### Per-Class Robustness Scores
"""

    # For large datasets (ImageNet), show top/bottom 10 instead of all 1000
    per_class_scores = r.get("per_class_scores", {})
    per_class_accuracy = r.get("per_class_accuracy", {})

    if num_classes > 30:
        # Sort by score and show bottom 10 + top 10
        sorted_classes = sorted(per_class_scores.keys(), key=lambda c: per_class_scores.get(c, 0))
        bottom_10 = sorted_classes[:10]
        top_10 = sorted_classes[-10:]

        max_score = max(per_class_scores.values()) if per_class_scores else 1.0

        summary += f"*Showing bottom 10 and top 10 of {num_classes} classes (sorted by score):*\n\n"

        summary += "**Bottom 10 (Most Vulnerable):**\n\n"
        summary += "| Class | Score | Accuracy | Visual |\n"
        summary += "|-------|-------|----------|--------|\n"
        for cls in bottom_10:
            score = per_class_scores.get(cls, 0)
            acc = per_class_accuracy.get(cls, 0)
            bar_len = int((score / max(max_score, 0.001)) * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            cls_display = readable_class(cls, dataset)
            summary += f"| {cls_display} | {score:.4f} | {acc:.1%} | {bar} |\n"

        summary += "\n**Top 10 (Most Robust):**\n\n"
        summary += "| Class | Score | Accuracy | Visual |\n"
        summary += "|-------|-------|----------|--------|\n"
        for cls in top_10:
            score = per_class_scores.get(cls, 0)
            acc = per_class_accuracy.get(cls, 0)
            bar_len = int((score / max(max_score, 0.001)) * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            cls_display = readable_class(cls, dataset)
            summary += f"| {cls_display} | {score:.4f} | {acc:.1%} | {bar} |\n"
    else:
        # For CIFAR-10, show all classes
        max_score = max(per_class_scores.get(cls, 0) for cls in class_names) if class_names else 1.0
        summary += "| Class | Score | Accuracy | Visual |\n"
        summary += "|-------|-------|----------|--------|\n"
        for cls in class_names:
            score = per_class_scores.get(cls, 0)
            acc = per_class_accuracy.get(cls, 0)
            bar_len = int((score / max(max_score, 0.001)) * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            cls_display = readable_class(cls, dataset)
            summary += f"| {cls_display} | {score:.4f} | {acc:.1%} | {bar} |\n"

    # Vulnerability ranking
    vuln = r.get("vulnerability_ranking", [])
    if vuln:
        summary += "\n### Vulnerability Ranking"
        if num_classes > 30:
            summary += f" (Top 10 most vulnerable of {num_classes})\n"
            vuln_display = vuln[:10]
        else:
            summary += " (Most → Least Vulnerable)\n"
            vuln_display = vuln

        for rank, (cls, score) in enumerate(vuln_display, 1):
            emoji = "🔴" if rank <= 3 else "🟡" if rank <= 7 else "🟢"
            cls_display = readable_class(cls, dataset)
            summary += f"{rank}. {emoji} **{cls_display}**: {score:.4f}\n"

    # Per-class scores compact text
    if num_classes <= 30:
        scores_text = "\n".join(
            f"- {cls}: {per_class_scores.get(cls, 0):.4f}"
            for cls in class_names
        )
    else:
        scores_text = f"({num_classes} classes total, see report for details)"

    # Generate HTML report
    from gf_score.auditing_tool.report_generator import generate_report

    result_with_lambda = dict(r)
    result_with_lambda["fp_great"] = fp_great_dynamic
    result_with_lambda["fp_lambda"] = lambda_val

    metadata = results.get("metadata", {})
    report_path = generate_report(
        model_name=model_display_name,
        model_result=result_with_lambda,
        data_source=metadata.get("data_source", f"{ds_label} test"),
        total_samples=metadata.get("total_samples", 0),
    )

    return summary, report_path, scores_text


# ======================================================================
# Gradio App
# ======================================================================

def build_app():
    """Build and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio not installed. Install with: pip install gradio>=4.0.0")
        sys.exit(1)

    available_datasets = get_available_datasets()
    if not available_datasets:
        print("WARNING: No evaluation results found. Run evaluation first.")
        available_datasets = ["cifar10"]

    # Pre-load results for the default dataset
    default_dataset = available_datasets[0]
    results_cache = {}
    for ds in available_datasets:
        results_cache[ds] = load_results(ds)

    with gr.Blocks(
        title="GF-Score Auditing Dashboard",
        theme=gr.themes.Soft(),
    ) as app:
        # State to hold current dataset and results
        current_dataset = gr.State(default_dataset)
        current_results = gr.State(results_cache.get(default_dataset))

        gr.Markdown("""
# 🛡️ GF-Score: Fairness-Aware Robustness Auditing Dashboard

Evaluate AI models for **class-conditional adversarial robustness** with fairness metrics.
Based on GREAT Score (NeurIPS 2024) extended with per-class decomposition and disparity analysis.

> **Tip**: Select a **dataset**, choose a **model**, adjust the **λ slider**, and click **Analyze** to inspect the model's per-class robustness profile.
> - λ=0: Pure robustness ranking (no fairness adjustment)
> - λ=1: Maximum fairness penalty (heavily penalizes class disparity)
        """)

        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                choices=[(get_dataset_label(ds), ds) for ds in available_datasets],
                label="Dataset",
                value=default_dataset,
                scale=2,
            )
            model_dropdown = gr.Dropdown(
                choices=get_model_choices(results_cache.get(default_dataset), default_dataset),
                label="Select Model",
                value=(
                    get_model_choices(results_cache.get(default_dataset), default_dataset)[0]
                    if results_cache.get(default_dataset)
                    else None
                ),
                scale=2,
            )

        with gr.Row():
            lambda_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                label="Fairness Penalty (λ)",
                info="Controls FP-GREAT: score = GREAT − λ × RDI",
                scale=3,
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)

        with gr.Row():
            summary_output = gr.Markdown(label="Analysis Results")

        with gr.Row():
            report_output = gr.Textbox(label="Audit Report Path", interactive=False)
            scores_output = gr.Textbox(label="Per-Class Scores", interactive=False, visible=False)

        # ---- Event handlers ----

        def on_dataset_change(dataset_choice):
            """When dataset changes, update model dropdown and reload results."""
            results = results_cache.get(dataset_choice)
            if results is None:
                results = load_results(dataset_choice)
                results_cache[dataset_choice] = results

            choices = get_model_choices(results, dataset_choice)
            default_model = choices[0] if choices else None

            return (
                gr.update(choices=choices, value=default_model),  # model_dropdown
                dataset_choice,   # current_dataset state
                results,          # current_results state
            )

        dataset_dropdown.change(
            fn=on_dataset_change,
            inputs=[dataset_dropdown],
            outputs=[model_dropdown, current_dataset, current_results],
        )

        def run_analysis(model_name, lambda_val, dataset, results):
            summary, report_path, scores = analyze_model(
                model_name, lambda_val, results, dataset
            )
            return summary, report_path or "", scores or ""

        analyze_btn.click(
            fn=run_analysis,
            inputs=[model_dropdown, lambda_slider, current_dataset, current_results],
            outputs=[summary_output, report_output, scores_output],
        )

        # Real-time updates on slider release
        lambda_slider.release(
            fn=run_analysis,
            inputs=[model_dropdown, lambda_slider, current_dataset, current_results],
            outputs=[summary_output, report_output, scores_output],
        )

        gr.Markdown("""
---
*GF-Score v0.1.0 | Metrics: RDI (Max Group Disparity), NRGC (Gini Index),
WCR (Rawlsian Maximin), FP-GREAT (Inequality-Adjusted Welfare)*
        """)

    return app


def main():
    app = build_app()
    print("\n" + "=" * 60)
    print("GF-SCORE AUDITING DASHBOARD")
    print("=" * 60)
    print("Launching Gradio interface...")
    print("Access at: http://localhost:7860")
    print("=" * 60)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
