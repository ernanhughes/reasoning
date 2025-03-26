import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# --- Simulated data loading (replace with your DB/real pipeline) ---
EXAMPLE_DATA = {
    "decision": "Apple saw a 10% drop in iPhone revenue due to supply chain issues.",
    "features": [
        {"id": 7421, "name": "Causal Reasoning", "activation": 0.93},
        {"id": 11982, "name": "Quantitative Comparison", "activation": 0.88},
        {"id": 9903, "name": "Uncertainty Marker", "activation": 0.65},
    ],
    "evidence_tokens": [
        {"token": "drop", "score": 0.91, "context": "10% drop in iPhone revenue"},
        {"token": "supply", "score": 0.89, "context": "due to supply chain issues"},
        {"token": "10%", "score": 0.87, "context": "10% drop in iPhone revenue"},
    ]
}

# --- Simulated signal trigger ---
def evaluate_signal(company):
    # Simulate a confidence score from another app (MARS-style)
    score = random.uniform(0, 1)
    signal = None
    if score > 0.85:
        signal = "Strong Buy"
    elif score < 0.15:
        signal = "Strong Sell"
    return signal, round(score, 2)

def plot_token_scores():
    tokens = EXAMPLE_DATA["evidence_tokens"]
    df_tok = pd.DataFrame(tokens)
    df_tok["score"] = df_tok["score"].round(2)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x="score", y="token", data=df_tok, palette="Blues_d", ax=ax)
    ax.set_xlabel("Activation Score")
    ax.set_ylabel("Token")
    fig.tight_layout()
    return fig

def explain_dashboard(selected_report, top_n):
    signal, score = evaluate_signal(selected_report)
    if signal not in ["Strong Buy", "Strong Sell"]:
        return "No strong signal detected.", "No decision made.", pd.DataFrame(), plt.figure(), ""

    data = EXAMPLE_DATA  # Simulated, would be dynamic
    decision = f"{signal} Signal: {data['decision']}"
    filing_excerpt = (
        f"{selected_report} Report: Apple saw a 10% drop in iPhone revenue due to supply chain issues, "
        f"impacting total gross margin by 2%.\nSignal Confidence: {score}"
    )

    features_df = pd.DataFrame(data["features"][:top_n])
    features_df["activation"] = features_df["activation"].round(3)

    evidence_text = "\n".join(
        f"**{tok['token']}**: \"{tok['context']}\" ({tok['score']})"
        for tok in data["evidence_tokens"][:top_n]
    )

    token_plot = plot_token_scores()

    return filing_excerpt, decision, features_df, token_plot, evidence_text

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Explainability Dashboard for Financial Analysts")

    with gr.Row():
        filing_dropdown = gr.Dropdown(
            choices=["AAPL Q4 2023", "MSFT Q2 2023", "TSLA Q1 2023"],
            value="AAPL Q4 2023",
            label="Select Filing"
        )
        top_n_slider = gr.Slider(1, 10, value=3, step=1, label="Top N Features")

    filing_box = gr.Textbox(label="🧾 Filing Section", lines=3)
    decision_box = gr.Textbox(label="💬 Generated Summary", lines=2)
    feature_table = gr.Dataframe(label="🧬 Activated Reasoning Features")
    token_plot = gr.Plot(label="🔦 Token Activation Scores")
    evidence_box = gr.Markdown(label="📍 Token Context")

    run_btn = gr.Button("Run Explainability Analysis")
    run_btn.click(
        explain_dashboard,
        inputs=[filing_dropdown, top_n_slider],
        outputs=[filing_box, decision_box, feature_table, token_plot, evidence_box]
    )

demo.launch()
