import argparse
import json
import torch
import numpy as np
from train_sae import SparseAutoencoder

def score_to_rgb(score, max_score):
    norm = min(score / max_score, 1.0)
    red = int(255 * norm)
    green = int(255 * (1 - norm))
    return f"rgb({red}, {green}, 0)"

def score_to_markdown(token, score, max_score):
    norm = score / max_score if max_score else 0
    if norm > 0.66:
        return f"**`{token}`**"
    elif norm > 0.33:
        return f"`{token}`"
    else:
        return token

def generate_html(samples, model, feature_idx, output_path):
    html = ['<html><body><h1>Token Activations</h1>']
    for i, sample in enumerate(samples):
        tokens = sample["text"].split()
        acts = np.array(sample["activations"])
        with torch.no_grad():
            x = torch.tensor(acts, dtype=torch.float32)
            _, h = model(x)
        scores = h[:, feature_idx].numpy()
        max_score = scores.max()

        html.append(f"<h3>Sample {i+1}</h3><p>")
        for token, score in zip(tokens, scores):
            color = score_to_rgb(score, max_score)
            html.append(f'<span style="background-color:{color};padding:2px;margin:2px;border-radius:4px;">{token}</span> ')
        html.append("</p>")
    html.append("</body></html>")
    with open(output_path, "w") as f:
        f.write("\n".join(html))
    print(f"Saved HTML report to {output_path}")

def generate_markdown(samples, model, feature_idx, output_path):
    lines = ["# Token Activation Heatmap (Markdown)"]
    for i, sample in enumerate(samples):
        tokens = sample["text"].split()
        acts = np.array(sample["activations"])
        with torch.no_grad():
            x = torch.tensor(acts, dtype=torch.float32)
            _, h = model(x)
        scores = h[:, feature_idx].numpy()
        max_score = scores.max()
        lines.append(f"\n### Sample {i+1}\n")
        lines.append(" ".join([score_to_markdown(tok, sc, max_score) for tok, sc in zip(tokens, scores)]))
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved Markdown report to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize SAE feature token activations")
    parser.add_argument("--json_path", required=True, help="Path to activations_with_text.json")
    parser.add_argument("--sae_model_path", required=True, help="Path to SAE model (.pt)")
    parser.add_argument("--feature_idx", type=int, required=True, help="Feature index to visualize")
    parser.add_argument("--output_path", required=True, help="Output path (html or md)")
    parser.add_argument("--format", choices=["html", "md"], default="html", help="Output format")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")

    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    samples = data[:args.num_samples]
    input_dim = len(samples[0]["activations"][0])
    model = SparseAutoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load(args.sae_model_path, map_location="cpu"))
    model.eval()

    if args.format == "html":
        generate_html(samples, model, args.feature_idx, args.output_path)
    else:
        generate_markdown(samples, model, args.feature_idx, args.output_path)

if __name__ == "__main__":
    main()
