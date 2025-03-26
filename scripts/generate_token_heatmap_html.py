import json
import torch
import numpy as np
from train_sae import SparseAutoencoder

def score_to_rgb(score, max_score):
    norm = min(score / max_score, 1.0)
    red = int(255 * norm)
    green = int(255 * (1 - norm))
    return f"rgb({red}, {green}, 0)"  # from green → red

def generate_html_report(json_path, sae_model_path, feature_idx, output_path, num_samples=5):
    with open(json_path, "r") as f:
        data = json.load(f)

    model = SparseAutoencoder(input_dim=len(data[0]["activations"][0]))
    model.load_state_dict(torch.load(sae_model_path, map_location="cpu"))
    model.eval()

    html = ['<html><body><h1>Token Activations</h1>']

    for i, sample in enumerate(data[:num_samples]):
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


def score_to_markdown(token, score, max_score):
    norm = score / max_score if max_score else 0
    if norm > 0.66:
        return f"**`{token}`**"
    elif norm > 0.33:
        return f"`{token}`"
    else:
        return token

def generate_markdown_report(json_path, sae_model_path, feature_idx, output_path, num_samples=5):
    with open(json_path, "r") as f:
        data = json.load(f)

    model = SparseAutoencoder(input_dim=len(data[0]["activations"][0]))
    model.load_state_dict(torch.load(sae_model_path, map_location="cpu"))
    model.eval()

    lines = ["# Token Activation Heatmap (Markdown)"]

    for i, sample in enumerate(data[:num_samples]):
        tokens = sample["text"].split()
        acts = np.array(sample["activations"])

        with torch.no_grad():
            x = torch.tensor(acts, dtype=torch.float32)
            _, h = model(x)

        scores = h[:, feature_idx].numpy()
        max_score = scores.max()

        lines.append(f"\n### Sample {i+1}\n")
        highlighted = [score_to_markdown(tok, sc, max_score) for tok, sc in zip(tokens, scores)]
        lines.append(" ".join(highlighted))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown report to {output_path}")


if __name__ == "__main__":
    generate_html_report(
        json_path="../data/activations_with_text.json",
        sae_model_path="../models/sae_tinyllama.pt",
        feature_idx=17456 % (2048 * 16),
        output_path="../reports/token_heatmap.html"
    )
