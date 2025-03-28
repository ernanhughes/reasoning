import torch
from typing import List, Optional
import os

DEFAULT_KEYWORDS = [
    "reason", "think", "because", "why", "if", "consider",
    "therefore", "hence", "logic", "conclude", "analyze", "inference"
]

def load_reasoning_keywords(path: Optional[str] = None) -> List[str]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_KEYWORDS


def build_reasoning_mask_from_keywords(token_ids: torch.Tensor, tokenizer, keywords: List[str] = None) -> torch.Tensor:
    """
    token_ids: [B, T] tokenized input
    tokenizer: Hugging Face tokenizer
    keywords: list of words to match

    Returns:
        reasoning_mask: [B, T] bool Tensor
    """
    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    reasoning_token_ids = set()
    for kw in keywords:
        ids = tokenizer.encode(kw, add_special_tokens=False)
        if len(ids) == 1:
            reasoning_token_ids.add(ids[0])  # only accept single-token matches

    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for rid in reasoning_token_ids:
        mask |= (token_ids == rid)

    return mask


def compute_reason_scores(activations: torch.Tensor, reasoning_mask: torch.Tensor) -> List[float]:
    """
    activations: [B, T, F] sparse activations from SAE
    reasoning_mask: [B, T] bool mask where reasoning tokens are True

    Returns:
        List of ReasonScores per feature
    """
    B, T, F = activations.shape
    flat_activations = activations.view(-1, F)            # [B*T, F]
    flat_mask = reasoning_mask.view(-1).float()           # [B*T]

    sum_reasoning = (flat_activations.T * flat_mask).T.sum(dim=0)  # [F]
    count_reasoning = flat_mask.sum()

    mean_reasoning = sum_reasoning / count_reasoning.clamp(min=1)

    mean_all = flat_activations.mean(dim=0)               # [F]

    reason_scores = (mean_reasoning / mean_all.clamp(min=1e-6)).tolist()
    return reason_scores

def compute_topk_reason_scores(z: torch.Tensor, reasoning_mask: torch.Tensor, k=10) -> List[float]:
    """
    Computes top-k mean reasoning activations across sparse features.

    Args:
        z: Tensor of shape [B, T, F] — sparse feature activations
        reasoning_mask: Tensor of shape [B, T] — binary mask for reasoning tokens
        k: Number of top active features to consider

    Returns:
        List[float] — mean score per feature across batch (length = F)
    """
    B, T, F = z.shape
    z = z.cpu()
    reasoning_mask = reasoning_mask.cpu()

    feature_scores = torch.zeros(F)
    feature_counts = torch.zeros(F)

    for b in range(B):
        z_b = z[b]                     # [T, F]
        mask_b = reasoning_mask[b]    # [T]

        if mask_b.sum() == 0 or z_b.shape[0] != mask_b.shape[0]:
            continue

        # Mean activation per feature across full sequence
        full_avg = z_b.mean(dim=0)  # [F]

        # Top-k active features in this prompt
        topk_indices = torch.topk(full_avg, k=k).indices

        # Mean activation across reasoning tokens only
        reasoning_avg = z_b[mask_b.bool()].mean(dim=0)  # [F]

        # For each top-k feature, compute ratio of reasoning vs overall
        for f in topk_indices:
            f = f.item()
            if full_avg[f] > 1e-6:  # avoid divide-by-zero
                score = reasoning_avg[f] / full_avg[f]
                feature_scores[f] += score
                feature_counts[f] += 1

    scores = (feature_scores / feature_counts.clamp(min=1)).tolist()
    return scores





def compute_mean_topk_feature_score(hidden_states: torch.Tensor, sae_model, top_k: int = 20) -> float:
    """
    Computes a rough ReasonScore by passing hidden states through SAE and
    averaging the top-K sparse activations.

    Args:
        hidden_states: torch.Tensor of shape [T, H] or [1, T, H]
        sae_model: trained SparseAutoencoder
        top_k: number of features to consider for scoring

    Returns:
        float: Mean top-k activation magnitude
    """
    if hidden_states.dim() == 3:
        hidden_states = hidden_states.squeeze(0)  # [1, T, H] → [T, H]

    sae_input = hidden_states.view(1, -1)  # [T, H] → [1, T*H]
    expected_input_dim = sae_model.config["input_dim"]
    actual_input_dim = sae_input.shape[1]

    if actual_input_dim < expected_input_dim:
        pad_len = expected_input_dim - actual_input_dim
        sae_input = torch.nn.functional.pad(sae_input, (0, pad_len))
    elif actual_input_dim > expected_input_dim:
        sae_input = sae_input[:, :expected_input_dim]

    _, z = sae_model(sae_input)  # z: [1, F]
    z = z.squeeze(0)
    topk_vals, _ = torch.topk(torch.abs(z), k=min(top_k, z.shape[0]))
    return topk_vals.mean().item()

def load_reasoning_token_ids_from_file(path: str) -> List[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reasoning token ID file not found: {path}")
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]


def build_reasoning_mask(token_ids: torch.Tensor, tokenizer, mode="keywords", config=None) -> torch.Tensor:
    """
    Builds a boolean mask [B, T] indicating where reasoning-related tokens are found.
    
    Args:
        token_ids: [B, T] tokenized input
        tokenizer: HF tokenizer
        mode: "keywords" or "tokens"
        config: dict with one of:
            - 'keyword_file': path to keywords.txt
            - 'token_file': path to token IDs file
            - 'token_ids': inline list of token IDs

    Returns:
        reasoning_mask: [B, T] bool Tensor
    """
    if config is None:
        config = {}

    reasoning_token_ids = set()

    if mode == "keywords":
        keywords = load_reasoning_keywords(config.get("keyword_file"))
        for kw in keywords:
            ids = tokenizer.encode(kw, add_special_tokens=False)
            if len(ids) == 1:
                reasoning_token_ids.add(ids[0])
    elif mode == "tokens":
        if "token_file" in config:
            reasoning_token_ids = set(load_reasoning_token_ids_from_file(config["token_file"]))
        else:
            reasoning_token_ids = set(config.get("token_ids", []))
    else:
        raise ValueError(f"Unsupported reason_score.mode: {mode}")

    # Build mask
    mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for rid in reasoning_token_ids:
        mask |= (token_ids == rid)

    return mask
