import torch
import torch.nn.functional as F

def preprocess_for_sae(hidden: torch.Tensor, expected_input_dim: int) -> torch.Tensor:
    """
    Flattens and pads/truncates hidden states to match the SAE input_dim.
    
    Args:
        hidden: [B, T, H] tensor from a transformer layer
        expected_input_dim: int (e.g. 64 * 2048)

    Returns:
        Tensor of shape [B, expected_input_dim]
    """
    sae_input = hidden.view(hidden.size(0), -1)  # [B, T*H]
    actual_input_dim = sae_input.shape[1]

    if actual_input_dim < expected_input_dim:
        pad_len = expected_input_dim - actual_input_dim
        sae_input = F.pad(sae_input, (0, pad_len))
    elif actual_input_dim > expected_input_dim:
        sae_input = sae_input[:, :expected_input_dim]

    return sae_input
