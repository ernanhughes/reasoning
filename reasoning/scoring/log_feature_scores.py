from reasoning.logs.feature_log import FeatureVectorLog
from datetime import datetime, timezone

def log_feature_scores(prompt_id: str, sae_config: str, layer_index: int,
                       reason_scores: list[float], topk_scores: list[float]):
    log = FeatureVectorLog(
        prompt_id=prompt_id,
        reason_scores=reason_scores,
        topk_scores=topk_scores,
        sae_config=sae_config_path,
        layer_index=layer_index
    )
    log.save()

