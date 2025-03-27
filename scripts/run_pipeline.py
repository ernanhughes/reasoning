import hydra
from omegaconf import DictConfig, OmegaConf
from reasoning.pipeline.reasoning_pipeline import ReasoningPipeline

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("🧠 Running pipeline with config:")
    print(OmegaConf.to_yaml(cfg))

    pipeline = ReasoningPipeline(cfg.pipeline)
    pipeline.process()

if __name__ == "__main__":
    main()
