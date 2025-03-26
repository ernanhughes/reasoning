from reasoning.pipeline.reasoning_pipeline import ReasoningPipeline

pipeline = ReasoningPipeline(config_path="configs/pipeline_config.yaml")
pipeline.process()
