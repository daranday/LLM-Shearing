from dispatch import run_pipeline_async
from pipeline import PipelineConfig

if __name__ == "__main__":

    cfg = PipelineConfig(
        from_model_name="NousResearch/Llama-2-7b-hf",
        from_model_size="7b",
        to_model_size="370m",
    )

    run_pipeline_async(exp_name="Async 1.3b -> 370m", config=cfg)
