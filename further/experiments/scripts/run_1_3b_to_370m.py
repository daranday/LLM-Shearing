from dispatch import run_pipeline_async
from pipeline import PipelineConfig

if __name__ == "__main__":

    config = PipelineConfig(
        from_model_name="princeton-nlp/Sheared-LLaMA-1.3B",
        from_model_size="1.3b",
        to_model_size="370m",
    )

    run_pipeline_async(exp_name="Async 1.3b -> 370m", config=config)
