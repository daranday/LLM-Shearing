from pipeline import PipelineConfig, run_pipeline

if __name__ == "__main__":
    pipeline_config = PipelineConfig(
        from_model_name="princeton-nlp/Sheared-LLaMA-1.3B",
        from_model_size="1.3b",
        to_model_size="370m",
    )
    run_pipeline(pipeline_config)
