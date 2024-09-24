from pipeline import PipelineConfig, run_pipeline

if __name__ == "__main__":

    cfg = PipelineConfig(
        from_model_name="NousResearch/Llama-2-7b-hf",
        from_model_size="7b",
        to_model_size="370m",
    )

    run_pipeline(cfg)
