import pytest
from continued_pretraining import run_continued_pretraining
from convert_composer_to_hf import convert_composer_to_hf
from convert_hf_to_composer import convert_hf_to_composer
from convert_to_pruned_model import prune_and_convert_model
from evaluate_model import evaluate_model
from pipeline import PipelineConfig, run_pipeline
from pruning import run_pruning

# NOTE:
#   The steps are:


@pytest.fixture
def pipeline_config():
    return PipelineConfig(
        from_model_name="princeton-nlp/Sheared-LLaMA-1.3B",
        from_model_size="1.3b",
        to_model_size="370m",
        overrides={
            "pruning.max_duration": "2ba",
            "pruning.save_interval": "2ba",
            "continued_pretraining.max_duration": "2ba",
            "continued_pretraining.save_interval": "2ba",
        },
    )


# pruning and continued pretraining


def test_prepare_model(pipeline_config: PipelineConfig):
    convert_hf_to_composer(pipeline_config.hf_to_composer_config)


def test_pruning(pipeline_config: PipelineConfig):
    run_pruning(pipeline_config.pruning_config)


def test_convert_to_pruned_model(pipeline_config: PipelineConfig):
    prune_and_convert_model(pipeline_config.convert_to_pruned_config)


def test_continued_pretraining(pipeline_config: PipelineConfig):
    run_continued_pretraining(pipeline_config.continued_pretraining_config)


# convert results to hf for evaluation


def test_convert_pruned_to_hf(pipeline_config: PipelineConfig):
    convert_composer_to_hf(pipeline_config.pruned_to_hf_config)


def test_convert_continued_pretraining_to_hf(pipeline_config: PipelineConfig):
    convert_composer_to_hf(pipeline_config.continued_pretraining_to_hf_config)


# evaluations


def test_evaluate_from_model(pipeline_config: PipelineConfig):
    evaluate_model(pipeline_config.from_model_eval_config)


def test_evaluate_pruned_model(pipeline_config: PipelineConfig):
    evaluate_model(pipeline_config.pruning_eval_config)


def test_evaluate_continued_pretraining_model(pipeline_config: PipelineConfig):
    evaluate_model(pipeline_config.continued_pretraining_eval_config)


def test_full_pipeline(pipeline_config: PipelineConfig):
    run_pipeline(pipeline_config)
