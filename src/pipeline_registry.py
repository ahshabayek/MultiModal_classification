from hateful_memes.pipelines import (
    clip_pipeline,
    data_preprocessing,
    evaluation,
    vilbert_pipeline,
    visualbert_pipeline,
)
from kedro.pipeline import Pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register all pipelines"""

    data_pipeline = data_preprocessing.create_pipeline()
    vilbert = vilbert_pipeline.create_pipeline()
    visualbert = visualbert_pipeline.create_pipeline()
    clip = clip_pipeline.create_pipeline()
    eval_pipeline = evaluation.create_pipeline()

    return {
        "__default__": data_pipeline + vilbert + visualbert + clip + eval_pipeline,
        "data": data_pipeline,
        "vilbert": data_pipeline + vilbert,
        "visualbert": data_pipeline + visualbert,
        "clip": data_pipeline + clip,
        "evaluation": eval_pipeline,
        "all_models": vilbert + visualbert + clip + eval_pipeline,
    }
