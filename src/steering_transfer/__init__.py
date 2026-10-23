"""Readable scientific choices; shared execution and result analysis."""

from .activation_extraction import (
    capture_prompt_activations,
    capture_response_activations,
)
from .comparisons import (
    compare_conditions,
    compare_directions,
    compare_judge_labels,
    compare_persona_conditions,
    compare_utility_scores,
)
from .datasets import (
    load_extraction_prompts,
    load_multiple_choice_dataset,
    load_prompt_dataset,
    load_prompt_pairs,
)
from .direction_processing import (
    compute_contrastive_directions,
    create_random_directions,
    load_direction_set,
    select_direction,
)
from .displays import sample_records, show_response_examples
from .historical_results import (
    load_geometry,
    load_persona_results,
    load_recorded_capture,
    load_recorded_direction,
    load_recorded_judgments,
    load_refusal_results,
)
from .interventions import (
    REMOVAL_LOCATIONS,
    add_direction,
    remove_direction,
    unchanged_model,
)
from .language_models import load_language_model
from .persona_experiments import (
    build_persona_evaluation_prompts,
    build_persona_prompts,
    create_persona_conditions,
    extract_persona_directions,
    select_persona_pairs,
)
from .records import GreedyDecoding, SampledDecoding
from .response_evaluation import (
    evaluate_response_rubrics,
    evaluate_responses,
    load_response_judge,
)
from .response_generation import generate_responses
from .result_storage import find_project_root, load_experiment, read_records
from .utility_evaluation import evaluate_multiple_choice
from .validation import (
    find_outlier_coordinates,
    screen_refusal_directions,
    select_validation_layer,
)

__all__ = [
    "REMOVAL_LOCATIONS",
    "load_recorded_capture",
    "load_recorded_direction",
    "find_project_root",
    "GreedyDecoding",
    "SampledDecoding",
    "add_direction",
    "build_persona_prompts",
    "build_persona_evaluation_prompts",
    "extract_persona_directions",
    "create_persona_conditions",
    "capture_prompt_activations",
    "capture_response_activations",
    "compare_conditions",
    "compare_persona_conditions",
    "compare_directions",
    "compare_judge_labels",
    "compare_utility_scores",
    "compute_contrastive_directions",
    "create_random_directions",
    "evaluate_multiple_choice",
    "evaluate_response_rubrics",
    "evaluate_responses",
    "find_outlier_coordinates",
    "generate_responses",
    "load_recorded_judgments",
    "load_direction_set",
    "load_experiment",
    "load_extraction_prompts",
    "load_geometry",
    "load_language_model",
    "load_multiple_choice_dataset",
    "load_persona_results",
    "load_prompt_dataset",
    "load_prompt_pairs",
    "load_refusal_results",
    "load_response_judge",
    "read_records",
    "remove_direction",
    "sample_records",
    "screen_refusal_directions",
    "select_direction",
    "select_persona_pairs",
    "select_validation_layer",
    "show_response_examples",
    "unchanged_model",
]
