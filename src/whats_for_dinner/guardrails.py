import json

from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from whats_for_dinner.config import Settings
from whats_for_dinner.models import (
    InputValidationResult,
    OutputValidationResult,
    RecipeRecommendation,
)


def _parse_structured_reply(raw_reply: str, model_class):
    return model_class.model_validate(json.loads(raw_reply))


def validate_input_ingredients(
    ingredients: str, settings: Settings
) -> InputValidationResult:
    """Validate that all user-provided ingredients are food items."""
    prompt = f"""You are an ingredient safety validator.
Determine whether the provided items are all edible food ingredients.
Return ONLY JSON matching this schema:
{{
  "is_food_only": boolean,
  "non_food_items": ["string", "..."],
  "reason": "string"
}}

Input ingredients:
{ingredients}
"""
    llm = OpenAIGenerator(
        api_key=Secret.from_token(settings.openai_api_key),
        model=settings.llm_model,
        generation_kwargs={"response_format": {"type": "json_object"}},
    )
    result = llm.run(prompt=prompt)
    replies = result["replies"]
    raw_reply = replies[0] if isinstance(replies[0], str) else str(replies[0])
    return _parse_structured_reply(raw_reply, InputValidationResult)


def validate_recipe_output(
    recommendation: RecipeRecommendation, settings: Settings
) -> OutputValidationResult:
    """Validate that generated recipe output is safe and food-related."""
    prompt = f"""You are an output safety validator for cooking recipes.
Return ONLY JSON matching this schema:
{{
  "is_valid": boolean,
  "issues": ["string", "..."]
}}

The recipe must be strictly about edible food preparation and must not include
instructions involving non-food items, dangerous chemicals, or harmful actions.

Recipe JSON:
{recommendation.model_dump_json()}
"""
    llm = OpenAIGenerator(
        api_key=Secret.from_token(settings.openai_api_key),
        model=settings.llm_model,
        generation_kwargs={"response_format": {"type": "json_object"}},
    )
    result = llm.run(prompt=prompt)
    replies = result["replies"]
    raw_reply = replies[0] if isinstance(replies[0], str) else str(replies[0])
    return _parse_structured_reply(raw_reply, OutputValidationResult)
