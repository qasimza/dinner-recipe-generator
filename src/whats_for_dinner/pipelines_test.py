from jinja2 import Template

from whats_for_dinner.models import RecipeRecommendation
from whats_for_dinner.pipelines import RECIPE_PROMPT_TEMPLATE


def test_prompt_template_renders_query():
    template = Template(RECIPE_PROMPT_TEMPLATE)
    rendered = template.render(
        query="chicken, garlic, soy sauce",
        documents=[],
    )
    assert "chicken, garlic, soy sauce" in rendered


def test_prompt_template_renders_documents():
    template = Template(RECIPE_PROMPT_TEMPLATE)
    rendered = template.render(
        query="chicken, garlic",
        documents=[
            {
                "meta": {
                    "title": "Quick Chicken Stir-Fry",
                    "instructions": "Heat oil in a wok. Add garlic and chicken.",
                },
                "content": "Quick Chicken Stir-Fry\n\nchicken, garlic, soy sauce",
            },
        ],
    )
    assert "Quick Chicken Stir-Fry" in rendered
    assert "Heat oil in a wok" in rendered
    assert '"title": "string"' in rendered
    assert "Do not include markdown, code fences, or additional keys." in rendered


def test_recipe_recommendation_to_markdown_without_modifications():
    recommendation = RecipeRecommendation(
        title="Quick Chicken Stir-Fry",
        match_reason="Best match for chicken and soy sauce with minimal substitutions.",
        ingredients=["chicken", "garlic", "soy sauce"],
        instructions=[
            "Heat oil in a wok.",
            "Add garlic and cook until fragrant.",
            "Add chicken and stir-fry until cooked through.",
        ],
        modifications=None,
    )

    markdown = recommendation.to_markdown()

    assert markdown.startswith("## Quick Chicken Stir-Fry")
    assert "### Ingredients" in markdown
    assert "- chicken" in markdown
    assert "### Instructions" in markdown
    assert "1. Heat oil in a wok." in markdown
    assert "### Modifications" not in markdown


def test_recipe_recommendation_to_markdown_with_modifications():
    recommendation = RecipeRecommendation(
        title="Tomato Pasta",
        match_reason="Tomatoes are the strongest match in your ingredient list.",
        ingredients=["pasta", "tomatoes", "olive oil"],
        instructions=["Boil pasta.", "Cook tomatoes in olive oil.", "Combine and serve."],
        modifications="Used dried basil instead of fresh basil.",
    )

    markdown = recommendation.to_markdown()

    assert "### Modifications" in markdown
    assert "Used dried basil instead of fresh basil." in markdown
