from jinja2 import Template

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
