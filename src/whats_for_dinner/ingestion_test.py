from pathlib import Path

from whats_for_dinner.ingestion import load_recipes, parse_recipe

RECIPES_DIR = Path(__file__).resolve().parents[2] / "data" / "recipes"


def test_parse_recipe():
    result = parse_recipe(RECIPES_DIR / "01.txt")
    assert result["title"] == "Quick Chicken Stir-Fry"
    assert "2 chicken breasts, diced" in result["ingredients"]
    assert "Heat oil in a wok" in result["instructions"]
    assert result["source_file"] == "01.txt"


def test_parse_recipe_keys():
    result = parse_recipe(RECIPES_DIR / "01.txt")
    assert set(result.keys()) == {"title", "ingredients", "instructions", "source_file"}


def test_load_recipes():
    docs = load_recipes(RECIPES_DIR)
    assert len(docs) == 20
    for doc in docs:
        assert doc.content is not None
        assert "title" in doc.meta
        assert "instructions" in doc.meta
        assert "source_file" in doc.meta


def test_load_recipes_content_is_title_and_ingredients():
    docs = load_recipes(RECIPES_DIR)
    doc = docs[0]
    content = doc.content or ""
    assert doc.meta["title"] in content
    assert "Instructions" not in content
