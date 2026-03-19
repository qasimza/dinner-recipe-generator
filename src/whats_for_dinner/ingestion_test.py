from pathlib import Path

from whats_for_dinner.ingestion import parse_recipe

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
