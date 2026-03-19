import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_recipe(file_path: Path) -> dict[str, str]:
    """Parse a recipe text file into its component parts.

    Args:
        file_path: Path to a recipe .txt file with title, ingredients, and instructions sections.

    Returns:
        Dict with keys: title, ingredients, instructions, source_file.
    """
    text = file_path.read_text()
    sections = text.split("Instructions:")

    header = sections[0]
    instructions = sections[1].strip() if len(sections) > 1 else ""

    header_parts = header.split("Ingredients:")
    title = header_parts[0].strip()
    ingredients = header_parts[1].strip() if len(header_parts) > 1 else ""

    return {
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions,
        "source_file": file_path.name,
    }
