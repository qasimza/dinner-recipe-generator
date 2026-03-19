import logging
from pathlib import Path

from haystack import Document

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


def load_recipes(recipes_dir: Path) -> list[Document]:
    """Load all recipe files and convert them to Haystack Documents.

    Args:
        recipes_dir: Directory containing recipe .txt files.

    Returns:
        List of Documents with content = title + ingredients (for embedding)
        and metadata containing title, instructions, and source_file.
    """
    documents: list[Document] = []
    for file_path in sorted(recipes_dir.glob("*.txt")):
        recipe = parse_recipe(file_path)
        doc = Document(
            content=f"{recipe['title']}\n\n{recipe['ingredients']}",
            meta={
                "title": recipe["title"],
                "instructions": recipe["instructions"],
                "source_file": recipe["source_file"],
            },
        )
        documents.append(doc)

    logger.info("Loaded %d recipes from %s", len(documents), recipes_dir)
    return documents
