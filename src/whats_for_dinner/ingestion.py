import logging
from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

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


def create_document_store(connection_string: str) -> PgvectorDocumentStore:
    """Create a PgvectorDocumentStore configured for OpenAI text-embedding-3-small (1536 dims)."""
    return PgvectorDocumentStore(
        connection_string=connection_string,
        table_name="recipes",
        embedding_dimension=1536,
        vector_function="cosine_similarity",
        recreate_table=True,
        search_strategy="hnsw",
    )


def build_indexing_pipeline(
    document_store: PgvectorDocumentStore, api_key: str
) -> Pipeline:
    """Build a Haystack pipeline that embeds documents and writes them to pgvector.

    Args:
        document_store: The PgvectorDocumentStore to write to.
        api_key: OpenAI API key for the embedder.

    Returns:
        A Pipeline wiring OpenAIDocumentEmbedder -> DocumentWriter.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(api_key=api_key, model="text-embedding-3-small"),
    )
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    pipeline.connect("embedder", "writer")
    return pipeline
