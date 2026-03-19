import logging
import json
import re
from datetime import datetime, UTC
from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from whats_for_dinner.config import Settings, get_settings
from whats_for_dinner.guardrails import validate_recipe_output
from whats_for_dinner.ingestion import build_indexing_pipeline
from whats_for_dinner.models import RecipeRecommendation

logger = logging.getLogger(__name__)

RECIPE_PROMPT_TEMPLATE = """
You are a helpful dinner recipe assistant. Based on the ingredients the user has available, 
create a practical recipe by using the provided context as reference examples.

## User's available ingredients
{{ query }}

## Reference recipes
{% for doc in documents %}
### {{ doc.meta.title }}
**Ingredients:** {{ doc.content }}
**Instructions:** {{ doc.meta.instructions }}
{% endfor %}

<task>
Use the retrieved recipes above as EXAMPLES and inspiration, not as options to select from.
Synthesize one recipe tailored to the user's ingredients by combining patterns, techniques, or flavor ideas from relevant references.
Focus on distinctive ingredients (proteins, vegetables, spices) rather than common pantry staples like garlic, salt, pepper, and olive oil, which appear in most recipes and are not useful for matching.
</task>

<grounding_rules>
- Ground your recipe in the retrieved context when possible, but do not copy a single retrieved recipe verbatim.
- If useful ingredients are missing from reference recipes, adapt with substitutions or modified steps so the recipe works with what the user has. Clearly note any modifications you made.
- If none of the retrieved recipes are a reasonable match, say so plainly and suggest what the 
user could make with their ingredients instead, based on general cooking knowledge.
- Do not ask clarifying questions. If the query is ambiguous, state your interpretation and proceed.
</grounding_rules>

<output_format>
Return ONLY valid JSON with this exact schema:
{
  "title": "string",
  "match_reason": "string",
  "ingredients": ["string", "..."],
  "instructions": ["string", "..."],
  "modifications": "string or null"
}

Do not include markdown, code fences, or additional keys.
</output_format>
"""


def build_rag_pipeline(
    document_store: PgvectorDocumentStore, settings: Settings
) -> Pipeline:
    """Build a Haystack RAG pipeline for recipe recommendation.

    Args:
        document_store: PgvectorDocumentStore with embedded recipe documents.
        settings: Application settings (API key, model names, top_k).

    Returns:
        A Pipeline wiring: OpenAITextEmbedder -> PgvectorEmbeddingRetriever -> PromptBuilder -> OpenAIGenerator.
    """
    api_key = Secret.from_token(settings.openai_api_key)

    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        OpenAITextEmbedder(api_key=api_key, model=settings.embedding_model),
    )
    pipeline.add_component(
        "retriever",
        PgvectorEmbeddingRetriever(document_store=document_store, top_k=settings.top_k),
    )
    pipeline.add_component(
        "prompt_builder",
        PromptBuilder(
            template=RECIPE_PROMPT_TEMPLATE,
            required_variables=["query", "documents"],
        ),
    )
    pipeline.add_component(
        "llm",
        OpenAIGenerator(
            api_key=api_key,
            model=settings.llm_model,
            generation_kwargs={"response_format": {"type": "json_object"}},
        ),
    )

    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    return pipeline


def _get_top_retrieval_score(result: dict) -> float | None:
    documents = result.get("retriever", {}).get("documents", [])
    if not documents:
        return None
    top = documents[0]
    score = getattr(top, "score", None)
    if isinstance(score, (float, int)):
        return float(score)
    return None


def _extract_items(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    parts = re.split(r"[\n,;]+", normalized)
    items: list[str] = []
    for part in parts:
        item = part.strip()
        if not item:
            continue
        item = re.sub(r"^[-*]\s*", "", item)
        item = re.sub(r"^\d+\.\s*", "", item)
        if item:
            items.append(item)
    return items


def _normalize_ingredient(item: str) -> str:
    normalized = item.lower()
    normalized = re.sub(r"\(.*?\)", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"^\d+(\.\d+)?\s*[a-z]+\s+", "", normalized)
    return normalized


def _extract_document_ingredients(document: Document) -> list[str]:
    content = document.content or ""
    title = str(document.meta.get("title", "")).strip()
    ingredients_block = content
    if title and content.startswith(title):
        ingredients_block = content[len(title) :].strip()
    return _extract_items(ingredients_block)


def _extract_document_instructions(document: Document) -> list[str]:
    instructions = str(document.meta.get("instructions", "")).strip()
    return _extract_items(instructions)


def _all_recipe_ingredients_available(ingredients: str, document: Document) -> bool:
    user_items = {
        _normalize_ingredient(item)
        for item in _extract_items(ingredients)
        if _normalize_ingredient(item)
    }
    recipe_items = {
        _normalize_ingredient(item)
        for item in _extract_document_ingredients(document)
        if _normalize_ingredient(item)
    }
    if not user_items or not recipe_items:
        return False
    return recipe_items.issubset(user_items)


def _recommendation_from_document(document: Document) -> RecipeRecommendation:
    title = str(document.meta.get("title", "")).strip() or "Recipe"
    ingredients = _extract_document_ingredients(document)
    instructions = _extract_document_instructions(document)
    return RecipeRecommendation(
        title=title,
        match_reason="Exact cache match: you already have all required ingredients for this recipe.",
        ingredients=ingredients,
        instructions=instructions,
        modifications=None,
    )


def _retrieve_documents(ingredients: str, pipeline: Pipeline) -> list[Document]:
    embedder = pipeline.get_component("embedder")
    retriever = pipeline.get_component("retriever")
    embedding_result = embedder.run(text=ingredients)
    retrieval_result = retriever.run(query_embedding=embedding_result["embedding"])
    return retrieval_result.get("documents", [])


def _persist_generated_recipe(
    recommendation: RecipeRecommendation,
    ingredients: str,
    settings: Settings,
    document_store: PgvectorDocumentStore,
) -> None:
    output_dir = Path(settings.generated_recipes_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / settings.generated_recipes_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]+", "-", recommendation.title.lower()).strip("-") or "recipe"
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{slug}.txt"
    file_path = output_dir / filename

    file_text = (
        f"{recommendation.title}\n\n"
        f"Ingredients:\n" + "\n".join(f"- {item}" for item in recommendation.ingredients) + "\n\n"
        "Instructions:\n"
        + "\n".join(
            f"{idx}. {step}"
            for idx, step in enumerate(recommendation.instructions, start=1)
        )
        + "\n\n"
        f"Generated From Ingredients:\n{ingredients}\n"
    )
    file_path.write_text(file_text)

    generated_doc = Document(
        content=f"{recommendation.title}\n\n{', '.join(recommendation.ingredients)}",
        meta={
            "title": recommendation.title,
            "instructions": "\n".join(recommendation.instructions),
            "source_file": filename,
            "generated": True,
            "match_reason": recommendation.match_reason,
            "modifications": recommendation.modifications or "",
        },
    )
    indexing_pipeline = build_indexing_pipeline(document_store, settings.openai_api_key)
    indexing_pipeline.run({"embedder": {"documents": [generated_doc]}})

    logger.info("Persisted generated recipe to %s and pgvector", file_path)


def recommend_recipe(
    ingredients: str,
    pipeline: Pipeline,
    settings: Settings | None = None,
    document_store: PgvectorDocumentStore | None = None,
) -> str:
    """Run the RAG pipeline to recommend a recipe based on available ingredients.

    Args:
        ingredients: A text description of the user's available ingredients.
        pipeline: A pre-built RAG pipeline from build_rag_pipeline().
        settings: Optional settings object; falls back to get_settings().
        document_store: Optional document store for persisting cache misses.

    Returns:
        Markdown-formatted recipe recommendation from GPT-4o.
    """
    retrieved_documents = _retrieve_documents(ingredients, pipeline)
    top_document = retrieved_documents[0] if retrieved_documents else None
    if top_document is not None and _all_recipe_ingredients_available(ingredients, top_document):
        logger.info("Exact cache hit by ingredient coverage. Returning original cached recipe.")
        return _recommendation_from_document(top_document).to_markdown()

    result = pipeline.run(
        {
            "embedder": {"text": ingredients},
            "prompt_builder": {"query": ingredients},
        }
    )

    replies = result["llm"]["replies"]
    if not replies:
        return "Sorry, I couldn't generate a recipe recommendation. Please try again."

    raw_reply = replies[0] if isinstance(replies[0], str) else str(replies[0])
    try:
        resolved_settings = settings or get_settings()
        recommendation = RecipeRecommendation.model_validate(json.loads(raw_reply))
        output_validation = validate_recipe_output(recommendation, settings=resolved_settings)
        if not output_validation.is_valid:
            logger.warning("Output guardrail blocked response: %s", output_validation.issues)
            return (
                "Sorry, I couldn't provide a safe recipe recommendation for that request. "
                "Please try a different ingredient list."
            )
        logger.info("Structured recommendation: %s", recommendation.model_dump_json())

        top_score = _get_top_retrieval_score(result)
        similar_match = (
            top_score is not None
            and top_score >= resolved_settings.rag_cache_similarity_threshold
        )
        if document_store is not None:
            logger.info(
                "Persisting generated recipe (%s, score=%s).",
                "similar match" if similar_match else "no good match",
                "none" if top_score is None else f"{top_score:.4f}",
            )
            _persist_generated_recipe(
                recommendation=recommendation,
                ingredients=ingredients,
                settings=resolved_settings,
                document_store=document_store,
            )
        else:
            logger.info(
                "Generated recipe not persisted because no document_store was provided (score=%s).",
                "none" if top_score is None else f"{top_score:.4f}",
            )

        return recommendation.to_markdown()
    except Exception:
        logger.warning("Failed to parse structured recommendation", exc_info=True)
        return raw_reply
