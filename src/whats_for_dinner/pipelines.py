import logging
import json

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from whats_for_dinner.config import Settings, get_settings
from whats_for_dinner.guardrails import validate_recipe_output
from whats_for_dinner.models import RecipeRecommendation

logger = logging.getLogger(__name__)

RECIPE_PROMPT_TEMPLATE = """\
You are a helpful dinner recipe assistant. Based on the ingredients the user has available, \
recommend the best matching recipe from the provided context.

## User's available ingredients
{{ query }}

## Retrieved recipes
{% for doc in documents %}
### {{ doc.meta.title }}
**Ingredients:** {{ doc.content }}
**Instructions:** {{ doc.meta.instructions }}
{% endfor %}

<task>
Pick the single best matching recipe from the retrieved recipes above. \
When choosing, focus on distinctive ingredients (proteins, vegetables, spices) rather than \
common pantry staples like garlic, salt, pepper, and olive oil, which appear in most recipes \
and are not useful for differentiating between them.
</task>

<grounding_rules>
- Prefer recommending recipes from the retrieved context. If none are a reasonable match, you may propose a legitimate recipe based on general cooking knowledge.
- If the user is missing ingredients from the chosen recipe, adapt it by suggesting \
substitutions or modifying the steps to work with only what the user has. Clearly note any \
modifications you made.
- If none of the retrieved recipes are a reasonable match, say so plainly and suggest what the \
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


def recommend_recipe(ingredients: str, pipeline: Pipeline) -> str:
    """Run the RAG pipeline to recommend a recipe based on available ingredients.

    Args:
        ingredients: A text description of the user's available ingredients.
        pipeline: A pre-built RAG pipeline from build_rag_pipeline().

    Returns:
        Markdown-formatted recipe recommendation from GPT-4o.
    """
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
        recommendation = RecipeRecommendation.model_validate(json.loads(raw_reply))
        output_validation = validate_recipe_output(recommendation, settings=get_settings())
        if not output_validation.is_valid:
            logger.warning("Output guardrail blocked response: %s", output_validation.issues)
            return (
                "Sorry, I couldn't provide a safe recipe recommendation for that request. "
                "Please try a different ingredient list."
            )
        logger.info("Structured recommendation: %s", recommendation.model_dump_json())
        return recommendation.to_markdown()
    except Exception:
        logger.warning("Failed to parse structured recommendation", exc_info=True)
        return raw_reply
