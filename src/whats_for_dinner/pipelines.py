import logging

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from whats_for_dinner.config import Settings

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
- ONLY recommend recipes from the retrieved context above. Never invent or fabricate a recipe.
- If the user is missing ingredients from the chosen recipe, adapt it by suggesting \
substitutions or modifying the steps to work with only what the user has. Clearly note any \
modifications you made.
- If none of the retrieved recipes are a reasonable match, say so plainly and suggest what the \
user could make with their ingredients instead, based on general cooking knowledge.
- Do not ask clarifying questions. If the query is ambiguous, state your interpretation and proceed.
</grounding_rules>

<output_format>
Respond in Markdown with the following structure:
1. Recipe title as a level-2 heading
2. A short sentence on why this recipe is a good match
3. Ingredients as a bullet list (note any substitutions)
4. Instructions as a numbered list
5. If modifications were made, a brief "Modifications" section explaining what changed and why

Keep the response concise and actionable. Do not repeat the user's query or add unnecessary preamble.
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
        PromptBuilder(template=RECIPE_PROMPT_TEMPLATE),
    )
    pipeline.add_component(
        "llm",
        OpenAIGenerator(api_key=api_key, model=settings.llm_model),
    )

    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder", "llm")

    return pipeline
