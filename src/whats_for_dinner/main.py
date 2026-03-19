import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException
from haystack import Pipeline

from whats_for_dinner.config import get_settings
from whats_for_dinner.guardrails import validate_input_ingredients
from whats_for_dinner.ingestion import create_document_store
from whats_for_dinner.models import RecipeResponse
from whats_for_dinner.pipelines import build_rag_pipeline, recommend_recipe

logger = logging.getLogger(__name__)

rag_pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global rag_pipeline
    try:
        settings = get_settings()
        document_store = create_document_store(
            settings.database_url, recreate_table=False
        )
        rag_pipeline = build_rag_pipeline(document_store, settings)
        logger.info("RAG pipeline initialized")
    except Exception:
        logger.warning("Could not initialize RAG pipeline", exc_info=True)
    yield


app = FastAPI(title="What's for Dinner?", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/recommend_recipe")
async def recommend_recipe_endpoint(
    ingredients: str = Form(...),
) -> RecipeResponse:
    """Recommend a dinner recipe based on available ingredients."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    try:
        input_validation = validate_input_ingredients(ingredients, settings=get_settings())
    except Exception:
        logger.warning("Input guardrail validation failed", exc_info=True)
        raise HTTPException(status_code=503, detail="Input guardrail unavailable")

    if not input_validation.is_food_only:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Ingredients must be edible food items only.",
                "non_food_items": input_validation.non_food_items,
                "reason": input_validation.reason,
            },
        )

    result = recommend_recipe(ingredients, rag_pipeline)
    return RecipeResponse(recipe=result)
