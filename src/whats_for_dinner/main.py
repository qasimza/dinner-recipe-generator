import logging
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from haystack import Pipeline
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

from whats_for_dinner.config import get_settings
from whats_for_dinner.custom_components import ExtractFoodItemsFromImage
from whats_for_dinner.guardrails import validate_input_ingredients
from whats_for_dinner.ingestion import create_document_store
from whats_for_dinner.models import RecipeResponse
from whats_for_dinner.pipelines import build_rag_pipeline, recommend_recipe

logger = logging.getLogger(__name__)

rag_pipeline: Pipeline | None = None
rag_document_store: PgvectorDocumentStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    global rag_pipeline, rag_document_store
    try:
        settings = get_settings()
        rag_document_store = create_document_store(
            settings.database_url, recreate_table=False
        )
        rag_pipeline = build_rag_pipeline(rag_document_store, settings)
        logger.info("RAG pipeline initialized")
    except Exception:
        logger.warning("Could not initialize RAG pipeline", exc_info=True)
    yield


app = FastAPI(title="What's for Dinner?", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


async def extract_ingredients_from_upload(image_file: UploadFile, api_key: str) -> str:
    """Extract food ingredients from an uploaded image using the helper component."""
    suffix = Path(image_file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        contents = await image_file.read()
        temp_file.write(contents)
        temp_file.flush()
        extractor = ExtractFoodItemsFromImage()
        result = extractor.run(temp_file.name, api_key=api_key)
    return str(result.get("answer", "")).strip()


@app.post("/recommend_recipe")
async def recommend_recipe_endpoint(
    ingredients: str = Form(""),
    image_file: UploadFile | None = File(None),
) -> RecipeResponse:
    """Recommend a dinner recipe based on text ingredients and/or an ingredient photo."""
    settings = get_settings()
    extracted_ingredients = ""
    if image_file is not None:
        try:
            extracted_ingredients = await extract_ingredients_from_upload(
                image_file, settings.openai_api_key
            )
        except Exception:
            logger.warning("Image ingredient extraction failed", exc_info=True)
            raise HTTPException(status_code=422, detail="Could not process the uploaded image.")

    combined_ingredients = ", ".join(
        part.strip() for part in [ingredients, extracted_ingredients] if part and part.strip()
    )
    if not combined_ingredients:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one ingredient or an image containing ingredients.",
        )

    if rag_pipeline is None or rag_document_store is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        input_validation = validate_input_ingredients(
            combined_ingredients, settings=settings
        )
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

    result = recommend_recipe(
        combined_ingredients,
        rag_pipeline,
        settings=settings,
        document_store=rag_document_store,
    )
    return RecipeResponse(recipe=result)
