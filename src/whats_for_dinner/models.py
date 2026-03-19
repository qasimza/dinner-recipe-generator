from pydantic import BaseModel


class RecipeRequest(BaseModel):
    ingredients: str


class RecipeResponse(BaseModel):
    recipe: str
