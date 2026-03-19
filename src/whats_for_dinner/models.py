from pydantic import BaseModel


class RecipeRequest(BaseModel):
    ingredients: str


class RecipeResponse(BaseModel):
    recipe: str


class RecipeRecommendation(BaseModel):
    title: str
    match_reason: str
    ingredients: list[str]
    instructions: list[str]
    modifications: str | None = None

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"## {self.title}",
            "",
            self.match_reason,
            "",
            "### Ingredients",
        ]
        lines.extend([f"- {item}" for item in self.ingredients])
        lines.append("")
        lines.append("### Instructions")
        lines.extend([f"{idx}. {step}" for idx, step in enumerate(self.instructions, start=1)])

        if self.modifications:
            lines.extend(["", "### Modifications", self.modifications])

        return "\n".join(lines).strip()
