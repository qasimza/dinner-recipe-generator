from fastapi import FastAPI

app = FastAPI(title="What's for Dinner?")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
