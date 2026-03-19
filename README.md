# What's for Dinner?

Recipe recommendation API powered by RAG (Haystack 2.x + pgvector + GPT-4o).

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker

### 1. Install dependencies

```bash
uv sync
```

### 2. Start Postgres

```bash
docker-compose up -d
```

### 3. Extract recipe data

```bash
unzip data.zip
```

### 4. Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://pipeline:pipeline-pass@localhost:5432/challenge
```

### 5. Ingest recipes

```bash
uv run python -m whats_for_dinner.ingestion
```

### 6. Run the API

```bash
uv run uvicorn whats_for_dinner.main:app --reload
```

### 7. Access the Fast API UI for the App
```bash
http://127.0.0.1:8000/docs
```

## Assumptions

- The application is scoped to **dinner recipes only** — we're not catering to snacks, breakfast, or lunch.
- We assume that the user has all utencils and appliances needed for the recipes, i.e. a fully equipped kitchen. 
- We assume the user is adept at cooking - we do not need to adjust the the difficulty level of a recipe according to a user's skill level. 

## Initial Findings and Design Decisions

### Structure
- All recipes follow a consistent format.
- No malformed or empty files.

### Images
- Photos contain raw ingredients, seasoning and utensils. 
- ExtractFoodItemsFromImage output is useful, i.e. we can supply ingredients extracted from the image to the recipe generator.

### Embedding strategy
- **Ingredients + Title** The retrieval query is a list of ingredients from a food photo, so embedding ingredients gives the best query-document alignment. Instructions add noise (cooking steps don't help match *what* to cook). Recipes are short (~90 words max), so the discriminating signal is in specific ingredient combinations.

### Metadata
- `title` — display in results
- `source_file` — traceability to original `.txt` file
- `instructions` — returned after retrieval, not embedded

### Retrieval
- Moderate overlap observed between recipes.
- Garlic, salt and pepper to taste, and olive oil are the most commonly occurring/overlapping ingredients. 
- Randomply sampling `top_k=3` seems reasonable for variety. 

## Notes on my approach
- Follow Test Driven Development (TDD), i.e. write unit tests before the actual code. Following a strict "Red-Green-Refactor" cycle. Writing a failing test, making it pass with minimal code, and then refactoring.
  

