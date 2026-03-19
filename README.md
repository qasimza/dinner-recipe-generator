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

## Assumptions

- The application is scoped to **dinner recipes only** — we're not catering to snacks, breakfast, or lunch.
- We assume that the user has all utencils and appliances needed for the recipes, i.e. a fully equipped kitchen. 
- We assume the user is adept at cooking - we do not need to adjust the the difficulty level of a recipe according to a user's skill level. 