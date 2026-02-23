from agent.schema import ResearchPack


def test_schema_minimal():
    payload = {
        "as_of": "2026-02-14T07:00:00-06:00",
        "earnings": [],
        "events": [],
        "headlines": [],
        "weekly_context": {"themes": [], "market_moves": []},
    }
    ResearchPack.model_validate(payload)
