from agent.pipeline import run_pipeline


def test_pipeline_mock_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_pipeline(
        date="2026-02-14",
        tz="America/Chicago",
        watchlist=["AAPL", "MSFT"],
        out_dir=tmp_path,
    )

    assert result.research_path.exists()
    assert result.brief_path.exists()
