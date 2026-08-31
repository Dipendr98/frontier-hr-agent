"""
Package bootstrap.

`.env` is loaded here, at package import, rather than in each entry point.
Previously only app.py called load_dotenv(), so a key sitting in .env was
invisible to the CLI agents and to pytest — which meant the LLM tests silently
skipped on a machine that had a perfectly good provider configured, and
`python advanced/orchestration/llm_agent.py EMP4` refused to run unless you
also remembered to `export`. A test that skips when it should run is worse than
one that fails.

Environment variables already set always win; python-dotenv does not override
them. python-dotenv itself is optional, so the no-key path has no hard
dependency on it.
"""
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass
