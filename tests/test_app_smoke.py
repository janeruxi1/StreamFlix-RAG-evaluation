"""The demo is the part of this project most people will actually open.

Nothing else in the suite runs it, so a change that breaks an import or
a record key would otherwise reach the public page unnoticed. These tests
run every mode headlessly, with no credential, the way the hosted demo
runs.

They are hermetic on purpose: the app would make paid judge calls if it
found a key, and a developer's local .env has one.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
MODES = ["Measured results", "Ask anything", "Golden set",
         "Is the judge trustworthy?"]


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("src.llm.provider._load_dotenv_if_present", lambda: None)


def _run(mode: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    if mode != MODES[0]:
        at.sidebar.radio[0].set_value(mode).run()
        assert not at.exception, at.exception
    return at


def test_the_replay_is_the_landing_page_and_offers_every_mode():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    radio = at.sidebar.radio[0]
    assert list(radio.options) == MODES
    assert radio.value == "Measured results"


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_renders_without_a_key(mode):
    at = _run(mode)
    assert at.title, f"{mode} rendered no title"


def test_replay_headline_matches_the_measured_record():
    """The four tiles are the memo's headline. They are computed from the
    per-question rows, so this pins the page to the measured run."""
    at = _run("Measured results")
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["cited: unanswerable questions refused"] == "25 of 25"
    assert tiles["naive: unanswerable questions refused"] == "9 of 25"
    assert tiles["cited: answerable questions answered"] == "76 of 95"
    assert tiles["cited: answers with an unsupported claim"] == "3 of 76"


def test_every_replay_view_has_questions_and_renders():
    at = _run("Measured results")
    show = at.selectbox[0]
    assert show.label == "Show"
    for view in list(show.options):
        at.selectbox[0].set_value(view).run()
        assert not at.exception, (view, at.exception)
        assert at.selectbox[1].options, f"view {view!r} is empty"


def test_the_live_modes_say_they_are_not_the_model():
    """With no key the live modes use the non-LLM baseline. The page has
    to say so, or a visitor reads baseline output as the model's."""
    at = _run("Ask anything")
    text = " ".join(c.value for c in at.caption)
    assert "non-LLM baseline" in text
