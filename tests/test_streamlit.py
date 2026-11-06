from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_companion_uses_the_shared_model():
    app = Path(__file__).resolve().parents[1] / "project" / "deployment" / "app.py"
    run = AppTest.from_file(str(app)).run(timeout=30)
    assert not run.exception
    run.number_input(key="campus-sleep_hours").set_value(9.0)
    run.button[0].click().run(timeout=30)
    assert not run.exception
    assert len(run.metric) == 2
    assert run.metric[0].value != run.metric[1].value
