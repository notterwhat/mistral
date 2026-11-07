import json
from datetime import datetime, timezone

from worker.workflows import hn_briefing as mod
from worker.workflows.hn_briefing import (
    Briefing,
    HNBriefingInput,
    Story,
    _extract_json_object,
    audio_filename,
    build_email_body,
    build_selection_prompt,
)


def _story(index: int, title: str = "Some AI startup raises $100M") -> Story:
    return Story(
        index=index,
        item_id=1000 + index,
        title=title,
        url="https://example.com/post",
        story_url=f"https://news.ycombinator.com/item?id={1000 + index}",
        by="pg",
        score=200,
        descendants=50,
    )


def test_extract_json_object_plain():
    assert _extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_in_code_fence():
    text = '```json\n{"a": 1, "b": [2, 3]}\n```'
    assert _extract_json_object(text) == {"a": 1, "b": [2, 3]}


def test_extract_json_object_with_surrounding_prose():
    text = 'Here you go:\n{"selected": []}\nHope that helps.'
    assert _extract_json_object(text) == {"selected": []}


def test_extract_json_object_raises_without_json():
    try:
        _extract_json_object("no json at all")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_build_selection_prompt_lists_stories_and_rules():
    prompt = build_selection_prompt([_story(0), _story(1, "New GPU announced")], max_stories=5)
    assert "[0]" in prompt and "[1]" in prompt
    assert "New GPU announced" in prompt
    assert "personal blogs" in prompt
    assert "podcast_script" in prompt
    assert "5" in prompt


def test_build_email_body_contains_summary_and_audio_link():
    plain, html = build_email_body("# Summary\n- item", "https://drive.google.com/file/d/abc/view", "file.mp3")
    assert "# Summary" in plain
    assert "https://drive.google.com/file/d/abc/view" in plain
    assert "file.mp3" in plain
    assert 'href="https://drive.google.com/file/d/abc/view"' in html


def test_audio_filename_format():
    name = audio_filename(datetime(2026, 11, 7, 22, 5, 9, tzinfo=timezone.utc))
    assert name == "hn-tech-briefing-20261107-220509.mp3"


def test_input_defaults():
    inp = HNBriefingInput()
    assert inp.to_email == "drchristophmpeters@gmail.com"
    assert inp.max_stories == 10


def test_briefing_model_roundtrip():
    b = Briefing(selected=[{"index": 0}], summary_md="s", podcast_script="p")
    assert json.loads(b.model_dump_json())["podcast_script"] == "p"


def test_workflow_is_discoverable():
    import mistralai.workflows as workflows

    classes = workflows.discover_all_workflows_in_package("worker.workflows")
    names = [c.__name__ for c in classes]
    assert "HNTechBriefingWorkflow" in names
