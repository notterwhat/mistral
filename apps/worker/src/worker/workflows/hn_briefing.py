import asyncio
import base64
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pydantic

import mistralai.workflows as workflows
from mistralai.workflows import Depends
from mistralai.workflows.client import get_mistral_client
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

gmail_connector = connector("gmail")
drive_connector = connector("google_drive_mcp")

HN_FRONTPAGE_URL = "https://hacker-news.firebaseio.com/v0/frontpage.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
HN_STORY_URL = "https://news.ycombinator.com/item?id={item_id}"

SUMMARY_MODEL = "mistral-medium-latest"
TTS_MODEL = "voxtral-mini-tts-3-0"
TTS_VOICE_ID = "gb_jane_neutral"

FETCH_LIMIT = 30
HTTP_TIMEOUT_SECONDS = 30.0


class HNBriefingInput(pydantic.BaseModel):
    to_email: str = "drchristophmpeters@gmail.com"
    max_stories: int = 10


class Story(pydantic.BaseModel):
    index: int
    item_id: int
    title: str
    url: str | None
    story_url: str
    by: str | None
    score: int | None
    descendants: int | None


class Briefing(pydantic.BaseModel):
    selected: list[dict[str, Any]]
    summary_md: str
    podcast_script: str


class HNBriefingOutput(pydantic.BaseModel):
    stories_fetched: int
    stories_selected: int
    summary_md: str
    audio_file_id: str
    audio_url: str
    audio_bytes: int
    email_subject: str
    sent_message_id: str | None


def _tool_result_as_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for part in content:
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
    return {"raw": str(result)}


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


def build_selection_prompt(stories: list[Story], max_stories: int) -> str:
    listing = "\n".join(
        f"[{s.index}] id={s.item_id} title={s.title!r} url={s.url or s.story_url} "
        f"by={s.by} score={s.score} comments={s.descendants}"
        for s in stories
    )
    return f"""You are filtering the current Hacker News front page for a tech-news briefing.

Rules:
- Keep ONLY stories that are clearly about the technology industry: AI/ML, semiconductors,
  cybersecurity, developer tools and infrastructure, big-tech products and companies,
  tech startups and funding, open-source projects, space technology, hardware, etc.
- Exclude personal blogs, personal opinion pieces, non-tech news (politics, sports,
  science that is not tech-related), lifestyle content, and pure "Ask HN" questions.
- Select at most {max_stories} stories, ordered by importance for a tech audience.
- Write "summary_md" in GERMAN: a short intro line plus one bullet per story
  (headline as a Markdown link to the story URL, then 1-2 sentences of context).
- Write "podcast_script" in ENGLISH: a podcast-style spoken script of about 200-260 words
  (roughly 1-2 minutes when read aloud). Start with a one-line intro, present each selected
  headline with a 1-2 sentence takeaway, and end with a one-line outro. Plain speakable text,
  no Markdown, no URLs read out loud.

Return ONLY a JSON object with this shape:
{{
  "selected": [{{"index": <int>, "title": <str>, "url": <str>, "why": <str>}}],
  "summary_md": <str>,
  "podcast_script": <str>
}}

Front page stories:
{listing}"""


def build_email_body(summary_md: str, audio_url: str, audio_title: str) -> tuple[str, str]:
    plain = (
        f"{summary_md}\n\n"
        f"Podcast-Audio (1-2 Minuten, engl.): {audio_title}\n{audio_url}\n"
    )
    html = (
        f"<div>{summary_md.replace(chr(10), '<br>')}</div>"
        f"<hr><p><b>Podcast-Audio (1-2 Minuten, engl.):</b> "
        f'<a href="{audio_url}">{audio_title}</a></p>'
    )
    return plain, html


def audio_filename(now: datetime) -> str:
    return f"hn-tech-briefing-{now:%Y%m%d-%H%M%S}.mp3"


@workflows.activity(start_to_close_timeout=timedelta(minutes=2))
async def fetch_frontpage_stories(fetch_limit: int) -> list[Story]:
    import httpx

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(HN_FRONTPAGE_URL)
        response.raise_for_status()
        item_ids = response.json()[:fetch_limit]

        async def fetch_item(item_id: int) -> dict[str, Any] | None:
            try:
                item_response = await client.get(HN_ITEM_URL.format(item_id=item_id))
                item_response.raise_for_status()
                return item_response.json()
            except httpx.HTTPError:
                return None

        items = [item for item in await asyncio.gather(*(fetch_item(i) for i in item_ids)) if item]

    stories = []
    for position, item in enumerate(items):
        if item.get("type") != "story" or not item.get("title"):
            continue
        stories.append(
            Story(
                index=len(stories),
                item_id=item["id"],
                title=item["title"],
                url=item.get("url"),
                story_url=HN_STORY_URL.format(item_id=item["id"]),
                by=item.get("by"),
                score=item.get("score"),
                descendants=item.get("descendants"),
            )
        )
    return stories


@workflows.activity(start_to_close_timeout=timedelta(minutes=3))
async def summarize_tech_news(stories: list[Story], max_stories: int) -> Briefing:
    client = get_mistral_client(use_executor_credentials=True)
    response = await client.chat.complete_async(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise tech-news editor. Answer only with the requested JSON object."},
            {"role": "user", "content": build_selection_prompt(stories, max_stories)},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    parsed = _extract_json_object(content)
    return Briefing(
        selected=parsed.get("selected", []),
        summary_md=parsed.get("summary_md", ""),
        podcast_script=parsed.get("podcast_script", ""),
    )


@workflows.activity(start_to_close_timeout=timedelta(minutes=3))
async def generate_podcast_audio(podcast_script: str) -> str:
    client = get_mistral_client(use_executor_credentials=True)
    response = await client.audio.speech.complete_async(
        model=TTS_MODEL,
        input=podcast_script,
        voice_id=TTS_VOICE_ID,
        response_format="mp3",
    )
    return response.audio_data


@workflows.activity(start_to_close_timeout=timedelta(minutes=2))
async def upload_audio_to_drive(
    audio_base64: str,
    filename: str,
    drive: ToolCallClient = Depends(drive_connector),
) -> dict[str, Any]:
    result = _tool_result_as_dict(
        await drive.call_tool(
            "create_file",
            {
                "title": filename,
                "base64_content": audio_base64,
                "content_mime_type": "audio/mpeg",
                "disable_conversion_to_google_type": True,
            },
        )
    )
    file_id = result.get("id")
    if not file_id:
        raise ValueError(f"Drive create_file returned no file id: {str(result)[:300]}")
    view_url = result.get("view_url") or f"https://drive.google.com/file/d/{file_id}/view"
    return {
        "id": file_id,
        "url": view_url,
        "title": result.get("title") or filename,
        "size": int(result.get("file_size") or 0),
    }


@workflows.activity(start_to_close_timeout=timedelta(minutes=2))
async def send_briefing_email(
    to_email: str,
    subject: str,
    summary_md: str,
    audio_url: str,
    audio_title: str,
    gmail: ToolCallClient = Depends(gmail_connector),
) -> dict[str, Any]:
    body, html_body = build_email_body(summary_md, audio_url, audio_title)
    result = _tool_result_as_dict(
        await gmail.call_tool(
            "send_email",
            {"to": [to_email], "subject": subject, "body": body, "htmlBody": html_body},
        )
    )
    return {"id": result.get("id"), "threadId": result.get("threadId")}


@workflows.workflow.define(
    name="hn-tech-briefing",
    workflow_display_name="Hacker News Tech Briefing",
    workflow_description=(
        "Liest die Hacker-News-Startseite, fasst die neuesten Tech-Nachrichten zusammen, "
        "erzeugt eine englische Podcast-Audiodatei (1-2 Min.) und mailt beides per Gmail. "
        "Manuell ausloesbar (keine Planung)."
    ),
    on_behalf_of=True,
)
@uses_connectors(gmail_connector, drive_connector)
class HNTechBriefingWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: HNBriefingInput) -> HNBriefingOutput:
        stories = await fetch_frontpage_stories(FETCH_LIMIT)
        briefing = await summarize_tech_news(stories, input.max_stories)
        audio_base64 = await generate_podcast_audio(briefing.podcast_script)
        audio_bytes = len(base64.b64decode(audio_base64))
        filename = audio_filename(datetime.now(timezone.utc))
        drive_file = await upload_audio_to_drive(audio_base64, filename)
        subject = f"Hacker News Tech Briefing - {datetime.now(timezone.utc):%Y-%m-%d}"
        sent = await send_briefing_email(
            input.to_email, subject, briefing.summary_md, drive_file["url"], filename
        )
        return HNBriefingOutput(
            stories_fetched=len(stories),
            stories_selected=len(briefing.selected),
            summary_md=briefing.summary_md,
            audio_file_id=drive_file["id"],
            audio_url=drive_file["url"],
            audio_bytes=audio_bytes,
            email_subject=subject,
            sent_message_id=sent.get("id"),
        )
