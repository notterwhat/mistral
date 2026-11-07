# Hacker News Tech Briefing (Mistral Workflow)

One-shot workflow `hn-tech-briefing` deployed on Mistral Cloud (managed deployment, hardened/OBO):

1. Reads the Hacker News front page (`hacker-news.firebaseio.com`).
2. An LLM selects strictly tech-industry stories (no personal blogs / non-tech) and writes a German markdown summary plus an English podcast script (~1-2 min).
3. Voxtral TTS (`voxtral-mini-tts-3-0`, voice `gb_jane_neutral`) renders the script to MP3.
4. The MP3 is uploaded to Google Drive; the summary email with the Drive link is sent via Gmail to the user's own inbox.

Run manually (no schedule) via `execute_workflow` on workflow `hn-tech-briefing`, input e.g. `{"to_email": "you@gmail.com", "max_stories": 10}`.

Layout: `apps/worker` is the Python worker, `deploy/docker/Dockerfile.worker` is the managed-deployment image (build context = repository root).
