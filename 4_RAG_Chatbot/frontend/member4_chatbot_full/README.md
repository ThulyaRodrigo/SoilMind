# Member 4: chat page ("Ask SoilMind" / "Knowledge Assistant")

## What it does
The farmer and officer chat screen from the prototype, wired to Member 4's API (`/api/m4`). Since Phase 6b a
farmer can **speak a question in Sinhala, check and correct the transcript, and hear the answer read aloud**.
Every answer, transcript and voice comes from models that run locally on our server (no outside AI service):
- copied DOA/RRDI sentences with citations
- or a bilingual template filled from the plot's live data
- or a polite refusal forwarded to an agriculture officer

## How it works
- `ChatPage.tsx`:
  - Sends `POST /api/m4/chat` with a per-tab `session_id` (sessionStorage, so follow-ups work), the persona and the
    UI language.
  - Renders each reply as a bubble: the answer lines (tables as small tables), the other-language note,
    live-value chips, and citation chips linking to doa.gov.lk.
  - Refusals and escalations get a separate amber style with the escalation reference.
- **Answer display:**
  - the best 2-3 sentences / table rows (picked by our sentence reranker)
  - the key numbers highlighted (`highlights` from the API)
  - a "Show full passage" expander with the whole cited passage, only the chosen lines highlighted, and the citation
- **Suggested questions:** dictionary strings in the current language. They contain no numbers (CLAUDE.md rule 2).
  A chip disappears once used.
- **Side panel:**
  - **Officers:** "Retrieved context" (top passages with score bars) and the guardrail note, with the threshold
    read from `GET /api/m4/model-info`. "Answer details" shows intent, confidence, answer path, best document
    score and time, and Sinhala answers are marked as not yet checked.
  - **Both personas:** "What your sensors say" / "Live sensor state used", from `GET /api/m3/kpis` (field means,
    reading time in Asia/Colombo), plus the live values the last answer used.
- **No hardcoded display numbers.** Every value, score, threshold and time comes from the API.
- `chat.css`: the prototype's SCREEN 4 styles, plus escalation, table, side-panel and voice styles.

### Voice (Phase 6b, `voice.tsx`)
- **Recording** (`useRecorder`): `MediaRecorder` (Opus in webm, or ogg) with echo cancellation and noise
  suppression. While recording, the button turns red and pulses, a **live waveform** (Web Audio `AnalyserNode`)
  and a `seconds / 30 s` timer are shown; it stops by itself at 30 s (one Whisper window).
- **Transcript:** the recording is sent to `POST /api/m4/stt` (multipart, with the chat language). The text is put
  **in the input box, not sent**: the farmer reads it, corrects any word, then presses send. Empty or failed
  transcriptions show a message instead.
- **Speaker button on every answer:** `POST /api/m4/tts` with the answer text returns a WAV (fetched once per
  answer and kept as a blob URL). One player for the page: starting an answer stops the one playing; the button
  shows *Listen / Preparing voice… / Stop*. A Sinhala answer that quotes an English passage is read in both voices.
- **Farmer mode is voice-first:** a large voice bar above the input (68 px mic, big labels, waveform, timer) and a
  **"Read answers aloud"** switch (auto-play, remembered in localStorage; off by default). Speaker buttons are
  larger with a text label.
- **Officer mode:** the prototype's compact mic inside the composer (the waveform replaces the input while
  recording), and a "Speech model" card with the model, speech length, latency and any farming-word corrections
  the term dictionary made.

## Data sources
`/api/m4/chat`, `/api/m4/model-info`, `/api/m4/stt`, `/api/m4/tts`, `/api/m3/kpis` (shapes in `frontend/shared/types.ts`, mirrored from
`backend/shared/schemas.py`).

## How to run and test
```bash
cd backend && uvicorn main:app --port 8000      # loads the chatbot models (~20 s)
cd backend/member1-iot && python -m member1_iot.feed.replayer   # fresh sensor readings for "my field" answers
cd frontend && npm run dev                       # http://localhost:5173/chat
npm run typecheck && npm run test:e2e            # e2e/chat.spec.ts + e2e/voice.spec.ts (system Edge)
```

## Current results
- Playwright `e2e/chat.spec.ts`: 2 / 2 pass.
  - Officer in English: a document answer with a doa.gov.lk citation and the retrieved panel, a live field answer,
    and a refusal with an ESC reference.
  - Farmer in Sinhala: chips, a Sinhala answer with citations, and the officer panel hidden.
- Screenshots: `docs/phase-reports/phase-6-figures/`.
- Playwright `e2e/voice.spec.ts` (Phase 6b): the browser's microphone is replaced by a WAV of a Sinhala question
  spoken by our Piper voice (`e2e/fixtures/si_question.wav`, made by
  `python -m member4_chatbot.pipeline.voice_e2e_fixture`). Farmer: record → transcript in the input → send →
  Sinhala answer → speaker plays our TTS WAV. Officer: compact mic and the speech-model card. Results and
  screenshots: `docs/phase-reports/phase-6b.md`, `phase-6b-figures/`.

## Phase 7: cross-links and faster speech

- **Arriving from a cross-link:** the twin's "Ask SoilMind" or the Advisory page's "Why?" opens this page with the
  question in the input box (router state). A chip shows what it is about ("About: south-east · nitrogen · A3-05,
  A3-06", ×). The context goes with the next question unless the chip is removed. A reload does not pre-fill again.
- **Map links:** an answer about a zone or sensor shows "Show on the map · …" buttons that open the twin with them
  highlighted.
- **Degraded answers:** if a model step was replaced (`degraded`), a small note says so.
- **Speech:**
  - Answers are spoken part by part (`/api/m4/tts/parts`): the first sentence plays while the rest is synthesised.
  - A farmer's answer has its first part fetched in advance.
  - If TTS fails, the message says the full answer is shown above as text.
