# SoilMind Member 4: Multilingual RAG Chatbot & Voice Assistant

Welcome to the **SoilMind AI Chatbot & Voice Assistant** component (Member 4). This module provides an intelligent agricultural advisory service supporting both Sinhala and English, combining a Retrieval-Augmented Generation (RAG) engine with real-time Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.

---

## 🌟 Key Features

1. **Multilingual RAG Engine**
   - Natural language comprehension in Sinhala and English.
   - Vector-based document retrieval using ChromaDB and cross-encoder reranking.
   - Grounded, extractive answer generation with source passage attribution.
   - Out-of-scope query filtering and fallback handling.

2. **Voice Assistant Pipeline**
   - **STT (Speech-to-Text)**: Speech recognition supporting conversational Sinhala and English powered by fine-tuned Whisper models.
   - **TTS (Text-to-Speech)**: Expressive Sinhala audio synthesis using SinhalaVITS with automated Sinhala number normalization and phonetic preprocessing.
   - Voice audio recording and playback directly integrated into the web client.

3. **Field & Telemetry Contextualization**
   - Ingests IoT telemetry and field profiles (soil pH, moisture, NPK values, crop types) to provide localized and field-specific recommendations.

4. **Rigorous Evaluation & Testing**
   - Automated benchmark suites measuring Retrieval Recall, MRR, Answer Faithfulness, and Sinhala WER/CER.
   - Comprehensive unit and integration test coverage across all pipeline stages.

---

## 📁 Project Structure

```
4_RAG_Chatbot/
├── README.md                          # Main documentation for Member 4
├── backend/                           # Backend API and Machine Learning service
│   ├── README.md                      # Backend setup and API guide
│   ├── member4_chatbot/               # Core Python package
│   │   ├── __init__.py
│   │   ├── lang.py                    # Language detection & normalizers (Sinhala / English)
│   │   ├── models.py                  # Pydantic schemas (ChatRequest, VoiceRequest, etc.)
│   │   ├── paths.py                   # Centralized path resolver for datasets and models
│   │   ├── router.py                  # FastAPI router exposing chat and voice endpoints
│   │   ├── chat/                      # RAG logic: store, retrieval, reranker, extractive QA
│   │   ├── collect/                   # Knowledge source scrapers & data acquisition
│   │   ├── dataset/                   # Synthetic QA generation, templates, paraphrasing
│   │   ├── eval/                      # RAG and intent benchmark runners
│   │   ├── ingest/                    # Document cleaning, chunking, and ChromaDB indexing
│   │   ├── pipeline/                  # End-to-end execution pipelines and orchestrators
│   │   └── voice/                     # Whisper STT, SinhalaVITS TTS, audio normalization
│   ├── tests/                         # Test suite (11 test files covering chat, voice, RAG)
│   ├── notebooks/                     # Exploratory analysis and model development notebooks
│   ├── data/
│   │   ├── training/                  # Curated domain intents, QA pairs, out-of-scope examples
│   │   └── processed/                 # Chunked passage corpus and extraction logs
│   ├── eval/                          # Evaluation datasets, testsets, and benchmark results
│   └── artifacts/                     # Metrics, evaluation outputs, model cards (configs only)
└── frontend/                          # Web user interfaces
    ├── README.md                      # Frontend documentation
    ├── package.json, vite.config.js   # Build configurations
    ├── src/                           # Original React JSX frontend
    │   ├── App.jsx, api.js, main.jsx, styles.css, Answer.jsx, FieldPanel.jsx
    └── member4_chatbot_full/          # Full production TypeScript UI
        ├── ChatPage.tsx               # Main chat interface with telemetry & voice modal
        ├── voice.tsx                  # Audio recorder & player hook
        ├── chat.css                   # Responsive UI stylesheet
        └── README.md
```

---

## 🚀 Getting Started

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create and activate a Python virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r notebooks/requirements.txt
```

#### Running the Backend Service

```bash
uvicorn member4_chatbot.router:app --host 0.0.0.0 --port 8000 --reload
```

The OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

### 2. Frontend Setup

```bash
# Navigate to the frontend directory
cd ../frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```

Visit `http://localhost:5173` to interact with the chatbot interface.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send user message with optional field context; returns RAG answer with sources |
| `POST` | `/api/v1/voice/transcribe` | Upload voice audio file (`.wav`/`.webm`) for STT transcription |
| `POST` | `/api/v1/voice/synthesize` | Convert text into natural Sinhala audio speech |
| `GET`  | `/api/v1/health` | Health check endpoint |
| `POST` | `/api/v1/eval/run` | Trigger evaluation benchmark across the test dataset |

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## 👥 Authors & Contribution
- **Lakna** (`AI_chatbot_Lakna`) — Member 4: Multilingual RAG Chatbot & Voice Assistant
- SoilMind Agricultural Intelligence Platform
