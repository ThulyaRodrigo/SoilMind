import React, { useEffect, useRef, useState } from "react";
import { getHealth, sendChat } from "./api.js";
import Answer from "./Answer.jsx";
import FieldPanel from "./FieldPanel.jsx";

const EMPTY_SOIL = { crop: "", ph: "", moisture: "", n: "", p: "", k: "" };

const STARTERS = [
  "Why are my rice leaves turning yellow?",
  "මගේ වී ගොයමේ කොළ කහ පාට වෙනවා. ඇයි?",
  "පසේ pH අගය අඩු නම් මොකද කරන්නේ?",
];

const num = (v) => (v === "" || Number.isNaN(parseFloat(v)) ? null : parseFloat(v));

function soilPayload(s) {
  const payload = {
    crop: s.crop || null,
    ph: num(s.ph),
    moisture: num(s.moisture),
    n: num(s.n),
    p: num(s.p),
    k: num(s.k),
  };
  return Object.values(payload).some((v) => v !== null) ? payload : null;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [soil, setSoil] = useState(EMPTY_SOIL);
  const [panelOpen, setPanelOpen] = useState(() => window.innerWidth > 860);
  const [health, setHealth] = useState({ state: "checking" });
  const endRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then((h) => setHealth({ state: "ok", ...h }))
      .catch(() => setHealth({ state: "down" }));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  async function submit(text) {
    const message = text.trim();
    if (!message || loading) return;
    setError("");
    setInput("");

    const history = messages.slice(-6).map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setLoading(true);

    try {
      const data = await sendChat({ message, history, soil: soilPayload(soil) });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources, mode: data.mode },
      ]);
    } catch (e) {
      setError(e.message === "Failed to fetch" ? "Cannot reach the backend. Check that it is running on port 8000." : e.message);
    } finally {
      setLoading(false);
    }
  }

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(input);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <h1>SoilMind</h1>
        <p>Soil advice from official guidelines, in your language</p>
      </header>

      {health.state === "down" && (
        <div className="banner warn" role="alert">
          The backend is not running. In the <code>backend</code> folder run <code>uvicorn app.main:app --reload</code>.
        </div>
      )}
      {health.state === "ok" && !health.llm_enabled && (
        <div className="banner info">
          AI answers are off. Add <code>ANTHROPIC_API_KEY</code> to <code>backend/.env</code>. Until then you will see the
          matching guideline passages only.
        </div>
      )}

      <main className="layout">
        <aside className="panel">
          <div className="strata" aria-hidden="true" />
          <button
            type="button"
            className="panel-toggle"
            aria-expanded={panelOpen}
            onClick={() => setPanelOpen((o) => !o)}
          >
            Your field
            <span className="chev" aria-hidden="true">{panelOpen ? "–" : "+"}</span>
          </button>
          {panelOpen && (
            <FieldPanel
              soil={soil}
              onChange={(k, v) => setSoil((s) => ({ ...s, [k]: v }))}
              onClear={() => setSoil(EMPTY_SOIL)}
            />
          )}
        </aside>

        <section className="chat" aria-label="Chat">
          <div className="messages" role="log" aria-live="polite">
            {messages.length === 0 && (
              <div className="empty">
                <h2>Ask about your soil or crop</h2>
                <p>
                  Type in Sinhala, Tamil or English. Answers come only from the documents loaded in the knowledge base
                  {health.state === "ok" ? ` (${health.documents} documents).` : "."}
                </p>
                <div className="starters">
                  {STARTERS.map((s) => (
                    <button key={s} type="button" onClick={() => submit(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) =>
              m.role === "user" ? (
                <div className="msg user" key={i}>
                  {m.content}
                </div>
              ) : (
                <div className="msg bot" key={i}>
                  <Answer text={m.content} sources={m.sources} />
                </div>
              )
            )}

            {loading && <div className="msg bot pending">Searching the guidelines…</div>}
            {error && (
              <div className="banner warn" role="alert">
                {error}
              </div>
            )}
            <div ref={endRef} />
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              submit(input);
            }}
          >
            <label htmlFor="q" className="sr-only">
              Your question
            </label>
            <textarea
              id="q"
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type your question here"
            />
            <button type="submit" disabled={loading || !input.trim()}>
              Send
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}
