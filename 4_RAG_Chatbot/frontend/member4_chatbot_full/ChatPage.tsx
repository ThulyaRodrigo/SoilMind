import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "@shared/api/client";
import { useApi } from "@shared/api/useApi";
import { useI18n } from "@shared/i18n/I18nContext";
import type { I18nKey } from "@shared/i18n/dictionary";
import { usePersona } from "@shared/persona/PersonaContext";
import { load, save } from "@shared/storage";
import type { ChatContext, ChatMapFocus, ChatResponse, Lang, LocalText, SttResponse } from "@shared/types";
import { Icon } from "@shared/ui/Icon";
import { PageHead } from "@shared/ui/PageHead";
import { MAX_RECORD_SECONDS, type PlayState, usePlayer, useRecorder, Waveform } from "./voice";
import "./chat.css";

type Entry =
  | { kind: "user"; text: string }
  | { kind: "bot"; r: ChatResponse }
  | { kind: "error"; text: string };

const AUTOPLAY_KEY = "soilmind.m4.autoplay";

const SUGGESTIONS: I18nKey[] = [
  "chatSuggestField",
  "chatSuggestUrea",
  "chatSuggestBph",
  "chatSuggestBlast",
  "chatSuggestWeeds",
  "chatSuggestLime",
];

const SENSOR_KEYS: { key: string; label: I18nKey; unit: string; digits: number }[] = [
  { key: "ph", label: "kvPh", unit: "", digits: 2 },
  { key: "n", label: "kvNitrogen", unit: " mg/kg", digits: 1 },
  { key: "moisture", label: "kvMoisture", unit: "%", digits: 1 },
];

function sessionId(): string {
  const k = "soilmind.m4.session";
  try {
    const have = sessionStorage.getItem(k);
    if (have) return have;
    const id = `s-${Math.random().toString(36).slice(2, 10)}`;
    sessionStorage.setItem(k, id);
    return id;
  } catch {
    return `s-${Math.random().toString(36).slice(2, 10)}`;
  }
}

/** UTC ISO -> "18 Oct, 14:00" (English) or "2026-10-18 14:00" (Sinhala), in Asia/Colombo time. */
function colombo(iso: string, lang: Lang): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  if (lang === "si") {
    // numeric date: Sinhala month abbreviations differ between browsers
    const p = (o: Intl.DateTimeFormatOptions) => d.toLocaleString("en-CA", { timeZone: "Asia/Colombo", ...o });
    return `${p({ year: "numeric", month: "2-digit", day: "2-digit" })} ${p({ hour: "2-digit", minute: "2-digit", hour12: false })}`;
  }
  return d.toLocaleString("en-GB", {
    timeZone: "Asia/Colombo",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function fmt(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return v.toLocaleString("en-US", { maximumFractionDigits: digits });
}

/** Wraps the key numbers (from the API's `highlights`) in <mark>, matching whole numbers only. */
function Marked({ text, marks }: { text: string; marks: string[] }) {
  if (!marks.length) return <>{text}</>;
  const esc = marks.map((m) => m.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(?<![\\d.])(${esc.join("|")})(?![\\d.])`, "g");
  const parts = text.split(re);
  return (
    <>
      {parts.map((part, i) => (i % 2 === 1 ? <mark key={i}>{part}</mark> : part))}
    </>
  );
}

/** A run of "a | b | c" lines is shown as a small table, as in the source document. */
function AnswerLines({ lines, marks = [], only }: { lines: string[]; marks?: string[]; only?: Set<string> }) {
  // `only`: highlight just these lines (in the full passage, the lines that were chosen as the answer)
  const marksFor = (l: string) => (only && !only.has(l.trim()) ? [] : marks);
  const blocks: { table: boolean; lines: string[] }[] = [];
  for (const l of lines) {
    const table = l.includes("|");
    const last = blocks[blocks.length - 1];
    if (last && last.table === table && table) last.lines.push(l);
    else blocks.push({ table, lines: [l] });
  }
  return (
    <>
      {blocks.map((b, i) =>
        b.table ? (
          <div className="ans-table" key={i}>
            <table>
              <tbody>
                {b.lines.map((row, j) => (
                  <tr key={j} className={j === 0 ? "head" : undefined}>
                    {row.split("|").map((c, k) => (
                      <td key={k}>
                        <Marked text={c.trim()} marks={marksFor(row)} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          b.lines.map((l, j) => (
            <p key={`${i}-${j}`}>
              <Marked text={l} marks={marksFor(l)} />
            </p>
          ))
        ),
      )}
    </>
  );
}

/** "Show on the map" for each zone / sensor the answer talks about: opens the twin with them highlighted. */
function MapLinks({ focus }: { focus: ChatMapFocus[] }) {
  const { t, lang } = useI18n();
  const navigate = useNavigate();
  const seen = new Set<string>();
  const items = focus.filter((f) => {
    const k = `${f.nodes.join(",")}|${f.variable ?? ""}`;
    if (!f.nodes.length || seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  if (!items.length) return null;
  return (
    <p className="maplinks">
      {items.map((f) => {
        const q = new URLSearchParams({ focus: f.nodes.join(","), label: f.label[lang] });
        if (f.variable) q.set("layer", f.variable);
        return (
          <button key={q.toString()} type="button" className="maplink" data-testid="maplink"
            onClick={() => navigate(`/twin?${q.toString()}`)}>
            <Icon name="map" />
            {t("xShowOnMap")} · {f.label[lang]}
          </button>
        );
      })}
    </p>
  );
}

function SpeakButton({ state, onPlay, onStop }: { state: PlayState; onPlay: () => void; onStop: () => void }) {
  const { t } = useI18n();
  const label = state === "loading" ? t("voiceLoading") : state === "playing" ? t("voiceStopPlay") : t("voiceListen");
  return (
    <button
      type="button"
      className={`speak ${state}`}
      onClick={state === "idle" ? onPlay : onStop}
      aria-label={label}
      data-testid="speak"
    >
      <Icon name={state === "playing" ? "stop" : "speaker"} />
      <span>{label}</span>
    </button>
  );
}

function BotBubble({
  r,
  officer,
  play,
  onPlay,
  onStop,
}: {
  r: ChatResponse;
  officer: boolean;
  play: PlayState;
  onPlay: () => void;
  onStop: () => void;
}) {
  const { t, lang } = useI18n();
  const esc = r.path === "refused";
  return (
    <div className={`msg bot${esc ? " esc" : ""}`}>
      <span className="av">
        <Icon name={esc ? "alert" : "leaf"} />
      </span>
      <div className="bub">
        {esc && <div className="esc-tag">{t("chatEscalated")}{r.escalation_ref ? ` · ${r.escalation_ref}` : ""}</div>}
        {r.note && <p className="note-line">{r.note}</p>}
        <AnswerLines lines={r.answer_lines} marks={r.highlights} />
        {r.degraded.length > 0 && <p className="unverified" data-testid="degraded">{t("xDegraded")}</p>}
        <MapLinks focus={r.map_focus} />
        {r.full_passage && r.citations[0] && (
          <details className="fullpass">
            <summary>{t("chatShowPassage")}</summary>
            <AnswerLines
              lines={r.full_passage.split("\n")}
              marks={r.highlights}
              only={new Set(r.answer_lines.map((l) => l.trim()))}
            />
            <div className="fp-cite">
              {r.citations[0].url.startsWith("http") ? (
                <a href={r.citations[0].url} target="_blank" rel="noreferrer">
                  {r.citations[0].title}
                </a>
              ) : (
                r.citations[0].title
              )}
              {r.citations[0].page > 0 && <> · {t("chatPage", { page: r.citations[0].page })}</>}
            </div>
          </details>
        )}
        {r.live_values.length > 0 && (
          <p className="lives">
            {r.live_values.map((v) => (
              <span className="livechip" key={v.key}>
                <span className={`dot${v.status === "act" || v.status === "crit" ? " warn" : ""}`} />
                {v.label[lang]}
                {v.value !== null ? ` · ${fmt(v.value, 2)}${v.unit ? ` ${v.unit}` : ""}` : ""}
                {v.key.startsWith("plan:") ? ` · ${v.at}` : ""}
              </span>
            ))}
          </p>
        )}
        {r.citations.length > 0 && (
          <div className="cite">
            <Icon name="quote" />
            <span>
              {t("chatSources")}:{" "}
              {r.citations.map((c, i) => (
                <span key={`${c.passage_id}-${i}`} className="citechip" title={c.snippet}>
                  {c.url.startsWith("http") ? (
                    <a href={c.url} target="_blank" rel="noreferrer">
                      {c.title}
                    </a>
                  ) : (
                    c.title
                  )}
                  {c.page > 0 && <> · {t("chatPage", { page: c.page })}</>}
                  {officer && c.score !== null && <> · {fmt(c.score, 2)}</>}
                </span>
              ))}
            </span>
          </div>
        )}
        {officer && r.answer_lang === "si" && !r.sinhala_verified && (
          <p className="unverified">{t("chatSinhalaUnverified")}</p>
        )}
        <SpeakButton state={play} onPlay={onPlay} onStop={onStop} />
      </div>
    </div>
  );
}

export function ChatPage() {
  const { t, lang } = useI18n();
  const { persona } = usePersona();
  const officer = persona === "officer";
  const session = useMemo(sessionId, []);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  // Phase 7 cross-links: the twin's "Ask SoilMind" and the Advisory page's "Why?" open this page with a pre-filled
  // question and its context (router state). The context goes with the next question, unless the user removes it.
  const location = useLocation();
  const navigate = useNavigate();
  const [context, setContext] = useState<ChatContext | null>(null);
  const [contextLabel, setContextLabel] = useState<LocalText | null>(null);
  useEffect(() => {
    type CrossLink = { prefill?: string | null; context?: ChatContext | null; label?: LocalText | null };
    const st = location.state as CrossLink | null;
    if (!st || (!st.prefill && !st.context)) return;
    if (st.prefill) setInput(st.prefill);
    setContext(st.context ?? null);
    setContextLabel(st.label ?? null);
    navigate(location.pathname, { replace: true, state: null }); // a reload does not pre-fill again
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [location.state, location.pathname, navigate]);
  const [used, setUsed] = useState<Set<I18nKey>>(new Set());
  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [voiceMsg, setVoiceMsg] = useState<{ kind: "hint" | "warn"; text: string } | null>(null);
  const [lastStt, setLastStt] = useState<SttResponse | null>(null);
  const [autoplay, setAutoplay] = useState(() => load(AUTOPLAY_KEY) === "1");
  const autoplayRef = useRef(autoplay);
  autoplayRef.current = autoplay && !officer;
  const onPlayError = useCallback((msg: string) => setVoiceMsg({ kind: "warn", text: t("voiceTtsError", { msg }) }), [t]);
  const player = usePlayer(onPlayError);
  const recorder = useRecorder(async (blob) => {
    setTranscribing(true);
    try {
      const r = await api.m4.stt(blob, lang);
      setLastStt(r);
      if (!r.text.trim()) {
        setVoiceMsg({ kind: "warn", text: t("voiceEmpty") });
      } else {
        setInput(r.text);
        setVoiceMsg({ kind: "hint", text: t("voiceCheck") });
        inputRef.current?.focus();
      }
    } catch (err) {
      setVoiceMsg({ kind: "warn", text: t("voiceSttError", { msg: (err as Error).message }) });
    } finally {
      setTranscribing(false);
    }
  });
  const recording = recorder.state !== "idle";

  async function toggleMic() {
    if (recording) return recorder.stop();
    player.stop();
    setVoiceMsg(null);
    try {
      await recorder.start();
    } catch (err) {
      setVoiceMsg({ kind: "warn", text: t("voiceMicError", { msg: (err as Error).message }) });
    }
  }

  function toggleAutoplay(on: boolean) {
    setAutoplay(on);
    save(AUTOPLAY_KEY, on ? "1" : "0");
    if (!on) player.stop();
  }
  const info = useApi(() => api.m4.modelInfo(), []);
  const kpis = useApi(() => api.m3.kpis(), [], 60000);

  const entriesRef = useRef(entries);
  entriesRef.current = entries;
  const last = [...entries].reverse().find((e): e is Extract<Entry, { kind: "bot" }> => e.kind === "bot")?.r;
  const threshold = info.data?.answer_config?.retrieval_threshold as number | undefined;

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [entries, busy]);

  async function send(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    setEntries((e) => [...e, { kind: "user", text: q }]);
    setInput("");
    setVoiceMsg(null);
    setBusy(true);
    try {
      const r = await api.m4.chat({ session_id: session, message: q, persona: persona ?? "farmer", ui_lang: lang, context });
      setContext(null);
      // the user turn above is already in `entries`; this bot turn goes right after it
      const id = `bot-${entriesRef.current.length}`;
      setEntries((e) => [...e, { kind: "bot", r }]);
      if (autoplayRef.current) void player.play(id, r.answer, r.answer_lang);
      else if (!officer) player.prefetch(id, r.answer, r.answer_lang); // the farmer's Listen then starts at once
    } catch (err) {
      setEntries((e) => [...e, { kind: "error", text: t("chatError", { msg: (err as Error).message }) }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHead
        officer={{ eyebrow: t("chatEyebrow"), title: t("chatHeadOfficer"), sub: t("chatSubOfficer") }}
        farmer={{ title: t("chatHeadFarmer"), sub: t("chatSubFarmer") }}
      />

      <div className="chatlayout">
        <div className="card chat">
          <div className="chat-log" ref={logRef} data-testid="chat-log">
            <div className="msg bot">
              <span className="av">
                <Icon name="leaf" />
              </span>
              <div className="bub">
                <p>{t("chatWelcome")}</p>
              </div>
            </div>
            {entries.map((e, i) =>
              e.kind === "user" ? (
                <div className="msg user" key={i}>
                  <span className="av">{officer ? "AO" : "SW"}</span>
                  <div className="bub">
                    <p>{e.text}</p>
                  </div>
                </div>
              ) : e.kind === "bot" ? (
                <BotBubble
                  key={i}
                  r={e.r}
                  officer={officer}
                  play={player.current?.id === `bot-${i}` ? player.current.state : "idle"}
                  onPlay={() => void player.play(`bot-${i}`, e.r.answer, e.r.answer_lang)}
                  onStop={player.stop}
                />
              ) : (
                <div className="msg bot esc" key={i}>
                  <span className="av">
                    <Icon name="alert" />
                  </span>
                  <div className="bub">
                    <p>{e.text}</p>
                  </div>
                </div>
              ),
            )}
            {busy && (
              <div className="msg bot" aria-live="polite">
                <span className="av">
                  <Icon name="leaf" />
                </span>
                <div className="bub typing" aria-label={t("chatThinking")}>
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}
          </div>
          <div className="chips">
            {SUGGESTIONS.filter((k) => !used.has(k)).map((k) => (
              <button
                key={k}
                type="button"
                disabled={busy}
                onClick={() => {
                  setUsed((u) => new Set(u).add(k));
                  void send(t(k));
                }}
              >
                {t(k)}
              </button>
            ))}
          </div>
          {!officer && (
            <div className={`voicebar${recording ? " rec" : ""}`} data-testid="voicebar">
              <button
                type="button"
                className="bigmic"
                onClick={() => void toggleMic()}
                disabled={transcribing || busy}
                aria-label={recording ? t("voiceStop") : t("micLabel")}
                aria-pressed={recording}
              >
                <Icon name={recording ? "stop" : "mic"} />
              </button>
              <div className="vb-main">
                <div className="vb-label">
                  {transcribing ? t("voiceTranscribing") : recording ? t("voiceListening") : t("voiceTapToSpeak")}
                  {recording && (
                    <span className="vb-time">
                      {Math.floor(recorder.elapsed)} / {MAX_RECORD_SECONDS} s
                    </span>
                  )}
                </div>
                {recording && <Waveform analyser={recorder.analyser} height={34} />}
                {transcribing && (
                  <div className="typing" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </div>
                )}
              </div>
              <label className="autoplay">
                <input type="checkbox" checked={autoplay} onChange={(ev) => toggleAutoplay(ev.target.checked)} />
                <Icon name="speaker" />
                {t("voiceAutoplay")}
              </label>
            </div>
          )}
          {voiceMsg && (
            <p className={`voicemsg ${voiceMsg.kind}`} role="status">
              {voiceMsg.text}
            </p>
          )}
          {context && (
            <p className="ctxchip" data-testid="chat-context">
              <Icon name={context.kind === "zone" ? "map" : "leaf"} />
              {t("xAbout", { what: contextLabel ? contextLabel[lang] : (context.product ?? context.zone ?? "") })}
              <button type="button" onClick={() => setContext(null)} aria-label={t("xAboutRemove")}>×</button>
            </p>
          )}
          <form
            className="composer"
            onSubmit={(ev) => {
              ev.preventDefault();
              void send(input);
            }}
          >
            {officer && (
              <button
                type="button"
                className={`mic${recording ? " rec" : ""}`}
                onClick={() => void toggleMic()}
                disabled={transcribing || busy}
                aria-label={recording ? t("voiceStop") : t("micLabel")}
                aria-pressed={recording}
              >
                <Icon name={recording ? "stop" : "mic"} />
              </button>
            )}
            {officer && recording ? (
              <div className="recline" aria-label={t("voiceListening")}>
                <Waveform analyser={recorder.analyser} height={28} />
                <span className="vb-time">
                  {Math.floor(recorder.elapsed)} / {MAX_RECORD_SECONDS} s
                </span>
              </div>
            ) : (
              <input
                ref={inputRef}
                value={input}
                onChange={(ev) => setInput(ev.target.value)}
                placeholder={transcribing ? t("voiceTranscribing") : t("placeholder")}
                aria-label={t("inputLabel")}
                maxLength={1000}
              />
            )}
            <button className="send" type="submit" aria-label={t("sendLabel")} disabled={busy || !input.trim()}>
              <Icon name="send" />
            </button>
          </form>
        </div>

        <aside className="chat-side">
          {officer && (
            <div className="card">
              <div className="card-h">
                <h3>{t("retrievedTitle")}</h3>
                <span className="sub">{t("lastTurn")}</span>
              </div>
              <div className="card-b retr">
                {!last && <p className="muted">{t("chatNoTurn")}</p>}
                {last && last.retrieved.length === 0 && <p className="muted">{t("chatNoRetrieval")}</p>}
                {last?.retrieved.map((d) => (
                  <div className="doc" key={d.passage_id}>
                    <div className="t">{d.title}</div>
                    <div className="s">
                      {d.doc} · {t("chatPage", { page: d.page })} · {d.lang}
                    </div>
                    <div className="sc">
                      <span className="b">
                        <i style={{ width: `${Math.max(0, Math.min(1, d.score)) * 100}%` }} />
                      </span>
                      <span className="n">{fmt(d.score, 2)}</span>
                    </div>
                  </div>
                ))}
                {threshold !== undefined && (
                  <div className="note">
                    <Icon name="alert" />
                    <span>{t("guardrailNote", { threshold: fmt(threshold, 2) })}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {officer && last && (
            <div className="card">
              <div className="card-h">
                <h3>{t("chatDetails")}</h3>
              </div>
              <div className="card-b">
                <div className="kv">
                  <span className="k">{t("chatIntent")}</span>
                  <span className="v mono">{last.intent}</span>
                </div>
                <div className="kv">
                  <span className="k">{t("chatConfidence")}</span>
                  <span className="v mono">{fmt(last.confidence, 2)}</span>
                </div>
                <div className="kv">
                  <span className="k">{t("chatPath")}</span>
                  <span className="v mono">{last.path}</span>
                </div>
                <div className="kv">
                  <span className="k">{t("chatSelection")}</span>
                  <span className="v mono">{last.selection ?? "–"}</span>
                </div>
                {last.field_profile && (
                  <div className="kv">
                    <span className="k">{t("chatFieldProfile")}</span>
                    <span className="v mono">
                      {[last.field_profile.zone, last.field_profile.water, last.field_profile.age && `${last.field_profile.age} mo`]
                        .filter(Boolean)
                        .join(" · ") || "–"}
                    </span>
                  </div>
                )}
                <div className="kv">
                  <span className="k">{t("chatRetrievalScore")}</span>
                  <span className="v mono">{fmt(last.retrieval_score, 2)}</span>
                </div>
                <div className="kv">
                  <span className="k">{t("chatLatency")}</span>
                  <span className="v mono">{fmt(last.latency_ms, 0)} ms</span>
                </div>
              </div>
            </div>
          )}

          {officer && lastStt && (
            <div className="card" data-testid="stt-card">
              <div className="card-h">
                <h3>{t("voiceModel")}</h3>
              </div>
              <div className="card-b">
                <div className="kv">
                  <span className="k">{t("voiceModel")}</span>
                  <span className="v mono">{lastStt.model.split(/[\\/]/).slice(-2).join("/")}</span>
                </div>
                <div className="kv">
                  <span className="k">{t("chatLatency")}</span>
                  <span className="v mono">
                    {fmt(lastStt.duration, 1)} s · {fmt(lastStt.latency_ms, 0)} ms
                  </span>
                </div>
                {lastStt.corrections.length > 0 && (
                  <p className="muted stt-fix">
                    {t("voiceCorrected", {
                      list: lastStt.corrections.map((c) => `${c.heard} → ${c.write}`).join(", "),
                    })}
                  </p>
                )}
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-h">
              <h3>{officer ? t("liveSensorOfficer") : t("liveSensorFarmer")}</h3>
            </div>
            <div className="card-b" data-testid="sensor-card">
              {kpis.loading && !kpis.data && <p className="muted">{t("chatSensorsLoading")}</p>}
              {kpis.error && !kpis.data && <p className="muted">{t("chatSensorsError")}</p>}
              {kpis.data &&
                SENSOR_KEYS.map((s) => (
                  <div className="kv" key={s.key}>
                    <span className="k">{t(s.label)}</span>
                    <span className="v">
                      {fmt(kpis.data!.kpis.averages[s.key], s.digits)}
                      {s.unit} · {t("chatFieldMean")}
                    </span>
                  </div>
                ))}
              {kpis.data?.last_reading_at && (
                <div className="kv">
                  <span className="k">{officer ? t("kvReadAt") : t("kvChecked")}</span>
                  <span className="v">{colombo(kpis.data.last_reading_at, lang)}</span>
                </div>
              )}
              {last && last.live_values.length > 0 && (
                <div className="lives side">
                  <div className="k">{t("chatLiveData")}</div>
                  {last.live_values.map((v) => (
                    <span className="livechip" key={v.key}>
                      <span className="dot" />
                      {v.label[lang]}
                      {v.value !== null ? ` · ${fmt(v.value, 2)}${v.unit ? ` ${v.unit}` : ""}` : ""}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </aside>
      </div>
    </>
  );
}
