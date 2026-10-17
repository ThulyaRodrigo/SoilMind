// Voice for the chat page (Phase 6b): recording a question (MediaRecorder + live waveform) and speaking answers.
// Speech is turned into text and back by our local models on the backend (/api/m4/stt, /api/m4/tts).
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@shared/api/client";
import type { Lang } from "@shared/types";

export const MAX_RECORD_SECONDS = 30; // one Whisper window; the backend caps uploads at 30 s too

function pickMime(): string {
  const options = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/webm", "audio/mp4"];
  return options.find((m) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) ?? "";
}

export type RecorderState = "idle" | "recording" | "stopping";

/** Microphone recorder. `start()` asks for the microphone; the returned promise of `stop()` gives the recording. */
export function useRecorder(onDone: (blob: Blob) => void) {
  const [state, setState] = useState<RecorderState>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const cleanup = useRef<() => void>(() => {});
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  const stop = useCallback(() => {
    if (rec.current && rec.current.state === "recording") {
      setState("stopping");
      rec.current.stop();
    }
  }, []);

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const mime = pickMime();
    const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    const ctx = new AudioContext();
    const node = ctx.createAnalyser();
    node.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(node);
    const t0 = performance.now();
    const timer = window.setInterval(() => {
      const s = (performance.now() - t0) / 1000;
      setElapsed(s);
      if (s >= MAX_RECORD_SECONDS) stop();
    }, 200);
    cleanup.current = () => {
      window.clearInterval(timer);
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
      setAnalyser(null);
    };
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
      cleanup.current();
      setState("idle");
      doneRef.current(new Blob(chunks, { type: recorder.mimeType || mime || "audio/webm" }));
    };
    rec.current = recorder;
    recorder.start(250);
    setElapsed(0);
    setAnalyser(node);
    setState("recording");
  }, [stop]);

  useEffect(() => () => cleanup.current(), []);
  return { state, elapsed, analyser, start, stop };
}

/** Live waveform of the microphone (time-domain samples of the AnalyserNode). */
export function Waveform({ analyser, height = 36 }: { analyser: AnalyserNode | null; height?: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvas.current;
    if (!el || !analyser) return;
    const g = el.getContext("2d");
    if (!g) return;
    const data = new Uint8Array(analyser.fftSize);
    const color = getComputedStyle(el).color;
    let raf = 0;
    const draw = () => {
      const w = (el.width = el.clientWidth * devicePixelRatio);
      const h = (el.height = el.clientHeight * devicePixelRatio);
      analyser.getByteTimeDomainData(data);
      g.clearRect(0, 0, w, h);
      g.lineWidth = 2 * devicePixelRatio;
      g.strokeStyle = color;
      g.beginPath();
      for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * w;
        const y = (data[i] / 255) * h;
        if (i === 0) g.moveTo(x, y);
        else g.lineTo(x, y);
      }
      g.stroke();
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [analyser]);
  return <canvas ref={canvas} className="wave" style={{ height }} aria-hidden="true" />;
}

export type PlayState = "idle" | "loading" | "playing";

/** One player for the whole chat: starting an answer stops the one playing.
 *
 *  Phase 7: an answer is spoken part by part (/api/m4/tts/parts: the first part is one sentence). Part 1 plays as
 *  soon as it is synthesised, while the next part is fetched, so a long Sinhala answer starts speaking after about
 *  one sentence instead of after the whole answer. Audio is fetched once per part and kept for replays.
 *  `prefetch()` prepares the first part in advance (used when a farmer's answer arrives). */
export function usePlayer(onError: (msg: string) => void) {
  const audio = useRef<HTMLAudioElement | null>(null);
  const parts = useRef(new Map<string, Promise<string[]>>()); // answer id -> its parts
  const urls = useRef(new Map<string, Promise<string>>()); // "id#i" -> object URL of part i
  const seq = useRef(0); // a newer play() or stop() cancels a play() still fetching its audio
  const [current, setCurrent] = useState<{ id: string; state: PlayState } | null>(null);

  const partsOf = useCallback((id: string, text: string, lang: Lang) => {
    let p = parts.current.get(id);
    if (!p) {
      p = api.m4.ttsParts({ text, lang }).then((r) => r.parts).catch(() => [text]);
      parts.current.set(id, p);
    }
    return p;
  }, []);
  const urlOf = useCallback((id: string, i: number, text: string, lang: Lang) => {
    const key = `${id}#${i}`;
    let u = urls.current.get(key);
    if (!u) {
      u = api.m4.tts({ text, lang }).then((b) => URL.createObjectURL(b));
      u.catch(() => urls.current.delete(key)); // a failed part is fetched again next time
      urls.current.set(key, u);
    }
    return u;
  }, []);

  const stop = useCallback(() => {
    seq.current += 1;
    audio.current?.pause();
    audio.current = null;
    setCurrent(null);
  }, []);

  const prefetch = useCallback(
    (id: string, text: string, lang: Lang) => {
      void partsOf(id, text, lang).then((ps) => ps[0] && urlOf(id, 0, ps[0], lang).catch(() => undefined));
    },
    [partsOf, urlOf],
  );

  const play = useCallback(
    async (id: string, text: string, lang: Lang) => {
      stop();
      const token = seq.current;
      setCurrent({ id, state: "loading" });
      try {
        const ps = await partsOf(id, text, lang);
        for (let i = 0; i < ps.length; i++) {
          const url = await urlOf(id, i, ps[i], lang);
          if (seq.current !== token) return;
          if (i + 1 < ps.length) void urlOf(id, i + 1, ps[i + 1], lang).catch(() => undefined); // one part ahead
          const a = new Audio(url);
          audio.current = a;
          const ended = new Promise<void>((resolve) => {
            a.onended = () => resolve();
            a.onpause = () => resolve();
          });
          await a.play();
          if (audio.current === a) setCurrent({ id, state: "playing" });
          await ended;
          if (seq.current !== token) return;
        }
        setCurrent(null);
      } catch (err) {
        if (seq.current !== token) return;
        setCurrent(null);
        if ((err as Error).name !== "AbortError") onError((err as Error).message);
      }
    },
    [stop, onError, partsOf, urlOf],
  );

  useEffect(
    () => () => {
      audio.current?.pause();
      urls.current.forEach((u) => void u.then((x) => URL.revokeObjectURL(x)).catch(() => undefined));
    },
    [],
  );
  return { current, play, stop, prefetch };
}
