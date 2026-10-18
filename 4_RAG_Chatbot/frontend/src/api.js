const BASE = import.meta.env.VITE_API_BASE || "";

export async function sendChat({ message, history, soil }) {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, soil }),
  });
  if (!res.ok) {
    let detail = "Something went wrong. Try again.";
    try {
      const data = await res.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${BASE}/api/health`);
  if (!res.ok) throw new Error("Backend not reachable");
  return res.json();
}
