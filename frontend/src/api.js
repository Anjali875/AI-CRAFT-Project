const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function post(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = "";
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      detail = res.statusText;
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

export function predictPCOS(payload) {
  return post("/api/predict/pcos", payload);
}

export function predictEndo(payload) {
  return post("/api/predict/endo", payload);
}
export function sendChat(payload) {
  return post("/api/chat", payload);
}

export async function downloadReport(payload) {
  const res = await fetch(`${API_BASE}/api/predict/download-report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = "";
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      detail = res.statusText;
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "womens_health_screening_report.pdf";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}