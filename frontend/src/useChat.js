import { useState } from "react";
import { useResult } from "./ResultContext";
import { sendChat } from "./api";

const COND_NAME = { pcos: "PCOS", endo: "Endometriosis" };

export const GENERAL_CHIPS = [
  "What is PCOS?",
  "What is endometriosis?",
  "Common symptoms to watch for",
];
export const RESULT_CHIPS = [
  "Help me understand my result",
  "What do my contributing factors mean?",
  "What lifestyle changes can help?",
];

export function useChat() {
  const { result } = useResult();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const hasResult = !!result;
  const condName = hasResult ? COND_NAME[result.condition] ?? "your" : null;
  const chips = hasResult ? RESULT_CHIPS : GENERAL_CHIPS;

  const handleSend = async (raw) => {
    const text = (raw ?? input).trim();
    if (!text || loading) return;
    setError("");
    const priorHistory = messages
      .slice(-8)
      .map((m) => ({ role: m.role, content: m.content.slice(0, 2000) }));
    setMessages((prev) => [...prev, { role: "user", content: text.slice(0, 500) }]);
    setInput("");

    const payload = { message: text.slice(0, 500), history: priorHistory, symptoms: {} };
    if (hasResult) {
      payload.condition = result.condition;
      payload.risk_level = result.risk_level;
      payload.risk_score =
        result.probability ?? (result.risk_percentage != null ? result.risk_percentage / 100 : 0);
      payload.contributing_factors = result.contributing_factors || [];
    }

    try {
      setLoading(true);
      const data = await sendChat(payload);
      setMessages((prev) => [
        ...prev,
        { role: "model", content: data.response ?? "Sorry, I couldn't generate a response." },
      ]);
    } catch (e) {
      if (String(e.message).startsWith("429")) {
        setError("You're sending messages quickly — please wait a moment and try again.");
      } else {
        setError("I couldn't reach the assistant. Make sure the backend is running, then try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setMessages([]);
    setInput("");
    setError("");
  };

  return { messages, input, setInput, loading, error, handleSend, hasResult, condName, chips, clear };
}