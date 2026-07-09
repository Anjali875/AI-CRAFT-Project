import { useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Bot, Send, Sparkles, Trash2, ClipboardCheck, ShieldCheck, TriangleAlert,
  ChevronRight, Check, Info,
} from "lucide-react";
import { useResult } from "../ResultContext";
import { useChat } from "../useChat";
import FormattedMessage from "../components/ChatMessageContent";

const TOPICS = {
  general: ["Understanding PCOS", "Understanding endometriosis", "Symptoms to watch for", "When to see a doctor", "Myths vs facts"],
  pcos: ["Understanding PCOS", "Causes of PCOS", "PCOS and fertility", "Managing PCOS naturally", "PCOS myths vs facts"],
  endo: ["Understanding endometriosis", "Causes of endometriosis", "Endometriosis and fertility", "Managing endometriosis pain", "Endometriosis myths vs facts"],
};

const HOW_TO = [
  "Ask general questions about PCOS, endometriosis, and menstrual health.",
  "After a screening, ask what your specific result means.",
  "It explains and educates — it can't diagnose or replace a doctor.",
];

const COND_NAME = { pcos: "PCOS", endo: "Endometriosis" };
const BAND_LABEL = { Low: "Lower likelihood", Moderate: "Moderate likelihood", High: "Higher likelihood" };

export default function Assistant() {
  const navigate = useNavigate();
  const { result } = useResult();
  const { messages, input, setInput, loading, error, handleSend, hasResult, condName, clear } = useChat();
  const scrollRef = useRef(null);

  const mode = hasResult ? result.condition : "general";
  const topics = TOPICS[mode] ?? TOPICS.general;

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-6 grid grid-cols-1 lg:grid-cols-[240px_1fr_260px] gap-5">
      {/* LEFT RAIL — orientation */}
      <aside className="hidden lg:flex flex-col gap-4">
        <div>
          <h2 className="font-heading font-semibold text-charcoal">AI Health Assistant</h2>
          <p className="text-xs text-body">Your women's health companion.</p>
        </div>
        <button
          onClick={clear}
          className="inline-flex items-center gap-2 rounded-input bg-ai-light text-ai text-sm font-medium px-3 py-2 hover:opacity-90 cursor-pointer"
        >
          <Sparkles size={15} /> New chat
        </button>

        <div className="rounded-card bg-white border border-divider p-4">
          <div className="flex items-center gap-2 mb-3">
            <Info size={16} className="text-ai" />
            <p className="text-sm font-semibold text-charcoal">How to use this assistant</p>
          </div>
          <ul className="flex flex-col gap-2.5">
            {HOW_TO.map((t, i) => (
              <li key={i} className="flex gap-2 text-xs text-body">
                <Check size={14} className="text-ai shrink-0 mt-0.5" />
                <span>{t}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-card bg-ai-light border border-ai/20 p-4">
          <img
            src="/illustrations/assistant-hero.svg"
            alt="AI health assistant illustration"
            className="w-full h-24 object-contain"
          />
          <div className="flex items-center gap-2 mt-3 mb-1">
            <ShieldCheck size={15} className="text-ai" />
            <p className="text-sm font-semibold text-charcoal">Your privacy matters</p>
          </div>
          <p className="text-xs text-body">Your conversation isn't stored or shared — it stays in this session only.</p>
        </div>
      </aside>

      {/* CENTER CHAT */}
      <div className="flex flex-col rounded-card bg-white border border-divider overflow-hidden" style={{ height: "calc(100vh - 7rem)" }}>
        <div className="flex items-center justify-between px-5 py-3 border-b border-divider">
          <div className="flex items-center gap-3">
            <span className="w-10 h-10 rounded-full bg-ai flex items-center justify-center shrink-0">
              <Bot size={20} className="text-white" />
            </span>
            <div>
              <p className="font-heading font-semibold text-charcoal text-sm">AI Health Assistant</p>
              <p className="text-xs text-body">
                {hasResult ? `Discussing your ${condName} result` : "Ask me anything about PCOS or endometriosis"}
              </p>
            </div>
          </div>
          {messages.length > 0 && (
            <button
              onClick={clear}
              className="inline-flex items-center gap-1 text-xs text-body hover:text-deep-rose transition-colors cursor-pointer"
            >
              <Trash2 size={14} /> Clear chat
            </button>
          )}
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
          <div className="bg-ai-light rounded-input px-4 py-3 text-charcoal self-start max-w-[85%]">
            {hasResult
              ? `Hi! I can help you understand your ${condName} screening result. Ask me anything about it.`
              : "Hi! I'm your AI health assistant. Ask me anything about PCOS, endometriosis, or menstrual health."}
          </div>

          {!hasResult && messages.length === 0 && (
            <div className="rounded-input border border-divider p-4 self-start max-w-[90%]">
              <p className="text-sm text-body mb-2">Want a personalized likelihood estimate? Take a quick screening and I can talk you through it.</p>
              <button onClick={() => navigate("/screening")} className="inline-flex items-center gap-1 text-sm font-medium text-ai hover:opacity-80 cursor-pointer">
                <Sparkles size={15} /> Start a screening
              </button>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="bg-pcos text-white rounded-input px-4 py-2.5 self-end max-w-[80%]">{m.content}</div>
            ) : (
              <div key={i} className="bg-ai-light text-charcoal rounded-input px-4 py-3 self-start max-w-[90%]">
                <FormattedMessage text={m.content} />
              </div>
            )
          )}

          {loading && (
            <div className="bg-ai-light rounded-input px-4 py-3 self-start">
              <span className="flex gap-1.5">
                <span className="w-2 h-2 rounded-full bg-ai animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2 h-2 rounded-full bg-ai animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2 h-2 rounded-full bg-ai animate-bounce" style={{ animationDelay: "300ms" }} />
              </span>
            </div>
          )}

          {error && <p className="text-sm text-deep-rose self-start">{error}</p>}
        </div>

        <div className="border-t border-divider p-3">
          <div className="flex items-center gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value.slice(0, 500))}
              onKeyDown={onKeyDown}
              placeholder="Type your message here..."
              className="flex-1 rounded-pill border border-divider px-5 py-3 text-charcoal focus:outline-none focus:border-ai"
            />
            <button
              onClick={() => handleSend()}
              disabled={loading || !input.trim()}
              aria-label="Send"
              className="w-11 h-11 rounded-full bg-ai text-white flex items-center justify-center disabled:opacity-40 hover:opacity-90 transition-opacity cursor-pointer shrink-0"
            >
              <Send size={18} />
            </button>
          </div>
          <p className="text-[11px] text-body mt-1 text-center">
            This assistant provides general information only and is not a substitute for professional medical advice.
          </p>
        </div>
      </div>

      {/* RIGHT RAIL */}
      <aside className="hidden lg:flex flex-col gap-4">
        {hasResult && (
          <div className="rounded-card bg-white border border-divider p-4">
            <div className="flex items-center gap-2 mb-3">
              <ClipboardCheck size={16} className="text-ai" />
              <p className="text-sm font-semibold text-charcoal">Your current screening</p>
            </div>
            <p className="text-xs text-body mb-3">{COND_NAME[result.condition]} Risk Assessment</p>
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-body">Result</span>
              <span className="font-semibold text-charcoal">{BAND_LABEL[result.risk_level] ?? "—"}</span>
            </div>
            <div className="flex items-center justify-between text-sm mb-4">
              <span className="text-body">Likelihood</span>
              <span className="font-semibold text-charcoal">{result.risk_percentage}%</span>
            </div>
            <button
              onClick={() => navigate("/results", { state: { result } })}
              className="w-full rounded-pill border border-ai text-ai text-sm font-medium py-2 hover:bg-ai-light transition-colors cursor-pointer"
            >
              View full results
            </button>
          </div>
        )}

        <div className="rounded-card bg-white border border-divider p-4">
          <p className="text-sm font-semibold text-charcoal mb-3">Suggested topics</p>
          <div className="flex flex-col gap-1">
            {topics.map((t) => (
              <button
                key={t}
                onClick={() => handleSend(t)}
                disabled={loading}
                className="flex items-center justify-between gap-2 text-left text-sm text-charcoal rounded-input px-2 py-2 hover:bg-ai-light hover:text-ai transition-colors cursor-pointer disabled:opacity-50"
              >
                {t} <ChevronRight size={14} className="shrink-0 text-body" />
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-card bg-blush border border-soft-rose p-4">
          <div className="flex items-center gap-2 mb-2">
            <TriangleAlert size={15} className="text-pcos" />
            <p className="text-sm font-semibold text-charcoal">Important reminder</p>
          </div>
          <p className="text-xs text-body">
            This information is educational only and doesn't replace medical advice. Please consult a healthcare professional for guidance specific to you.
          </p>
        </div>
      </aside>
    </div>
  );
}