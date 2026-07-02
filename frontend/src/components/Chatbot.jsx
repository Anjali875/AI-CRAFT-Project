import { useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Bot, X, Send, Sparkles } from "lucide-react";
import { useResult } from "../ResultContext";
import { useChat } from "../useChat";
import FormattedMessage from "./ChatMessageContent";

export default function Chatbot() {
  const navigate = useNavigate();
  const { chatOpen, setChatOpen } = useResult();
  const { messages, input, setInput, loading, error, handleSend, hasResult, condName, chips } = useChat();
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!chatOpen) {
    return (
      <button
        onClick={() => setChatOpen(true)}
        aria-label="Open AI Health Assistant"
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-ai text-white shadow-lg flex items-center justify-center hover:opacity-90 transition-opacity cursor-pointer"
      >
        <Bot size={24} />
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-[360px] max-w-[calc(100vw-2rem)] h-[520px] max-h-[calc(100vh-3rem)] bg-white rounded-card border border-divider shadow-xl flex flex-col overflow-hidden">
      <div className="bg-ai-light px-4 py-3 flex items-center justify-between border-b border-divider">
        <div className="flex items-center gap-2">
          <span className="w-9 h-9 rounded-full bg-ai flex items-center justify-center">
            <Bot size={18} className="text-white" />
          </span>
          <div>
            <p className="font-heading font-semibold text-charcoal text-sm">AI Health Assistant</p>
            <p className="text-xs text-body flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500" /> Online
            </p>
          </div>
        </div>
        <button onClick={() => setChatOpen(false)} aria-label="Close chat" className="text-body hover:text-charcoal cursor-pointer">
          <X size={20} />
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
        <div className="bg-ai-light rounded-input px-3 py-2 text-sm text-charcoal self-start max-w-[85%]">
          {hasResult
            ? `Hi! I can help you understand your ${condName} screening result. Ask me anything about it.`
            : "Hi! I'm your AI health assistant. Ask me anything about PCOS, endometriosis, or menstrual health."}
        </div>

        {!hasResult && messages.length === 0 && (
          <div className="rounded-input border border-divider p-3 self-start max-w-[90%]">
            <p className="text-xs text-body mb-2">
              Want a personalized likelihood estimate? Take a quick screening and I can talk you through it.
            </p>
            <button
              onClick={() => { setChatOpen(false); navigate("/screening"); }}
              className="inline-flex items-center gap-1 text-xs font-medium text-ai hover:opacity-80 cursor-pointer"
            >
              <Sparkles size={13} /> Start a screening
            </button>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="bg-pcos text-white rounded-input px-3 py-2 text-sm self-end max-w-[85%]">
              {m.content}
            </div>
          ) : (
            <div key={i} className="bg-ai-light text-charcoal rounded-input px-3 py-2 self-start max-w-[90%]">
              <FormattedMessage text={m.content} />
            </div>
          )
        )}

        {loading && (
          <div className="bg-ai-light rounded-input px-3 py-2 self-start">
            <span className="flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-ai animate-bounce" style={{ animationDelay: "0ms" }} />
              <span className="w-1.5 h-1.5 rounded-full bg-ai animate-bounce" style={{ animationDelay: "150ms" }} />
              <span className="w-1.5 h-1.5 rounded-full bg-ai animate-bounce" style={{ animationDelay: "300ms" }} />
            </span>
          </div>
        )}

        {messages.length === 0 && (
          <div className="flex flex-wrap gap-2 mt-1">
            {chips.map((c) => (
              <button
                key={c}
                onClick={() => handleSend(c)}
                className="text-xs border border-ai text-ai rounded-pill px-3 py-1 hover:bg-ai-light transition-colors cursor-pointer"
              >
                {c}
              </button>
            ))}
          </div>
        )}

        {error && <p className="text-xs text-deep-rose self-start">{error}</p>}
      </div>

      <div className="border-t border-divider p-3">
        <div className="flex items-center gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value.slice(0, 500))}
            onKeyDown={onKeyDown}
            placeholder="Type your message..."
            className="flex-1 rounded-pill border border-divider px-4 py-2 text-sm text-charcoal focus:outline-none focus:border-ai"
          />
          <button
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            aria-label="Send"
            className="w-9 h-9 rounded-full bg-ai text-white flex items-center justify-center disabled:opacity-40 hover:opacity-90 transition-opacity cursor-pointer shrink-0"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-[10px] text-body mt-1 text-right">{input.length}/500</p>
      </div>
    </div>
  );
}