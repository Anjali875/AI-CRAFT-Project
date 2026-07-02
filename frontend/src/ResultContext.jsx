import { createContext, useContext, useState } from "react";

const ResultContext = createContext(null);

export function ResultProvider({ children }) {
  const [result, setResult] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);

  return (
    <ResultContext.Provider value={{ result, setResult, chatOpen, setChatOpen }}>
      {children}
    </ResultContext.Provider>
  );
}

export function useResult() {
  const ctx = useContext(ResultContext);
  if (!ctx) throw new Error("useResult must be used inside ResultProvider");
  return ctx;
}