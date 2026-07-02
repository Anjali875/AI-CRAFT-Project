import { Routes, Route, Navigate } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Chatbot from "./components/Chatbot";
import Home from "./pages/Home";
import Screening from "./pages/Screening";
import Questionnaire from "./pages/Questionnaire";
import Results from "./pages/Results";
import HealthLibrary from "./pages/HealthLibrary";
import About from "./pages/About";
import Assistant from "./pages/Assistant";

function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/screening" element={<Screening />} />
          <Route path="/questionnaire/:condition" element={<Questionnaire />} />
          <Route path="/results" element={<Results />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/health-library" element={<HealthLibrary />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <Footer />
      <Chatbot />
    </div>
  );
}

export default App;