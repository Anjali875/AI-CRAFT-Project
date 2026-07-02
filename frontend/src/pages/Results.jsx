import { useState } from "react";
import { useLocation, useNavigate, Link } from "react-router-dom";
import {
  ArrowLeft, RefreshCw, MessageCircle, Download, Info, ClipboardList,
  Scale, Calendar, Activity, Salad, Moon, Dumbbell, Stethoscope, Baby,
  Flame, Circle, Image as ImageIcon,
} from "lucide-react";
import Button from "../components/Button";
import { downloadReport } from "../api";

const CONDITION = {
  pcos: {
    name: "PCOS",
    full: "Polycystic Ovary Syndrome (PCOS)",
    headerBg: "bg-blush",
    accent: "text-pcos",
    chip: "bg-blush text-pcos",
    iconBg: "bg-soft-rose/40",
    other: "endo",
    otherName: "Endometriosis",
    meaning:
      "PCOS is a hormonal condition that affects how the ovaries work. It's common and manageable — early awareness and healthy habits can help you stay ahead of symptoms.",
    aspects: [
      { Icon: Scale, label: "Hormonal imbalance" },
      { Icon: Calendar, label: "Irregular periods" },
      { Icon: Circle, label: "Possible ovarian cysts" },
      { Icon: Activity, label: "Metabolic changes" },
    ],
    guidance: [
      { Icon: Activity, title: "Maintain a healthy weight", text: "Small, sustainable changes can support hormonal balance." },
      { Icon: Dumbbell, title: "Stay active", text: "Regular movement on most days supports metabolic health." },
      { Icon: Salad, title: "Eat balanced meals", text: "Whole foods and less added sugar can help how you feel day to day." },
      { Icon: Moon, title: "Rest & manage stress", text: "Consistent sleep and lower stress support hormone regulation." },
    ],
  },
  endo: {
    name: "Endometriosis",
    full: "Endometriosis",
    headerBg: "bg-endo-light",
    accent: "text-endo",
    chip: "bg-endo-light text-endo",
    iconBg: "bg-endo-light",
    other: "pcos",
    otherName: "PCOS",
    meaning:
      "Endometriosis is when tissue similar to the uterine lining grows outside the uterus, which can cause pain and other symptoms. It's often under-recognized, so awareness and early conversations with a doctor matter.",
    aspects: [
      { Icon: Flame, label: "Pelvic pain" },
      { Icon: Calendar, label: "Painful periods" },
      { Icon: Baby, label: "Possible fertility impact" },
      { Icon: Activity, label: "Chronic inflammation" },
    ],
    guidance: [
      { Icon: ClipboardList, title: "Track your symptoms", text: "Noting pain patterns and timing helps you and your doctor see the bigger picture." },
      { Icon: Stethoscope, title: "Talk to a specialist", text: "A gynecologist can properly evaluate symptoms — endometriosis is often under-diagnosed." },
      { Icon: Activity, title: "Manage pain", text: "Ask a professional about options that can help with day-to-day pain." },
      { Icon: Moon, title: "Rest & support", text: "Rest during flare-ups and lean on support networks when symptoms are tough." },
    ],
  },
};

const BAND = {
  Low: {
    label: "Lower likelihood",
    bar: "bg-soft-rose",
    describe: (c) =>
      `Your responses show relatively few of the patterns commonly associated with ${c}. That's reassuring — though it isn't a guarantee. If you have symptoms that worry you, it's still worth raising them with a doctor.`,
  },
  Moderate: {
    label: "Moderate likelihood",
    bar: "bg-pcos",
    describe: (c) =>
      `Your responses show some patterns that can be associated with ${c}. This doesn't mean you have it, but it's worth attention — consider talking with a healthcare provider, especially if symptoms persist or worsen.`,
  },
  High: {
    label: "Higher likelihood",
    bar: "bg-deep-rose",
    describe: (c) =>
      `Your responses show several patterns often associated with ${c}. This is not a diagnosis, but we'd encourage you to speak with a healthcare provider who can properly evaluate your symptoms.`,
  },
};

function IllustrationPlaceholder({ label, className = "" }) {
  return (
    <div className={`rounded-card border border-dashed border-divider bg-white/40 flex flex-col items-center justify-center text-body gap-2 ${className}`}>
      <ImageIcon size={26} className="opacity-40" />
      <span className="text-xs">{label}</span>
    </div>
  );
}

function SectionCard({ children, className = "" }) {
  return <div className={`rounded-card bg-white border border-divider p-6 sm:p-7 ${className}`}>{children}</div>;
}

export default function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const result = location.state?.result;
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState("");

  const handleDownload = async () => {
    if (!result) return;
    setPdfError("");
    try {
      setPdfLoading(true);
      await downloadReport({
        condition: result.condition,
        risk_level: result.risk_level,
        risk_percentage: result.risk_percentage,
        contributing_factors: result.contributing_factors || [],
        symptoms: {},
      });
    } catch (e) {
      setPdfError(
        String(e.message).startsWith("429")
          ? "You've downloaded a few times quickly — please wait a moment and try again."
          : "Couldn't generate the report. Make sure the backend is running and try again."
      );
    } finally {
      setPdfLoading(false);
    }
  };

  if (!result) {
    return (
      <div className="max-w-xl mx-auto px-6 py-20 text-center">
        <h1 className="text-2xl font-bold text-charcoal mb-3">No result to show</h1>
        <p className="text-body mb-6">
          It looks like you landed here directly, or the page was refreshed. Results aren't
          saved, so you'll need to take the assessment again to see them.
        </p>
        <Button onClick={() => navigate("/screening")}>Start a screening</Button>
      </div>
    );
  }

  const cond = CONDITION[result.condition] ?? CONDITION.pcos;
  const band = BAND[result.risk_level] ?? BAND.Moderate;
  const pct = Math.max(0, Math.min(100, result.risk_percentage ?? 0));

  return (
    <div>
      {/* Header */}
      <div className={`${cond.headerBg} border-b border-divider`}>
        <div className="max-w-6xl mx-auto px-6 py-8 flex items-center justify-between gap-8">
          <div>
            <Link to="/screening" className={`inline-flex items-center gap-2 text-sm font-medium mb-4 hover:opacity-80 ${cond.accent}`}>
              <ArrowLeft size={16} /> Back to screening
            </Link>
            <p className="text-sm text-body">{cond.full}</p>
            <h1 className="text-3xl font-bold text-charcoal">{cond.name} Assessment Results</h1>
            <p className="text-body mt-2 flex items-center gap-2 text-sm">
              <Info size={15} className={cond.accent} /> This is not a medical diagnosis.
            </p>
          </div>
          <img
            src="/illustrations/results-hero.svg"
            alt=""
            className="hidden lg:block w-64 h-40 object-contain shrink-0"
            onError={(e) => { e.currentTarget.style.display = "none"; }}
          />
        </div>
      </div>

      {/* Body */}
      <div className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {/* Result card */}
          <SectionCard>
            <span className={`inline-block text-sm font-medium px-3 py-1 rounded-pill mb-4 ${cond.chip}`}>{cond.name}</span>
            <h2 className="text-2xl font-heading font-semibold text-charcoal mb-1">{band.label}</h2>
            <p className="text-body mb-6">Estimated from your responses — not a diagnosis.</p>

            <div className="mb-2 flex items-center justify-between text-sm">
              <span className="text-body">Likelihood</span>
              <span className="font-semibold text-charcoal">{pct}%</span>
            </div>
            <div className="w-full h-3 rounded-pill bg-divider overflow-hidden">
              <div className={`h-full rounded-pill ${band.bar} transition-all`} style={{ width: `${pct}%` }} />
            </div>

            <div className={`mt-6 rounded-input ${cond.headerBg} border border-divider p-4 flex gap-3`}>
              <Info size={18} className={`${cond.accent} shrink-0 mt-0.5`} />
              <p className="text-sm text-body">{band.describe(cond.name)}</p>
            </div>
          </SectionCard>

          {/* What this means */}
          <SectionCard>
            <h3 className="text-lg font-heading font-semibold text-charcoal mb-2">What this means</h3>
            <p className="text-body mb-5">{cond.meaning}</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {cond.aspects.map((a) => (
                <div key={a.label} className="flex flex-col items-center text-center gap-2">
                  <span className={`w-11 h-11 rounded-full ${cond.iconBg} flex items-center justify-center`}>
                    <a.Icon size={18} className={cond.accent} />
                  </span>
                  <span className="text-xs text-charcoal">{a.label}</span>
                </div>
              ))}
            </div>
          </SectionCard>

          {/* General guidance */}
          <SectionCard>
            <h3 className="text-lg font-heading font-semibold text-charcoal mb-1">General guidance</h3>
            <p className="text-sm text-body mb-5">
              General information that may support your health — the same for everyone, not personalized to your answers.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {cond.guidance.map((g) => (
                <div key={g.title} className="flex gap-3">
                  <span className={`w-9 h-9 rounded-full ${cond.iconBg} flex items-center justify-center shrink-0`}>
                    <g.Icon size={16} className={cond.accent} />
                  </span>
                  <div>
                    <p className="text-sm font-medium text-charcoal">{g.title}</p>
                    <p className="text-xs text-body">{g.text}</p>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-body mt-5 pt-4 border-t border-divider">
              This is general guidance. Please consult a healthcare professional for advice specific to you.
            </p>
          </SectionCard>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-6">
          {/* Contributing factors */}
          <SectionCard>
            <div className="flex items-center gap-2 mb-1">
              <ClipboardList size={18} className={cond.accent} />
              <h3 className="text-lg font-heading font-semibold text-charcoal">Key contributing factors</h3>
            </div>
            <p className="text-sm text-body mb-4">Factors from your answers the model weighed most.</p>
            {result.contributing_factors?.length > 0 ? (
              <ul className="flex flex-col gap-3">
                {result.contributing_factors.map((f) => (
                  <li key={f} className="flex items-center gap-3 text-charcoal text-sm">
                    <span className={`w-2 h-2 rounded-full ${band.bar} shrink-0`} />
                    {f}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-body">No standout factors — your responses were fairly balanced.</p>
            )}
            <p className="text-xs text-body mt-4 pt-4 border-t border-divider">
              These are contributors, not causes or a diagnosis.
            </p>
          </SectionCard>

          {/* What's next */}
          <SectionCard>
            <h3 className="text-lg font-heading font-semibold text-charcoal mb-1">What's next?</h3>
            <p className="text-sm text-body mb-4">Save your result or ask questions about it.</p>
            <div className="flex flex-col gap-3">
              <button
                onClick={() => navigate("/assistant")}
                className="inline-flex items-center justify-center gap-2 rounded-pill bg-ai text-white px-5 py-3 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer"
              >
                <MessageCircle size={16} /> Chat with AI Assistant
              </button>
              <button
                onClick={handleDownload}
                disabled={pdfLoading}
                className="inline-flex items-center justify-center gap-2 rounded-pill border border-divider text-charcoal px-5 py-3 text-sm font-medium hover:border-pcos hover:text-pcos transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Download size={16} /> {pdfLoading ? "Preparing report..." : "Download Report (PDF)"}
              </button>
              {pdfError && <p className="text-xs text-deep-rose mt-1">{pdfError}</p>}
            </div>
          </SectionCard>

          {/* Working CTAs */}
          <div className="flex flex-col gap-3">
            <Button variant="secondary" className="gap-2" onClick={() => navigate(`/questionnaire/${result.condition}`)}>
              <RefreshCw size={16} /> Retake assessment
            </Button>
            <Button onClick={() => navigate(`/questionnaire/${cond.other}`)}>
              Screen for {cond.otherName}
            </Button>
          </div>
        </div>
      </div>

      {/* Disclaimer strip */}
      <div className="bg-blush border-t border-divider">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center gap-2 justify-center text-center">
          <Info size={15} className="text-pcos shrink-0" />
          <p className="text-xs text-body">
            This assessment provides risk screening and educational information only, and is not a
            substitute for professional medical advice, diagnosis, or treatment.
          </p>
        </div>
      </div>
    </div>
  );
}