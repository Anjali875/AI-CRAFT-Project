import { useNavigate } from "react-router-dom";
import {
  Activity,
  Flower2,
  Bot,
  Brain,
  BookOpen,
  Lock,
  ShieldCheck,
  MessageCircle,
} from "lucide-react";
import Card from "../components/Card";

export default function Home() {
  const navigate = useNavigate();

  const infoCards = [
    {
      variant: "pcos",
      Icon: Activity,
      accent: "text-pcos",
      iconBg: "bg-soft-rose",
      title: "PCOS Risk Assessment",
      description:
        "Assess your likelihood of PCOS based on your symptoms and health factors.",
    },
    {
      variant: "endo",
      Icon: Flower2,
      accent: "text-endo",
      iconBg: "bg-endo-light",
      title: "Endometriosis Risk Assessment",
      description:
        "Analyze your likelihood of Endometriosis using symptoms, pain scores, and history.",
    },
    {
      variant: "ai",
      Icon: Bot,
      accent: "text-ai",
      iconBg: "bg-ai-light",
      title: "AI Health Assistant",
      description:
        "Ask questions, understand your results, and get supportive, informed guidance.",
    },
  ];

  const trust = [
    {
      Icon: Brain,
      title: "AI-Powered Insights",
      description:
        "Machine learning models surface likelihood signals from your answers.",
    },
    {
      Icon: BookOpen,
      title: "Evidence-Based Information",
      description:
        "Educational content grounded in established medical understanding.",
    },
    {
      Icon: Lock,
      title: "Privacy First",
      description:
        "No account required, and your screening answers aren't stored.",
    },
    {
      Icon: MessageCircle,
      title: "Supportive Guidance",
      description:
        "An AI assistant on hand to help you make sense of your results.",
    },
  ];

  return (
    <div>
      {/* Hero */}
      <section className="bg-blush">
        <div className="max-w-6xl mx-auto px-6 py-16 md:py-20 grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-charcoal leading-tight mb-4">
              Empowering Women's Health{" "}
              <span className="text-pcos">Through AI</span>
            </h1>
            <p className="text-body/80 text-lg mb-8 max-w-md">
              AI-powered likelihood screening for PCOS and Endometriosis, with
              personalized insights and intelligent support for your health
              journey.
            </p>
            <button
              onClick={() => navigate("/screening")}
              className="inline-flex items-center gap-2 rounded-pill bg-pcos text-white px-7 py-3 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer"
            >
              Start Screening
            </button>
            <div className="flex items-center gap-2 mt-5 text-sm text-body/70">
              <ShieldCheck size={16} className="text-pcos" />
              No account needed · Not a diagnostic tool
            </div>
          </div>

          {/* Illustration placeholder */}
          <div className="rounded-card aspect-[4/3] flex items-center justify-center p-4">
            <img
              src="/illustrations/home-hero.svg"
              alt=""
              className="w-full h-full object-contain"
              onError={(e) => { e.currentTarget.style.display = "none"; }}
            />
          </div>
        </div>
      </section>

      {/* What the platform offers — informational, non-clickable */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {infoCards.map((c) => (
            <Card key={c.title} variant={c.variant} className="flex flex-col">
              <div
                className={`w-14 h-14 rounded-full ${c.iconBg} flex items-center justify-center mb-5`}
              >
                <c.Icon size={26} className={c.accent} />
              </div>
              <h3 className={`text-xl font-semibold ${c.accent} mb-2`}>
                {c.title}
              </h3>
              <p className="text-body/80 text-sm">{c.description}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* Why Trust Our Platform */}
      <section className="bg-blush">
        <div className="max-w-6xl mx-auto px-6 py-16">
          <h2 className="text-2xl md:text-3xl font-bold text-charcoal text-center mb-12">
            Why Trust Our Platform?
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            {trust.map((t) => (
              <div key={t.title} className="flex flex-col items-start">
                <div className="w-12 h-12 rounded-full bg-soft-rose flex items-center justify-center mb-4">
                  <t.Icon size={22} className="text-pcos" />
                </div>
                <h3 className="text-base font-semibold text-charcoal mb-1">
                  {t.title}
                </h3>
                <p className="text-body/80 text-sm">{t.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Better Awareness — links to Health Library */}
      <section className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-2xl md:text-3xl font-bold text-charcoal text-center mb-12">
          Better Awareness, Better Health
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card variant="default" className="flex flex-col md:flex-row gap-5">
            <div className="rounded-input aspect-video md:w-40 md:aspect-square flex items-center justify-center shrink-0 p-2">
              <img
                src="/illustrations/home-pcos.svg"
                alt=""
                className="w-full h-full object-contain"
                onError={(e) => { e.currentTarget.style.display = "none"; }}
              />
            </div>
            <div className="flex flex-col">
              <h3 className="text-lg font-semibold text-pcos mb-2">
                What is PCOS?
              </h3>
              <p className="text-body/80 text-sm mb-4 flex-1">
                Polycystic Ovary Syndrome is a common hormonal condition among
                women of reproductive age. Early awareness can help you manage
                symptoms and quality of life.
              </p>
              <button
                onClick={() => navigate("/health-library#pcos")}
                className="inline-flex items-center justify-center rounded-pill bg-pcos text-white px-5 py-2 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer self-start"
              >
                Learn More
              </button>
            </div>
          </Card>

          <Card variant="endo" className="flex flex-col md:flex-row gap-5">
            <div className="rounded-input aspect-video md:w-40 md:aspect-square flex items-center justify-center shrink-0 p-2">
              <img
                src="/illustrations/home-endo.svg"
                alt=""
                className="w-full h-full object-contain"
                onError={(e) => { e.currentTarget.style.display = "none"; }}
              />
            </div>
            <div className="flex flex-col">
              <h3 className="text-lg font-semibold text-endo mb-2">
                What is Endometriosis?
              </h3>
              <p className="text-body/80 text-sm mb-4 flex-1">
                Endometriosis occurs when tissue similar to the uterine lining
                grows outside the uterus, causing pain and other symptoms. Early
                awareness matters.
              </p>
              <button
                onClick={() => navigate("/health-library#endo")}
                className="inline-flex items-center justify-center rounded-pill bg-endo text-white px-5 py-2 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer self-start"
              >
                Learn More
              </button>
            </div>
          </Card>
        </div>
      </section>
    </div>
  );
}