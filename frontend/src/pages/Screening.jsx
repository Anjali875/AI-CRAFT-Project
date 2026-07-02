import { useNavigate } from "react-router-dom";
import { Activity, Flower2, ArrowRight } from "lucide-react";
import Card from "../components/Card";

export default function Screening() {
  const navigate = useNavigate();

  const options = [
    {
      condition: "pcos",
      variant: "pcos",
      Icon: Activity,
      title: "PCOS Risk Assessment",
      description:
        "Assess your likelihood of Polycystic Ovary Syndrome based on symptoms and health factors.",
      accent: "text-pcos",
      iconBg: "bg-soft-rose",
    },
    {
      condition: "endo",
      variant: "endo",
      Icon: Flower2,
      title: "Endometriosis Risk Assessment",
      description:
        "Analyze your likelihood of Endometriosis based on symptoms, pain scores, and history.",
      accent: "text-endo",
      iconBg: "bg-endo-light",
    },
  ];

  return (
    <div className="max-w-5xl mx-auto px-6 py-16">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-charcoal mb-3">
          Choose Your Screening
        </h1>
        <p className="text-body max-w-xl mx-auto">
          Select the condition you'd like to screen for. Each assessment takes only
          a few minutes and gives you a personalized likelihood result.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {options.map((opt) => (
          <Card
            key={opt.condition}
            variant={opt.variant}
            className="cursor-pointer hover:-translate-y-1 transition-transform flex flex-col"
            onClick={() => navigate(`/questionnaire/${opt.condition}`)}
          >
            <div className={`w-14 h-14 rounded-full ${opt.iconBg} flex items-center justify-center mb-5`}>
              <opt.Icon size={26} className={opt.accent} />
            </div>
            <h2 className={`text-2xl font-semibold ${opt.accent} mb-2`}>
              {opt.title}
            </h2>
            <p className="text-body mb-6 flex-1">{opt.description}</p>
            <span className={`inline-flex items-center gap-2 font-medium ${opt.accent}`}>
              Start assessment <ArrowRight size={18} />
            </span>
          </Card>
        ))}
      </div>

      <p className="text-center text-sm text-body mt-10">
        This tool provides risk screening and educational information only — it is
        not a medical diagnosis.
      </p>
    </div>
  );
}