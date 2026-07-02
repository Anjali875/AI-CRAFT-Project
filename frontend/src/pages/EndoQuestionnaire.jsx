import { useState, useMemo, useRef, useEffect, Fragment } from "react";
import { useNavigate } from "react-router-dom";
import {
  User, Calendar, Activity, Users, Droplets, Brain,
  Info, ShieldCheck, TriangleAlert, ArrowLeft,
} from "lucide-react";
import { predictEndo } from "../api";
import Button from "../components/Button";
import Card from "../components/Card";

const SECTIONS = [
  { key: "personal", label: "Personal" },
  { key: "cycle", label: "Cycle" },
  { key: "scores", label: "Symptoms" },
  { key: "history", label: "History" },
];

function SectionHeader({ Icon, title, subtitle }) {
  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="w-8 h-8 rounded-full bg-endo-light flex items-center justify-center">
          <Icon size={16} className="text-endo" />
        </span>
        <h2 className="text-lg font-heading font-semibold text-charcoal">{title}</h2>
      </div>
      {subtitle && <p className="text-sm text-body mt-1">{subtitle}</p>}
    </div>
  );
}

function NumberField({ label, hint, value, onChange, placeholder, suffix }) {
  return (
    <div>
      <label className="block text-sm font-medium text-charcoal mb-1">{label}</label>
      {hint && <p className="text-xs text-body mb-2">{hint}</p>}
      <div className="relative">
        <input
          type="number"
          inputMode="decimal"
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-input border border-divider px-4 py-3 pr-16 text-charcoal focus:outline-none focus:border-endo"
        />
        {suffix && (
          <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-body pointer-events-none">
            {suffix}
          </span>
        )}
      </div>
    </div>
  );
}

function YesNo({ Icon, label, value, onChange }) {
  return (
    <div className="rounded-input border border-divider p-4 flex items-center gap-3">
      <span className="w-10 h-10 rounded-full bg-endo-light flex items-center justify-center shrink-0">
        <Icon size={18} className="text-endo" />
      </span>
      <span className="text-sm font-medium text-charcoal flex-1">{label}</span>
      <div className="flex gap-2 shrink-0">
        {[{ t: "Yes", v: 1 }, { t: "No", v: 0 }].map(({ t, v }) => (
          <button
            key={t}
            type="button"
            onClick={() => onChange(v)}
            className={`px-4 py-1.5 rounded-pill text-sm font-medium border transition-colors cursor-pointer ${
              value === v
                ? "bg-endo text-white border-endo"
                : "bg-white text-charcoal border-divider hover:border-endo"
            }`}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}

function ScoreSlider({ label, hint, min, max, value, onChange, lowLabel, highLabel }) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <label className="text-sm font-medium text-charcoal">{label}</label>
        <span className="text-lg font-semibold text-endo">{value}</span>
      </div>
      {hint && <p className="text-xs text-body mb-2">{hint}</p>}
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value, 10))}
        className="w-full accent-endo cursor-pointer"
      />
      <div className="flex justify-between text-xs text-body mt-1">
        <span>{lowLabel}</span>
        <span>{highLabel}</span>
      </div>
    </div>
  );
}

export default function EndoQuestionnaire() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    age: "", weight: "", height: "", cycle_length: "", age_of_menarche: "",
    family_history: 0, infertility_status: 0,
  });
  const [dysmenorrhea, setDysmenorrhea] = useState(0);
  const [urinary, setUrinary] = useState(0);
  const [mentalHealth, setMentalHealth] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeSection, setActiveSection] = useState("personal");

  const refs = {
    personal: useRef(null),
    cycle: useRef(null),
    scores: useRef(null),
    history: useRef(null),
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) setActiveSection(entry.target.dataset.section);
        });
      },
      { rootMargin: "-40% 0px -55% 0px" }
    );
    Object.values(refs).forEach((r) => r.current && observer.observe(r.current));
    return () => observer.disconnect();
  }, []);

  const scrollTo = (key) =>
    refs[key].current?.scrollIntoView({ behavior: "smooth", block: "start" });

  const bmi = useMemo(() => {
    const w = parseFloat(form.weight);
    const h = parseFloat(form.height);
    if (!w || !h) return null;
    const m = h / 100;
    return +(w / (m * m)).toFixed(1);
  }, [form.weight, form.height]);

  const set = (key, val) => setForm((f) => ({ ...f, [key]: val }));

  const validate = () => {
    const age = parseFloat(form.age);
    const weight = parseFloat(form.weight);
    const height = parseFloat(form.height);
    const cycle = parseFloat(form.cycle_length);
    const menarche = parseFloat(form.age_of_menarche);
    if (Number.isNaN(age) || age < 15 || age > 55) return "Enter an age between 15 and 55.";
    if (Number.isNaN(weight) || weight < 25 || weight > 200) return "Enter a weight between 25 and 200 kg.";
    if (Number.isNaN(height) || height < 100 || height > 220) return "Enter a height between 100 and 220 cm.";
    if (bmi === null || bmi < 10 || bmi > 60) return "The calculated BMI is out of range — check your weight and height.";
    if (Number.isNaN(cycle) || cycle < 15 || cycle > 45) return "Enter a cycle length between 15 and 45 days.";
    if (Number.isNaN(menarche) || menarche < 8 || menarche > 18) return "Enter an age of first period between 8 and 18.";
    return "";
  };

  const handleSubmit = async () => {
    setError("");
    setResult(null);
    const v = validate();
    if (v) {
      setError(v);
      scrollTo("personal");
      return;
    }
    const payload = {
      age: parseFloat(form.age),
      weight: parseFloat(form.weight),
      height: parseFloat(form.height),
      bmi,
      cycle_length: parseFloat(form.cycle_length),
      age_of_menarche: parseFloat(form.age_of_menarche),
      dysmenorrhea_score: dysmenorrhea,
      urinary_symptoms_score: urinary,
      family_history: form.family_history,
      infertility_status: form.infertility_status,
      mental_health_score: mentalHealth,
    };
    try {
      setLoading(true);
      const res = await predictEndo(payload);
      navigate("/results", { state: { result: res } });
    } catch (e) {
      setError(
        `Couldn't reach the server. Make sure your backend is running on port 8000 and that you opened this site as localhost:5173, not 127.0.0.1. (${e.message})`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="bg-endo-light border-b border-divider">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <button
            onClick={() => navigate("/screening")}
            className="inline-flex items-center gap-2 text-endo text-sm font-medium mb-4 hover:opacity-80 cursor-pointer"
          >
            <ArrowLeft size={16} /> Back
          </button>
          <h1 className="text-3xl font-bold text-charcoal">Endometriosis Risk Assessment</h1>
          <p className="text-body mt-1 max-w-xl">
            Please answer honestly. Your information is private and used only for this
            assessment — this is not a medical diagnosis.
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-8">
        <div className="min-w-0">
          <div className="mb-8">
            <div className="flex items-center">
              {SECTIONS.map((s, i) => (
                <Fragment key={s.key}>
                  <button
                    type="button"
                    onClick={() => scrollTo(s.key)}
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold border-2 transition-colors cursor-pointer ${
                      activeSection === s.key
                        ? "bg-endo text-white border-endo"
                        : "bg-white text-body border-divider"
                    }`}
                  >
                    {i + 1}
                  </button>
                  {i < SECTIONS.length - 1 && <div className="flex-1 h-0.5 bg-divider mx-2" />}
                </Fragment>
              ))}
            </div>
            <div className="flex justify-between mt-2">
              {SECTIONS.map((s) => (
                <span
                  key={s.key}
                  className={`text-xs font-medium ${
                    activeSection === s.key ? "text-endo" : "text-body"
                  }`}
                >
                  {s.label}
                </span>
              ))}
            </div>
          </div>

          <div className="rounded-card bg-white border border-divider p-6 sm:p-8 flex flex-col gap-8">
            <section ref={refs.personal} data-section="personal" className="scroll-mt-24">
              <SectionHeader Icon={User} title="Personal Information" subtitle="Tell us a bit about yourself." />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                <NumberField label="Age" value={form.age} onChange={(v) => set("age", v)} placeholder="e.g. 28" suffix="years" />
                <NumberField label="Weight" value={form.weight} onChange={(v) => set("weight", v)} placeholder="e.g. 62" suffix="kg" />
                <NumberField label="Height" value={form.height} onChange={(v) => set("height", v)} placeholder="e.g. 165" suffix="cm" />
                <div>
                  <label className="block text-sm font-medium text-charcoal mb-1">BMI</label>
                  <p className="text-xs text-body mb-2">Auto-calculated from weight and height.</p>
                  <div className="relative">
                    <div className="w-full rounded-input border border-divider px-4 py-3 pr-16 bg-endo-light text-charcoal">
                      {bmi ?? "—"}
                    </div>
                    <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-body">kg/m²</span>
                  </div>
                </div>
              </div>
            </section>

            <section ref={refs.cycle} data-section="cycle" className="scroll-mt-24 border-t border-divider pt-8">
              <SectionHeader Icon={Calendar} title="Menstrual History" />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                <NumberField
                  label="Average cycle length"
                  hint="Days from the first day of one period to the first day of the next (typically 21–35)."
                  value={form.cycle_length}
                  onChange={(v) => set("cycle_length", v)}
                  placeholder="e.g. 28"
                  suffix="days"
                />
                <NumberField
                  label="Age of your first period"
                  hint="How old you were when menstruation began."
                  value={form.age_of_menarche}
                  onChange={(v) => set("age_of_menarche", v)}
                  placeholder="e.g. 13"
                  suffix="years"
                />
              </div>
            </section>

            <section ref={refs.scores} data-section="scores" className="scroll-mt-24 border-t border-divider pt-8">
              <SectionHeader Icon={Activity} title="Symptom Severity" subtitle="Slide to rate each on the given scale." />
              <div className="flex flex-col gap-7 mt-5">
                <ScoreSlider
                  label="Menstrual pain (period cramps)"
                  hint="How severe is your period pain, typically?"
                  min={0} max={10} value={dysmenorrhea} onChange={setDysmenorrhea}
                  lowLabel="No pain (0)" highLabel="Severe (10)"
                />
                <ScoreSlider
                  label="Urinary symptoms"
                  hint="Discomfort, urgency, or pain when urinating, especially around your period."
                  min={0} max={9} value={urinary} onChange={setUrinary}
                  lowLabel="None (0)" highLabel="Severe (9)"
                />
                <ScoreSlider
                  label="Impact on mental wellbeing"
                  hint="How much have symptoms affected your mood, anxiety, or daily life recently?"
                  min={0} max={10} value={mentalHealth} onChange={setMentalHealth}
                  lowLabel="Not at all (0)" highLabel="Severely (10)"
                />
              </div>
            </section>

            <section ref={refs.history} data-section="history" className="scroll-mt-24 border-t border-divider pt-8">
              <SectionHeader Icon={Users} title="Medical History" subtitle="A few yes/no questions." />
              <div className="flex flex-col gap-3 mt-4">
                <YesNo Icon={Users} label="Family history of endometriosis (mother or sister)" value={form.family_history} onChange={(v) => set("family_history", v)} />
                <YesNo Icon={Brain} label="Difficulty conceiving / diagnosed infertility" value={form.infertility_status} onChange={(v) => set("infertility_status", v)} />
              </div>
            </section>

            {error && (
              <div className="rounded-input bg-endo-light border border-endo/40 text-endo text-sm px-4 py-3">
                {error}
              </div>
            )}

            <div className="flex items-center justify-between gap-4 border-t border-divider pt-6">
              <Button variant="secondary" onClick={() => navigate("/screening")}>Back</Button>
              <Button onClick={handleSubmit} className={loading ? "opacity-60" : ""} disabled={loading}>
                {loading ? "Analyzing..." : "Get My Result"}
              </Button>
            </div>
          </div>

          {result && (
            <Card variant="endo" className="mt-8">
              <p className="text-sm text-body mb-1">Temporary result preview (the real Results page comes next phase)</p>
              <h2 className="text-2xl font-semibold text-endo mb-2">
                {result.risk_level} likelihood — {result.risk_percentage}%
              </h2>
              {result.contributing_factors?.length > 0 && (
                <div className="mt-3">
                  <p className="text-sm font-medium text-charcoal mb-1">Contributing factors:</p>
                  <ul className="list-disc list-inside text-body text-sm">
                    {result.contributing_factors.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </Card>
          )}
        </div>

        <aside className="lg:sticky lg:top-24 h-fit flex flex-col gap-4">
          <div className="rounded-card bg-white border border-divider p-5">
            <div className="flex items-center gap-2 mb-2">
              <Info size={18} className="text-endo" />
              <h3 className="font-heading font-semibold text-charcoal">Why We Ask This</h3>
            </div>
            <p className="text-sm text-body">
              Endometriosis is often missed for years. These questions help the model spot
              patterns linked to it and estimate your likelihood.
            </p>
          </div>
          <div className="rounded-card bg-endo-light border border-endo/30 p-5">
            <div className="flex items-center gap-2 mb-2">
              <ShieldCheck size={18} className="text-endo" />
              <h3 className="font-heading font-semibold text-charcoal">Your Privacy Matters</h3>
            </div>
            <p className="text-sm text-body">
              Your answers are used only to calculate your result. Nothing is stored or shared.
            </p>
          </div>
          <div className="rounded-card bg-white border border-divider p-5">
            <div className="flex items-center gap-2 mb-2">
              <TriangleAlert size={18} className="text-endo" />
              <h3 className="font-heading font-semibold text-charcoal">Important</h3>
            </div>
            <p className="text-sm text-body">
              This is a screening tool, not a medical diagnosis. Please consult a healthcare
              professional about any concerns.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}