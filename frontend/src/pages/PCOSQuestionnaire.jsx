import { useState, useMemo, useRef, useEffect, Fragment } from "react";
import { useNavigate } from "react-router-dom";
import {
  User, Calendar, Activity, Dumbbell, TrendingUp, Sparkles, Sun, Scissors,
  Droplet, Info, ShieldCheck, TriangleAlert, ArrowLeft,
} from "lucide-react";
import { predictPCOS } from "../api";
import Button from "../components/Button";
import Card from "../components/Card";

const SECTIONS = [
  { key: "personal", label: "Personal Info" },
  { key: "cycle", label: "Your Cycle" },
  { key: "symptoms", label: "Symptoms" },
  { key: "lifestyle", label: "Lifestyle" },
];

const SYMPTOMS = [
  { key: "weight_gain", label: "Unexplained weight gain", Icon: TrendingUp },
  { key: "hair_growth", label: "Excess hair growth (face, chin, body)", Icon: Sparkles },
  { key: "skin_darkening", label: "Skin darkening (neck, underarms, groin)", Icon: Sun },
  { key: "hair_loss", label: "Hair loss or scalp thinning", Icon: Scissors },
  { key: "pimples", label: "Frequent acne or pimples", Icon: Droplet },
];

const FAST_FOOD_OPTIONS = [
  { label: "Rarely / never", value: 0 },
  { label: "Few times a month", value: 0 },
  { label: "Few times a week", value: 1 },
  { label: "Almost daily", value: 1 },
];

const EXERCISE_OPTIONS = [
  { label: "Rarely / never", value: 0 },
  { label: "1–2 / week", value: 0 },
  { label: "3–4 / week", value: 1 },
  { label: "5+ / week", value: 1 },
];

function SectionHeader({ Icon, title, subtitle }) {
  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="w-8 h-8 rounded-full bg-soft-rose/40 flex items-center justify-center">
          <Icon size={16} className="text-pcos" />
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
          className="w-full rounded-input border border-divider px-4 py-3 pr-14 text-charcoal focus:outline-none focus:border-pcos"
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

function SymptomCard({ Icon, label, value, onChange }) {
  return (
    <div className="rounded-input border border-divider p-4 flex items-center gap-3">
      <span className="w-10 h-10 rounded-full bg-soft-rose/40 flex items-center justify-center shrink-0">
        <Icon size={18} className="text-pcos" />
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
                ? "bg-pcos text-white border-pcos"
                : "bg-white text-charcoal border-divider hover:border-pcos"
            }`}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}

function FrequencyField({ label, options, index, onChange }) {
  return (
    <div>
      <p className="text-sm font-medium text-charcoal mb-2">{label}</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {options.map((opt, i) => (
          <button
            key={i}
            type="button"
            onClick={() => onChange(i)}
            className={`px-3 py-2 rounded-input border text-sm font-medium transition-colors cursor-pointer ${
              index === i
                ? "bg-pcos text-white border-pcos"
                : "bg-white text-charcoal border-divider hover:border-pcos"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function PCOSQuestionnaire() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    age: "", weight: "", height: "", cycle_length: "",
    weight_gain: 0, hair_growth: 0, skin_darkening: 0, hair_loss: 0, pimples: 0,
  });
  const [fastFoodIdx, setFastFoodIdx] = useState(0);
  const [exerciseIdx, setExerciseIdx] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeSection, setActiveSection] = useState("personal");

  const refs = {
    personal: useRef(null),
    cycle: useRef(null),
    symptoms: useRef(null),
    lifestyle: useRef(null),
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
    if (Number.isNaN(age) || age < 12 || age > 60) return "Enter an age between 12 and 60.";
    if (Number.isNaN(weight) || weight < 25 || weight > 200) return "Enter a weight between 25 and 200 kg.";
    if (Number.isNaN(height) || height < 100 || height > 220) return "Enter a height between 100 and 220 cm.";
    if (Number.isNaN(cycle) || cycle < 2 || cycle > 12) return "Enter a period duration between 2 and 12 days.";
    if (bmi === null || bmi < 10 || bmi > 60) return "The calculated BMI is out of range — check your weight and height.";
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
      weight_gain: form.weight_gain,
      hair_growth: form.hair_growth,
      skin_darkening: form.skin_darkening,
      hair_loss: form.hair_loss,
      pimples: form.pimples,
      fast_food: FAST_FOOD_OPTIONS[fastFoodIdx].value,
      regular_exercise: EXERCISE_OPTIONS[exerciseIdx].value,
    };
    try {
      setLoading(true);
      const res = await predictPCOS(payload);
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
      {/* Header */}
      <div className="bg-blush border-b border-divider">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <button
            onClick={() => navigate("/screening")}
            className="inline-flex items-center gap-2 text-pcos text-sm font-medium mb-4 hover:opacity-80 cursor-pointer"
          >
            <ArrowLeft size={16} /> Back
          </button>
          <h1 className="text-3xl font-bold text-charcoal">PCOS Risk Assessment</h1>
          <p className="text-body mt-1 max-w-xl">
            Please answer honestly. Your information is private and used only for this
            assessment — this is not a medical diagnosis.
          </p>
        </div>
      </div>

      {/* Two-panel body */}
      <div className="max-w-6xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-8">
        {/* Main column */}
        <div className="min-w-0">
          {/* Section progress bar */}
          <div className="mb-8">
            <div className="flex items-center">
              {SECTIONS.map((s, i) => (
                <Fragment key={s.key}>
                  <button
                    type="button"
                    onClick={() => scrollTo(s.key)}
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold border-2 transition-colors cursor-pointer ${
                      activeSection === s.key
                        ? "bg-pcos text-white border-pcos"
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
                    activeSection === s.key ? "text-pcos" : "text-body"
                  }`}
                >
                  {s.label}
                </span>
              ))}
            </div>
          </div>

          {/* Form card */}
          <div className="rounded-card bg-white border border-divider p-6 sm:p-8 flex flex-col gap-8">
            <section ref={refs.personal} data-section="personal" className="scroll-mt-24">
              <SectionHeader Icon={User} title="Personal Information" subtitle="Tell us a bit about yourself." />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
                <NumberField label="Age" value={form.age} onChange={(v) => set("age", v)} placeholder="e.g. 24" suffix="years" />
                <NumberField label="Weight" value={form.weight} onChange={(v) => set("weight", v)} placeholder="e.g. 62" suffix="kg" />
                <NumberField label="Height" value={form.height} onChange={(v) => set("height", v)} placeholder="e.g. 165" suffix="cm" />
                <div>
                  <label className="block text-sm font-medium text-charcoal mb-1">BMI</label>
                  <p className="text-xs text-body mb-2">Auto-calculated from weight and height.</p>
                  <div className="relative">
                    <div className="w-full rounded-input border border-divider px-4 py-3 pr-16 bg-blush text-charcoal">
                      {bmi ?? "—"}
                    </div>
                    <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-body">kg/m²</span>
                  </div>
                </div>
              </div>
            </section>

            <section ref={refs.cycle} data-section="cycle" className="scroll-mt-24 border-t border-divider pt-8">
              <SectionHeader Icon={Calendar} title="Your Cycle" />
              <div className="mt-4">
                <NumberField
                  label="How many days does your period usually last?"
                  hint="The number of days you bleed — not your full cycle length. (2–12 days)"
                  value={form.cycle_length}
                  onChange={(v) => set("cycle_length", v)}
                  placeholder="e.g. 5"
                  suffix="days"
                />
              </div>
            </section>

            <section ref={refs.symptoms} data-section="symptoms" className="scroll-mt-24 border-t border-divider pt-8">
              <SectionHeader Icon={Activity} title="Symptoms" subtitle="Indicate if you experience any of the following." />
              <div className="flex flex-col gap-3 mt-4">
                {SYMPTOMS.map((s) => (
                  <SymptomCard key={s.key} Icon={s.Icon} label={s.label} value={form[s.key]} onChange={(v) => set(s.key, v)} />
                ))}
              </div>
            </section>

            <section ref={refs.lifestyle} data-section="lifestyle" className="scroll-mt-24 border-t border-divider pt-8">
              <SectionHeader Icon={Dumbbell} title="Lifestyle" subtitle="Help us understand your daily habits." />
              <div className="flex flex-col gap-5 mt-4">
                <FrequencyField label="How often do you eat fast food?" options={FAST_FOOD_OPTIONS} index={fastFoodIdx} onChange={setFastFoodIdx} />
                <FrequencyField label="How often do you exercise?" options={EXERCISE_OPTIONS} index={exerciseIdx} onChange={setExerciseIdx} />
              </div>
            </section>

            {error && (
              <div className="rounded-input bg-blush border border-soft-rose text-deep-rose text-sm px-4 py-3">
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
            <Card variant="pcos" className="mt-8">
              <p className="text-sm text-body mb-1">Temporary result preview (the real Results page comes next phase)</p>
              <h2 className="text-2xl font-semibold text-pcos mb-2">
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

        {/* Sidebar */}
        <aside className="lg:sticky lg:top-24 h-fit flex flex-col gap-4">
          <div className="rounded-card bg-white border border-divider p-5">
            <div className="flex items-center gap-2 mb-2">
              <Info size={18} className="text-pcos" />
              <h3 className="font-heading font-semibold text-charcoal">Why We Ask This</h3>
            </div>
            <p className="text-sm text-body">
              These questions help the model recognize patterns linked to PCOS and give you a
              more accurate likelihood estimate.
            </p>
          </div>
          <div className="rounded-card bg-blush border border-soft-rose p-5">
            <div className="flex items-center gap-2 mb-2">
              <ShieldCheck size={18} className="text-pcos" />
              <h3 className="font-heading font-semibold text-charcoal">Your Privacy Matters</h3>
            </div>
            <p className="text-sm text-body">
              Your answers are used only to calculate your result. Nothing is stored or shared.
            </p>
          </div>
          <div className="rounded-card bg-ai-light border border-ai/30 p-5">
            <div className="flex items-center gap-2 mb-2">
              <TriangleAlert size={18} className="text-ai" />
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