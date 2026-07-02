import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { Activity, Flower2 } from "lucide-react";
import Card from "../components/Card";

export default function HealthLibrary() {
  const { hash } = useLocation();

  // React Router doesn't scroll to #anchors on its own — do it manually.
  useEffect(() => {
    if (hash) {
      const el = document.getElementById(hash.replace("#", ""));
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [hash]);

  const Bullets = ({ items, color = "bg-pcos" }) => (
    <ul className="space-y-2">
      {items.map((t, i) => (
        <li key={i} className="flex gap-3 text-sm text-body/85">
          <span className={`mt-2 w-1.5 h-1.5 rounded-full ${color} shrink-0`} />
          <span>{t}</span>
        </li>
      ))}
    </ul>
  );

  const pcosSymptoms = [
    "Irregular, unpredictable, or absent periods — or heavy, long, or painful periods",
    "Difficulty conceiving or infertility",
    "Excessive hair growth on the face or body (hirsutism)",
    "Female-pattern hair thinning or baldness",
    "Acne or oily skin",
  ];
  const pcosRisks = [
    "Insulin resistance and type 2 diabetes",
    "Gestational diabetes or high blood pressure in pregnancy",
    "Weight gain, especially around the abdomen",
    "High blood pressure and high cholesterol",
    "Cardiovascular disease",
    "Sleep apnea",
    "Endometrial hyperplasia or endometrial cancer",
  ];
  const endoSymptoms = [
    "Pelvic pain far beyond normal cramping, during or outside periods",
    "Cramps that begin before and extend after a period",
    "Lower back or abdominal pain",
    "Pain during or after sex (dyspareunia)",
    "Pain with bowel movements or urination",
    "Heavy menstrual bleeding or bleeding between periods",
    "Infertility",
    "Fatigue, constipation, bloating, or nausea, especially during periods",
  ];
  const endoRiskFactors = [
    "Never having given birth",
    "Menstrual cycles more frequent than every 28 days",
    "Heavy, prolonged periods lasting longer than seven days",
    "Higher levels of estrogen in the body",
    "Low body mass index",
    "Family history of endometriosis",
    "Starting periods early or reaching menopause later",
  ];

  const comparison = [
    {
      aspect: "Nature",
      pcos: "A hormonal disorder affecting ovulation and metabolism.",
      endo: "A chronic inflammatory disease where endometrial-like tissue grows outside the uterus.",
    },
    {
      aspect: "Key hormonal issue",
      pcos: "Excess androgens (male hormones).",
      endo: "Abnormally high levels of estrogen.",
    },
    {
      aspect: "Primary symptoms",
      pcos: "Irregular or missed periods, excess hair growth, acne, weight gain.",
      endo: "Severe pelvic pain, painful periods, pain during sex, pain with bowel/bladder function.",
    },
    {
      aspect: "Key feature",
      pcos: "Hormonal and metabolic imbalance; ovaries may show multiple cysts.",
      endo: "Inflammation, scarring, and pain from misplaced tissue; may cause endometriomas.",
    },
    {
      aspect: "Diagnosis",
      pcos: "Based on symptoms, hormone levels, and ultrasound.",
      endo: "Often needs laparoscopy to confirm, though ultrasound/MRI can help.",
    },
  ];

  const myths = [
    {
      myth: "Only overweight women get PCOS.",
      fact: "PCOS affects women of all body types. Lean PCOS is real and well-documented.",
    },
    {
      myth: "You can't get pregnant with PCOS or endometriosis.",
      fact: "Many women with either condition conceive, naturally or with medical help. Neither is an absolute barrier to fertility.",
    },
    {
      myth: "Endometriosis is just painful periods.",
      fact: "Its pain is far more severe than normal cramps and can include chronic pelvic pain, pain during sex, and bowel/bladder issues.",
    },
    {
      myth: "PCOS and endometriosis are the same thing.",
      fact: "They are distinct conditions with different causes, mechanisms, and treatments.",
    },
    {
      myth: "A hysterectomy cures endometriosis.",
      fact: "It isn't a guaranteed cure, especially if endometriosis tissue remains outside the uterus.",
    },
  ];

  const references = [
    { label: "World Health Organization — Polycystic ovary syndrome (WHO Fact Sheets)", url: "https://www.who.int/news-room/fact-sheets/detail/polycystic-ovary-syndrome" },
    { label: "World Health Organization — Endometriosis (WHO Fact Sheets)", url: "https://www.who.int/news-room/fact-sheets/detail/endometriosis" },
    { label: "Mayo Clinic — Endometriosis: Symptoms and causes", url: "https://www.mayoclinic.org/diseases-conditions/endometriosis/symptoms-causes/syc-20354656" },
    { label: "Baptist Health — PCOS vs Endometriosis: Differences Explained", url: "https://www.baptisthealth.com/blog/womens-care/pcos-vs-endometriosis" },
    { label: "Office on Women's Health — Polycystic ovary syndrome", url: "https://womenshealth.gov" },
    { label: "Merck Manual (Consumer Version) — Polycystic Ovary Syndrome (PCOS)", url: "https://www.merckmanuals.com" },
  ];

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <header className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold text-charcoal mb-3">
          Health Library
        </h1>
        <p className="text-body/80 max-w-2xl mx-auto text-sm">
          Educational information about PCOS and Endometriosis. This is for awareness
          only and is not medical advice, diagnosis, or treatment — always consult a
          qualified healthcare provider about your own health.
        </p>
      </header>

      {/* PCOS */}
      <section id="pcos" className="scroll-mt-24 mb-14">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-5">
          <div className="flex items-center gap-3">
            <span className="w-11 h-11 rounded-full bg-soft-rose flex items-center justify-center shrink-0">
              <Activity size={22} className="text-pcos" />
            </span>
            <h2 className="text-2xl font-semibold text-pcos">Polycystic Ovary Syndrome (PCOS)</h2>
          </div>
          <div className="w-full sm:w-28 h-24 sm:h-20 flex items-center justify-center shrink-0 sm:ml-auto p-1">
            <img
              src="/illustrations/health-pcos.svg"
              alt=""
              className="w-full h-full object-contain"
              onError={(e) => { e.currentTarget.style.display = "none"; }}
            />
          </div>
        </div>

        <Card variant="pcos" className="mb-5">
          <h3 className="font-semibold text-charcoal mb-2">What is PCOS?</h3>
          <p className="text-sm text-body/85 mb-3">
            PCOS is a common hormonal disorder affecting women during their reproductive
            years and beyond. It occurs when hormonal signaling leads to higher-than-normal
            androgen (male hormone) levels and other imbalances, which can cause irregular
            periods, abnormal ovulation, changes in hair growth, and more.
          </p>
          <p className="text-sm text-body/85">
            It is a leading cause of irregular periods and one of the most common causes of
            infertility worldwide, and it persists as a chronic metabolic condition beyond
            the reproductive years.
          </p>
        </Card>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-3">Key facts</h3>
          <Bullets
            items={[
              "Affects an estimated 10–13% of reproductive-aged women globally.",
              "Up to 70% of women with PCOS worldwide may be undiagnosed.",
              "The most common cause of anovulation (lack of ovulation) among women.",
              "Runs in families and presents differently from person to person.",
              "Raises long-term risk of insulin resistance, type 2 diabetes, and obesity.",
            ]}
          />
        </div>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-3">Common symptoms</h3>
          <Bullets items={pcosSymptoms} />
        </div>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-3">Associated health risks</h3>
          <Bullets items={pcosRisks} />
        </div>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-2">Cause</h3>
          <p className="text-sm text-body/85">
            The cause is unknown, but a family history of PCOS or type 2 diabetes raises the
            likelihood. It can begin in adolescence but is often detected when women have
            difficulty becoming pregnant.
          </p>
        </div>

        <div>
          <h3 className="font-semibold text-charcoal mb-2">Treatment</h3>
          <p className="text-sm text-body/85">
            There is no cure, but lifestyle changes, medications, and fertility treatments can
            reduce symptoms, improve fertility, and protect long-term health. Options include
            insulin-sensitizing medication such as metformin, contraceptive therapies, and
            lifestyle changes like regular exercise, a healthy diet, weight management, and
            stress reduction.
          </p>
        </div>
      </section>

      {/* Endometriosis */}
      <section id="endo" className="scroll-mt-24 mb-14">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-5">
          <div className="flex items-center gap-3">
            <span className="w-11 h-11 rounded-full bg-endo-light flex items-center justify-center shrink-0">
              <Flower2 size={22} className="text-endo" />
            </span>
            <h2 className="text-2xl font-semibold text-endo">Endometriosis</h2>
          </div>
          <div className="w-full sm:w-28 h-24 sm:h-20 flex items-center justify-center shrink-0 sm:ml-auto p-1">
            <img
              src="/illustrations/health-endo.svg"
              alt=""
              className="w-full h-full object-contain"
              onError={(e) => { e.currentTarget.style.display = "none"; }}
            />
          </div>
        </div>

        <Card variant="endo" className="mb-5">
          <h3 className="font-semibold text-charcoal mb-2">What is Endometriosis?</h3>
          <p className="text-sm text-body/85">
            Endometriosis is a chronic, estrogen-dependent inflammatory disease in which tissue
            similar to the uterine lining grows outside the uterus. It causes inflammation and
            scar tissue and can bind organs together, most often involving the ovaries, fallopian
            tubes, and pelvic lining — and occasionally areas beyond the pelvis.
          </p>
        </Card>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-3">Key facts</h3>
          <Bullets
            color="bg-endo"
            items={[
              "Affects an estimated 10% (about 190 million) of reproductive-age women worldwide.",
              "A chronic disease with no known cure.",
              "Can affect intercourse, bowel and bladder function, and mental health.",
              "Also affects transgender men and non-binary people who menstruate.",
            ]}
          />
        </div>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-3">Common symptoms</h3>
          <Bullets color="bg-endo" items={endoSymptoms} />
        </div>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-3">Cause and risk factors</h3>
          <p className="text-sm text-body/85 mb-3">
            The cause is unknown. Emerging research links it to immune system dysregulation, and
            people with endometriosis have higher rates of other immune-mediated conditions. A
            family history also increases likelihood. Factors that may raise the chance include:
          </p>
          <Bullets color="bg-endo" items={endoRiskFactors} />
        </div>

        <div className="mb-5">
          <h3 className="font-semibold text-charcoal mb-2">Diagnosis</h3>
          <p className="text-sm text-body/85">
            Symptoms vary widely and some people have none, so it can be hard to diagnose — the
            average time to diagnosis is currently between 4 and 12 years. Methods include a careful
            menstrual history, pelvic exam, ultrasound or MRI, and laparoscopic surgery, which is
            often needed for definitive diagnosis. Newer approaches include symptom checklists and
            emerging saliva- or menstrual-blood-based tests.
          </p>
        </div>

        <div>
          <h3 className="font-semibold text-charcoal mb-2">Treatment</h3>
          <p className="text-sm text-body/85">
            There is no cure; treatment aims to control symptoms and limit long-term impact. Options
            include NSAID pain relief (such as ibuprofen or naproxen), hormonal medicines (combined
            contraceptives, progestins, GnRH analogues), and surgery to remove tissue when other
            treatments fail. Some hormonal treatments may not suit those trying to conceive.
          </p>
        </div>
      </section>

      {/* Comparison — stacked cards, not a table (mobile-friendly) */}
      <section className="mb-14">
        <h2 className="text-2xl font-semibold text-charcoal mb-2">PCOS vs Endometriosis</h2>
        <p className="text-sm text-body/80 mb-5">
          Both affect the reproductive system and can impact fertility, but they are
          fundamentally different conditions.
        </p>
        <div className="space-y-4">
          {comparison.map((row) => (
            <Card key={row.aspect} variant="default">
              <p className="text-xs uppercase tracking-wide text-body/50 font-semibold mb-3">
                {row.aspect}
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs font-semibold text-pcos mb-1">PCOS</p>
                  <p className="text-sm text-body/85">{row.pcos}</p>
                </div>
                <div>
                  <p className="text-xs font-semibold text-endo mb-1">Endometriosis</p>
                  <p className="text-sm text-body/85">{row.endo}</p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </section>

      {/* Myths — loose blush blocks, not a table */}
      <section className="mb-14">
        <h2 className="text-2xl font-semibold text-charcoal mb-5">Myths &amp; Facts</h2>
        <div className="space-y-3">
          {myths.map((m, i) => (
            <div key={i} className="rounded-card bg-blush border border-soft-rose p-5">
              <p className="text-sm text-body/60 mb-1">
                <span className="font-semibold text-deep-rose">Myth:</span>{" "}
                <span className="line-through">{m.myth}</span>
              </p>
              <p className="text-sm text-body/90">
                <span className="font-semibold text-pcos">Fact:</span> {m.fact}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Lifestyle */}
      <section className="mb-14">
        <h2 className="text-2xl font-semibold text-charcoal mb-2">Lifestyle &amp; Self-Care</h2>
        <p className="text-sm text-body/80 mb-5">
          Since both conditions are managed rather than cured, everyday habits matter.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <Card variant="default">
            <h3 className="font-semibold text-charcoal mb-3">Diet</h3>
            <Bullets
              items={[
                "Anti-inflammatory foods — leafy greens, berries, fatty fish, nuts.",
                "Low-glycemic-index carbohydrates to help manage insulin resistance (PCOS).",
                "Less processed food, sugar, and trans fat.",
              ]}
            />
          </Card>
          <Card variant="default">
            <h3 className="font-semibold text-charcoal mb-3">Physical activity</h3>
            <Bullets
              items={[
                "Moderate exercise — around 30 minutes, 5 days a week.",
                "Helps insulin sensitivity and reduces inflammation.",
              ]}
            />
          </Card>
          <Card variant="default">
            <h3 className="font-semibold text-charcoal mb-3">Stress management</h3>
            <Bullets
              items={[
                "Yoga, meditation, and adequate sleep support hormonal balance.",
              ]}
            />
          </Card>
          <Card variant="default">
            <h3 className="font-semibold text-charcoal mb-3">Tracking symptoms</h3>
            <Bullets
              items={[
                "Keep a symptom diary — period dates, pain levels, mood.",
                "This record also helps your doctor with diagnosis.",
              ]}
            />
          </Card>
        </div>
      </section>

      {/* References */}
      <section>
        <h2 className="text-xl font-semibold text-charcoal mb-4">References</h2>
        <ol className="space-y-2 list-decimal list-inside">
          {references.map((r, i) => (
            <li key={i} className="text-sm text-body/80">
              <a
                href={r.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-ai hover:text-pcos underline underline-offset-2 break-words"
              >
                {r.label}
              </a>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}