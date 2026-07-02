import { useNavigate } from "react-router-dom";
import { Brain, Sparkles, MessageCircle, ShieldAlert } from "lucide-react";
import Card from "../components/Card";

export default function About() {
  const navigate = useNavigate();

  const models = [
    {
      name: "Logistic Regression",
      body: "A foundational statistical model that estimates the probability of a condition from your symptom inputs. Simple, interpretable, and a strong baseline.",
    },
    {
      name: "Random Forest",
      body: "An ensemble of many decision trees. Averaging across trees improves accuracy and reduces the overfitting a single tree can suffer from.",
    },
    {
      name: "XGBoost",
      body: "A high-performance gradient-boosting method that builds trees sequentially, each correcting the last. Known for strong accuracy on structured, tabular data.",
    },
  ];

  const steps = [
    "Choose the condition you want to screen for — PCOS or Endometriosis.",
    "Answer a short, symptom-based questionnaire.",
    "Your responses are analysed by the model selected for that condition.",
    "You receive a likelihood result, with the option to ask our assistant about it.",
  ];

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      {/* Intro */}
      <header className="mb-12">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-charcoal mb-4">
              About This Project
            </h1>
            <p className="text-body/85 text-sm mb-3">
              This website is a student research project using machine learning to
              support earlier risk awareness for two common but frequently
              underdiagnosed reproductive health conditions: Polycystic Ovary
              Syndrome (PCOS) and Endometriosis.
            </p>
            <p className="text-body/85 text-sm">
              It offers a free, private, preliminary screening: enter a few details,
              get a likelihood estimate, and decide your next step. Nothing you enter
              is stored, and no account is needed.
            </p>
          </div>
          <div className="rounded-card aspect-[4/3] flex items-center justify-center p-4">
            <img
              src="/illustrations/about-hero.svg"
              alt=""
              className="w-full h-full object-contain"
              onError={(e) => { e.currentTarget.style.display = "none"; }}
            />
          </div>
        </div>
      </header>

      {/* Why we built it */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-charcoal mb-4">Why We Built This</h2>
        <div className="space-y-3 text-sm text-body/85">
          <p>
            When we started, we looked for existing tools and data in this space and
            found relatively little — few accessible screening tools aimed at everyday
            users, and limited public datasets to work from. At the same time, delayed
            diagnosis is a well-documented problem: many women live with symptoms for
            years before anyone takes them seriously.
          </p>
          <p>
            Too often those symptoms get dismissed — brushed off as normal, or as
            stress or anxiety — and that dismissal pushes a real diagnosis further and
            further back. We wanted to push against that. A website can reach far and
            wide with nothing more than an internet connection, so it felt like the
            most practical way to help people recognise that their symptoms are worth
            paying attention to.
          </p>
          <p>
            This tool doesn't diagnose anything, and it isn't meant to. It gives a
            likelihood and encourages you to see a doctor. Whether or not you turn out
            to have PCOS or endometriosis, taking your health seriously early is better
            than finding out late.
          </p>
        </div>
      </section>

      {/* How the models work */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-charcoal mb-4">
          How Our Machine Learning Works
        </h2>
        <p className="text-sm text-body/85 mb-5">
          We trained and compared three algorithms, each with different strengths:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
          {models.map((m) => (
            <Card key={m.name} variant="default" className="flex flex-col">
              <div className="w-11 h-11 rounded-full bg-soft-rose flex items-center justify-center mb-4">
                <Brain size={20} className="text-pcos" />
              </div>
              <h3 className="font-semibold text-charcoal mb-2">{m.name}</h3>
              <p className="text-xs text-body/80">{m.body}</p>
            </Card>
          ))}
        </div>
        <Card variant="pcos">
          <div className="flex items-start gap-3">
            <Sparkles size={20} className="text-pcos shrink-0 mt-0.5" />
            <p className="text-sm text-body/85">
              After comparing all three, we kept the best performer for each condition
              in our testing: <span className="font-semibold text-pcos">XGBoost</span>{" "}
              for PCOS and{" "}
              <span className="font-semibold text-endo">Logistic Regression</span> for
              Endometriosis. These reflect the results we observed on our own data —
              not a general claim that one algorithm is best.
            </p>
          </div>
        </Card>
      </section>

      {/* The process */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-charcoal mb-4">How It Works</h2>
        <ol className="space-y-3">
          {steps.map((s, i) => (
            <li key={i} className="flex gap-3 text-sm text-body/85">
              <span className="w-6 h-6 rounded-full bg-pcos text-white text-xs font-semibold flex items-center justify-center shrink-0">
                {i + 1}
              </span>
              <span className="pt-0.5">{s}</span>
            </li>
          ))}
        </ol>
      </section>

      {/* Chatbot */}
      <section className="mb-12">
        <h2 className="text-2xl font-semibold text-charcoal mb-4">The AI Assistant</h2>
        <Card variant="ai">
          <div className="flex items-start gap-3">
            <MessageCircle size={20} className="text-ai shrink-0 mt-0.5" />
            <p className="text-sm text-body/85">
              After a screening you may still have questions — what a result means, or
              what to ask a doctor. We've built in an assistant powered by a large
              language model (LLM) to answer general questions, explain terminology,
              and point you toward sensible next steps, so you don't leave with your
              questions unanswered.
            </p>
          </div>
        </Card>
      </section>

      {/* Disclaimer */}
      <section className="mb-12">
        <Card variant="default" className="border-deep-rose/30">
          <div className="flex items-start gap-3">
            <ShieldAlert size={20} className="text-deep-rose shrink-0 mt-0.5" />
            <div>
              <h2 className="text-lg font-semibold text-charcoal mb-2">
                Not a Medical Diagnosis
              </h2>
              <div className="space-y-2 text-sm text-body/85">
                <p>
                  This tool is for educational and research purposes only. Its models
                  are trained on historical medical data to recognise patterns and
                  estimate likelihood — they do not diagnose.
                </p>
                <p>
                  It does not replace a qualified professional. Only a registered
                  gynaecologist or healthcare provider can diagnose PCOS,
                  Endometriosis, or any other condition.
                </p>
                <p>
                  If your result shows a higher likelihood, we strongly encourage you
                  to see a doctor for proper evaluation, which may include blood tests
                  and imaging such as an ultrasound. Even if your result shows a lower
                  likelihood, it is still advised to consult a healthcare professional.
                </p>
              </div>
            </div>
          </div>
        </Card>
      </section>

      {/* Developers */}
      <section className="mb-4">
        <h2 className="text-2xl font-semibold text-charcoal mb-4">About the Developers</h2>
        <p className="text-sm text-body/85 mb-4">
          We're a team of two — Meihul Saini and Anjali — second-year students heading
          into our third year of a B.Tech in Data Science at Amity University, Noida.
          We built this to bring machine learning to bear on a real gap in women's
          healthcare.
        </p>
        <p className="text-sm text-body/85">
          We plan to expand the platform over time: more conditions, a more capable
          assistant, and models refined on larger, more diverse datasets.
        </p>
      </section>

      <div className="text-center">
        <button
          onClick={() => navigate("/screening")}
          className="inline-flex items-center gap-2 rounded-pill bg-pcos text-white px-7 py-3 text-sm font-medium hover:opacity-90 transition-opacity cursor-pointer"
        >
          Start Screening
        </button>
      </div>
    </div>
  );
}