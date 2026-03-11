import re

def analyze_sentiment(text: str) -> str:
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(text)["compound"]

        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    except Exception:
        text_lower = text.lower()
        pos_words = ["thanks","great","good","happy","better","hopeful","okay","fine","nice","love","help","grateful"]
        neg_words = ["scared","worried","afraid","anxious","bad","worse","sad","depressed","upset","pain","hurt","fear","terrible","awful"]

        pos = sum(1 for w in pos_words if w in text_lower)
        neg = sum(1 for w in neg_words if w in text_lower)

        if pos > neg:
            return "Positive"
        elif neg > pos:
            return "Negative"
        else:
            return "Neutral"

TONE_PREFIX = {
    "Positive": "That's a great attitude! 😊 ",
    "Neutral":  "",
    "Negative": "I understand this can feel overwhelming. 💙 Take a breath — you're not alone. ",
}

TONE_SUFFIX = {
    "Positive": " Keep up that positive energy!",
    "Neutral":  "",
    "Negative": " Remember, early awareness is powerful. You're taking a great first step by being here.",
}

PCOS_KB = {
    "symptom": {
        "patterns": [r"symptom", r"sign", r"what is pcos", r"what's pcos", r"how do i know"],
        "response": (
            "Common PCOS symptoms include:\n"
            "• Irregular or missed periods 📅\n"
            "• Excess facial/body hair (hirsutism) 🪒\n"
            "• Acne or oily skin 😔\n"
            "• Hair thinning or loss 💇‍♀️\n"
            "• Weight gain, especially around the abdomen ⚖️\n"
            "• Darkening of skin folds (acanthosis nigricans)\n"
            "• Small cysts on ovaries (detected via ultrasound)\n\n"
            "Not everyone with PCOS will have all symptoms — it varies person to person."
        ),
    },
    "cycle": {
        "patterns": [r"cycle", r"period", r"irregular", r"menstrual", r"late period", r"missed period"],
        "response": (
            "Irregular menstrual cycles are one of the hallmark signs of PCOS. "
            "A normal cycle is 21–35 days. Cycles longer than 35 days, or fewer than 8 periods per year, "
            "may indicate hormonal imbalance.\n\n"
            "💡 Tip: Track your cycle using an app. Share the data with your doctor — it's very useful for diagnosis."
        ),
    },
    "diet": {
        "patterns": [r"diet", r"eat", r"food", r"nutrition", r"meal", r"avoid eating"],
        "response": (
            "Diet plays a huge role in managing PCOS:\n\n"
            "✅ Eat more: Leafy greens, whole grains, lean protein, legumes, berries, nuts\n"
            "❌ Reduce: Refined sugar, white bread/rice, processed snacks, sugary drinks\n\n"
            "A low-GI (glycaemic index) diet helps manage insulin resistance, which is common in PCOS. "
            "Consider consulting a nutritionist for a personalized plan."
        ),
    },
    "exercise": {
        "patterns": [r"exercise", r"workout", r"gym", r"physical activity", r"yoga", r"walk"],
        "response": (
            "Exercise is one of the most effective natural treatments for PCOS! 🏃‍♀️\n\n"
            "• Aim for: 150+ minutes of moderate exercise per week\n"
            "• Best types: Brisk walking, cycling, swimming, yoga, strength training\n"
            "• Why it helps: Improves insulin sensitivity, reduces androgens, helps weight management\n\n"
            "Even 30 minutes of walking daily can make a measurable difference."
        ),
    },
    "weight": {
        "patterns": [r"weight", r"fat", r"obese", r"bmi", r"overweight", r"lose weight"],
        "response": (
            "Weight and PCOS have a two-way relationship — PCOS can cause weight gain, "
            "and excess weight can worsen PCOS symptoms.\n\n"
            "Even a 5–10% reduction in body weight can significantly improve:\n"
            "• Menstrual regularity\n"
            "• Insulin sensitivity\n"
            "• Hormonal balance\n\n"
            "This doesn't mean crash dieting — small, sustainable changes work best."
        ),
    },
    "hair": {
        "patterns": [r"hair", r"hirsutism", r"facial hair", r"hair loss", r"thinning"],
        "response": (
            "Hair-related symptoms in PCOS are caused by excess androgens (male hormones):\n\n"
            "• Hirsutism — unwanted hair on face, chest, back\n"
            "• Androgenic alopecia — thinning/loss of scalp hair\n\n"
            "These are manageable with treatment. Options include:\n"
            "✅ Hormonal contraceptives (to reduce androgens)\n"
            "✅ Anti-androgen medications (prescribed by doctors)\n"
            "✅ Laser hair removal\n"
            "✅ Minoxidil for scalp hair (consult doctor first)"
        ),
    },
    "skin": {
        "patterns": [r"skin", r"acne", r"dark", r"pimple", r"acanthosis", r"darkening"],
        "response": (
            "PCOS commonly affects skin in two ways:\n\n"
            "1. Hormonal Acne — Often along the jawline and chin. Caused by high androgens.\n"
            "2. Acanthosis Nigricans — Dark, velvety patches on neck, underarms, or groin. "
            "This is a sign of insulin resistance.\n\n"
            "💡 If you notice skin darkening, this is a key sign to discuss with your doctor. "
            "Managing insulin resistance (through diet & exercise) often helps both."
        ),
    },
    "pregnancy": {
        "patterns": [r"pregnant", r"pregnan", r"fertility", r"conceive", r"baby", r"infertil"],
        "response": (
            "PCOS is one of the most common causes of irregular ovulation, which can affect fertility. "
            "However, many people with PCOS successfully conceive with proper treatment.\n\n"
            "Options include:\n"
            "• Lifestyle changes (diet + exercise) to regularize ovulation\n"
            "• Ovulation induction medications (e.g., Clomiphene, Letrozole)\n"
            "• IVF in some cases\n\n"
            "Please consult a gynaecologist or fertility specialist for personalized guidance. 💙"
        ),
    },
    "stress": {
        "patterns": [r"stress", r"anxious", r"anxiety", r"worried", r"mental health", r"depress"],
        "response": (
            "Mental health and PCOS are deeply connected. PCOS can increase the risk of anxiety and depression — "
            "you're not alone in feeling this way. 💙\n\n"
            "Helpful strategies:\n"
            "• Mindfulness and meditation (even 10 min/day helps)\n"
            "• Regular physical activity — a natural mood booster\n"
            "• Talking to a therapist or counsellor\n"
            "• Connecting with PCOS support communities\n\n"
            "Your emotional health is just as important as physical health. Please seek support if needed."
        ),
    },
    "diagnosis": {
        "patterns": [r"diagnos", r"test", r"doctor", r"check", r"ultrasound", r"blood test", r"confirm"],
        "response": (
            "PCOS is typically diagnosed using the Rotterdam Criteria — at least 2 of 3:\n"
            "1. Irregular or absent periods\n"
            "2. High androgen levels (blood test or clinical signs)\n"
            "3. Polycystic ovaries (ultrasound)\n\n"
            "Your doctor may also order:\n"
            "• Blood tests: LH/FSH ratio, testosterone, insulin, thyroid\n"
            "• Pelvic ultrasound\n\n"
            "⚠️ This app gives an estimate only — please see a gynaecologist for formal diagnosis."
        ),
    },
    "treatment": {
        "patterns": [r"treat", r"cure", r"medication", r"medicine", r"manage", r"control"],
        "response": (
            "PCOS cannot currently be 'cured', but it is very manageable! 💪\n\n"
            "Lifestyle (first-line):\n"
            "• Balanced diet + regular exercise\n"
            "• Stress management\n"
            "• Adequate sleep\n\n"
            "Medical (doctor-prescribed):\n"
            "• Hormonal contraceptives (regulate periods)\n"
            "• Metformin (insulin resistance)\n"
            "• Anti-androgens (hair/acne)\n"
            "• Fertility treatments (if needed)\n\n"
            "Treatment is always tailored to your specific symptoms and goals."
        ),
    },
    "result": {
        "patterns": [r"result", r"score", r"risk", r"my result", r"what does it mean", r"what now"],
        "response": "__RESULT_CONTEXT__",
    },
    "greet": {
        "patterns": [r"^hi$", r"^hello$", r"^hey$", r"^hii", r"^good (morning|afternoon|evening)"],
        "response": (
            "Hello! 👋 I'm your PCOS awareness assistant.\n\n"
            "I can help you with:\n"
            "• Understanding PCOS symptoms\n"
            "• Diet and lifestyle tips\n"
            "• Understanding your risk result\n"
            "• Answers to general PCOS questions\n\n"
            "What would you like to know?"
        ),
    },
    "thanks": {
        "patterns": [r"thank", r"thanks", r"helpful", r"appreciate"],
        "response": (
            "You're very welcome! 😊 Remember, awareness is the first step to taking control of your health. "
            "Stay consistent with healthy habits, and don't hesitate to reach out to a healthcare professional. "
            "Take care! 💜"
        ),
    },
}

RISK_ADVICE = {
    "Low": (
        "Your risk score suggests a low likelihood of PCOS based on the indicators provided. 🟢\n\n"
        "This is great news! To maintain this:\n"
        "• Continue regular exercise\n"
        "• Maintain a balanced, low-GI diet\n"
        "• Schedule routine gynaecological check-ups\n"
        "• Monitor your menstrual cycle regularly\n\n"
        "Remember, this is a screening tool — not a diagnosis. Annual check-ups are still recommended."
    ),
    "Moderate": (
        "Your risk score indicates a moderate likelihood of PCOS. ⚠️\n\n"
        "Some of your indicators suggest possible hormonal imbalance. I recommend:\n"
        "• Consulting a gynaecologist for further evaluation\n"
        "• Getting blood tests (LH, FSH, testosterone, insulin levels)\n"
        "• Improving diet — reduce sugar and processed foods\n"
        "• Increasing physical activity\n"
        "• Tracking your menstrual cycle for the next 2–3 months\n\n"
        "Early intervention makes a significant difference!"
    ),
    "High": (
        "Your risk score suggests a high likelihood of PCOS. 🔴\n\n"
        "Please don't panic — PCOS is very manageable with the right support. However, we strongly recommend:\n"
        "• Seeing a gynaecologist or endocrinologist as soon as possible\n"
        "• Getting a pelvic ultrasound and hormonal blood tests\n"
        "• Discussing medication options if appropriate\n"
        "• Starting lifestyle changes now (diet, exercise, stress management)\n\n"
        "You're taking the right step by being proactive. 💙 You've got this."
    ),
}

DEFAULT_RESPONSE = (
    "That's a great question! While I may not have a specific answer for that, "
    "here are some topics I can help with:\n\n"
    "• PCOS symptoms and signs\n"
    "• Diet and nutrition tips\n"
    "• Exercise recommendations\n"
    "• Understanding your risk result\n"
    "• Treatment and management options\n"
    "• Fertility and pregnancy with PCOS\n\n"
    "Try asking something like: 'What are PCOS symptoms?' or 'What should I eat?'"
)


def get_bot_response(user_text: str, sentiment: str, last_prediction: dict = None) -> str:
    text = user_text.lower().strip()
    prefix = TONE_PREFIX.get(sentiment, "")
    suffix = TONE_SUFFIX.get(sentiment, "")

    for topic, data in PCOS_KB.items():
        for pattern in data["patterns"]:
            if re.search(pattern, text):
                response = data["response"]

                if response == "__RESULT_CONTEXT__":
                    if last_prediction:
                        level = last_prediction.get("level", "Unknown")
                        prob  = last_prediction.get("prob", 0)
                        top   = last_prediction.get("top_feats", [])
                        response = (
                            f"Based on your last prediction:\n\n"
                            f"📊 Risk Score: {prob*100:.1f}%\n"
                            f"🏷️ Risk Level: {level}\n"
                            f"🔍 Top Contributing Factors: {', '.join(top) if top else 'N/A'}\n\n"
                        )
                        response += RISK_ADVICE.get(level, "")
                    else:
                        response = (
                            "You haven't run a prediction yet! Head to the 🔮 Risk Prediction page, "
                            "fill in your details, and come back to discuss your results. 😊"
                        )

                return f"{prefix}{response}{suffix}"

    return f"{prefix}{DEFAULT_RESPONSE}{suffix}"
