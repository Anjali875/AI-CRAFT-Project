from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime

ROSE = colors.HexColor("#E8658A")
LAVENDER = colors.HexColor("#C9A8D4")
CHARCOAL = colors.HexColor("#2D2D2D")
GRAY = colors.HexColor("#6B7280")
LIGHT_ROSE = colors.HexColor("#FDF0F4")

def get_condition_color(condition: str):
    return ROSE if condition == "pcos" else LAVENDER

def build_pdf(
    condition: str,
    risk_level: str,
    risk_percentage: float,
    contributing_factors: list,
    symptoms: dict,
) -> bytes:
    buffer = io.BytesIO()
    condition_name = "PCOS" if condition == "pcos" else "Endometriosis"
    condition_color = get_condition_color(condition)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontSize=22,
        textColor=condition_color,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=GRAY,
        fontName="Helvetica",
        alignment=TA_CENTER,
        spaceAfter=4,
    )

    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontSize=13,
        textColor=condition_color,
        fontName="Helvetica-Bold",
        spaceBefore=16,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=CHARCOAL,
        fontName="Helvetica",
        spaceAfter=4,
        leading=16,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=GRAY,
        fontName="Helvetica-Oblique",
        alignment=TA_CENTER,
        spaceAfter=4,
        leading=14,
    )

    story = []

    # Header
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Women's Health AI", title_style))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"{condition_name} Screening Report", subtitle_style))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=condition_color))
    story.append(Spacer(1, 0.4 * cm))

    # Disclaimer banner
    story.append(Paragraph(
        "⚠ This report is for educational awareness only and does not constitute a medical diagnosis. "
        "Please consult a qualified gynaecologist for proper evaluation.",
        disclaimer_style
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))

    # Screening result
    story.append(Paragraph("Screening Result", section_heading_style))
    story.append(Paragraph(
        f"<b>Condition Screened:</b> {condition_name}",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Result Category:</b> {risk_level} likelihood",
        body_style
    ))
    story.append(Paragraph(
        f"<b>Estimated Risk Score:</b> {risk_percentage:.1f}%",
        body_style
    ))

    # Risk explanation
    story.append(Paragraph("What This Means", section_heading_style))
    if risk_level == "Low":
        explanation = (
            "Your responses suggest a lower likelihood of indicators associated with "
            f"{condition_name}. This is encouraging, but it does not rule out the condition entirely. "
            "Continue maintaining a healthy lifestyle and consult a doctor if you notice any symptoms."
        )
    elif risk_level == "Moderate":
        explanation = (
            "Your responses suggest some indicators associated with "
            f"{condition_name} were identified. This does not mean you have the condition, "
            "but it may be worth discussing your symptoms with a qualified gynaecologist."
        )
    else:
        explanation = (
            "Your responses suggest several indicators associated with "
            f"{condition_name} were identified. This screening does not diagnose the condition. "
            "We strongly encourage you to schedule an appointment with a gynaecologist for a "
            "proper clinical evaluation."
        )
    story.append(Paragraph(explanation, body_style))

    # Contributing factors
    if contributing_factors:
        story.append(Paragraph("Key Contributing Factors", section_heading_style))
        story.append(Paragraph(
            "The following factors from your responses contributed to your screening result:",
            body_style
        ))
        for factor in contributing_factors:
            story.append(Paragraph(f"• {factor}", body_style))

    # Symptom summary
    if symptoms:
        story.append(Paragraph("Your Responses Summary", section_heading_style))
        for key, value in symptoms.items():
            story.append(Paragraph(f"<b>{key}:</b> {value}", body_style))

    # Next steps
    story.append(Paragraph("Recommended Next Steps", section_heading_style))
    next_steps = [
        "Share this report with a qualified gynaecologist for professional evaluation.",
        "Do not self-diagnose or start any treatment based on this screening alone.",
        "Maintain a balanced diet, regular exercise, adequate sleep, and manage stress.",
        "Track your symptoms over time and note any changes before your doctor's visit.",
        "If you experience severe pain or heavy bleeding, seek medical attention immediately.",
    ]
    for step in next_steps:
        story.append(Paragraph(f"• {step}", body_style))

    # Footer
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Women's Health AI — Educational Screening Tool | Not a substitute for medical advice",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()