from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from validators.schemas import PCOSInput, EndoInput, ReportInput
from services.pcos_service import predict_pcos
from services.endo_service import predict_endo
from services.pdf_service import build_pdf

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/predict", tags=["Predictions"])

@router.post("/pcos")
@limiter.limit("10/minute")
def pcos_prediction(request: Request, data: PCOSInput):
    try:
        result = predict_pcos(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/endo")
@limiter.limit("10/minute")
def endo_prediction(request: Request, data: EndoInput):
    try:
        result = predict_endo(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/download-report")
@limiter.limit("5/minute")
def download_report(request: Request, data: ReportInput):
    try:
        pdf_bytes = build_pdf(
            condition=data.condition,
            risk_level=data.risk_level,
            risk_percentage=data.risk_percentage,
            contributing_factors=data.contributing_factors,
            symptoms=data.symptoms,
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=womens_health_screening_report.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))