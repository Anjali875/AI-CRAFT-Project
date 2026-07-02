from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from validators.schemas import ChatInput
from services.gemini_service import get_chat_response

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api", tags=["Chat"])

@router.post("/chat")
@limiter.limit("20/minute")
def chat(request: Request, data: ChatInput):
    try:
        response = get_chat_response(
            message=data.message,
            condition=data.condition,
            risk_level=data.risk_level,
            risk_score=data.risk_score,
            contributing_factors=data.contributing_factors,
            symptoms=data.symptoms,
            history=[h.dict() for h in data.history],
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))