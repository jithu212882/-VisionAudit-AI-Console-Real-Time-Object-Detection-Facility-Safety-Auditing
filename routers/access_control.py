import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Form, Request, HTTPException, status
from pydantic import BaseModel, Field

from services.face_service import face_service

logger = logging.getLogger("visionaudit.access_control")

router = APIRouter(tags=["Access Control & Door Simulator"])


class VerifyAccessRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64 encoded face image or data URI")
    vector: Optional[List[float]] = Field(None, description="Direct 128-D facial embedding vector")
    threshold: float = Field(0.6, ge=0.0, le=1.0, description="Cosine similarity threshold for access grant")


class RegisterFaceResponse(BaseModel):
    status: str
    name: str
    enrolled_at: str
    embedding_dimension: int
    message: str


class VerifyAccessResponse(BaseModel):
    access_granted: bool
    matched_name: str
    confidence: float
    timestamp: str
    boxes: Optional[List[Dict[str, Any]]] = None
    detail: Optional[str] = None


@router.post(
    "/admin/register-face",
    response_model=RegisterFaceResponse,
    summary="Enroll new personnel with face embeddings",
    description="Extracts 128-D facial embeddings and caches them in memory and JSON storage."
)
async def register_face(
    request: Request,
    name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    role: Optional[str] = Form("Authorized Personnel"),
    image_base64: Optional[str] = Form(None)
):
    # Support JSON payload as well as multipart form
    content_type = request.headers.get("content-type", "")
    user_name = name
    user_role = role or "Authorized Personnel"
    image_data = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            user_name = body.get("name", user_name)
            user_role = body.get("role", user_role)
            image_data = body.get("image") or body.get("image_base64")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {e}"
            )
    else:
        if file is not None:
            image_data = await file.read()
        elif image_base64:
            image_data = image_base64

    if not user_name or not user_name.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Personnel name is required for enrollment."
        )

    if image_data is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Image file or base64 image data is required."
        )

    try:
        result = face_service.register_face(
            name=user_name.strip(),
            image_data=image_data,
            role=user_role.strip()
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error registering face for {user_name}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.get(
    "/admin/users",
    summary="List all enrolled personnel",
    description="Returns list of registered personnel with enrollment timestamps and thumbnail avatars."
)
async def list_enrolled_users():
    return face_service.get_enrolled_users()


@router.delete(
    "/admin/users/{name}",
    summary="Delete enrolled personnel",
    description="Removes a person from access control database."
)
async def delete_enrolled_user(name: str):
    success = face_service.delete_user(name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{name}' not found."
        )
    return {"status": "success", "message": f"User '{name}' removed successfully."}


@router.post(
    "/verify-access",
    response_model=VerifyAccessResponse,
    summary="Evaluate incoming face vector or frame for door access",
    description="Compares face embedding against enrolled users using cosine distance (threshold = 0.6)."
)
async def verify_access(
    request: Request,
    file: Optional[UploadFile] = File(None),
    threshold: float = Form(0.6)
):
    content_type = request.headers.get("content-type", "")
    target_data = None
    eval_threshold = threshold

    if "application/json" in content_type:
        try:
            body = await request.json()
            if "vector" in body and body["vector"]:
                target_data = body["vector"]
            elif "image" in body and body["image"]:
                target_data = body["image"]
            elif "image_base64" in body and body["image_base64"]:
                target_data = body["image_base64"]
            
            if "threshold" in body:
                eval_threshold = float(body["threshold"])
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {e}"
            )
    else:
        if file is not None:
            target_data = await file.read()

    if target_data is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Must provide an image file, base64 image string, or embedding vector."
        )

    try:
        result = face_service.verify_access(target_data, threshold=eval_threshold)
        return result
    except Exception as e:
        logger.error(f"Error during access verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )
