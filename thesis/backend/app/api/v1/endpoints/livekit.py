import secrets
from fastapi import APIRouter, HTTPException, Response, Depends
from datetime import timedelta
from livekit import api
from app.core.config import settings
from app.models.livekit import ConnectionDetails
from app.api import deps
from app.models import user as user_models

router = APIRouter()


@router.get("/connection-details", response_model=ConnectionDetails)
async def get_livekit_connection_details(
    response: Response,
    current_user: user_models.UserInDB = Depends(deps.get_current_active_user),
):
    """Generate LiveKit connection details with a short-lived token."""
    response.headers["Cache-Control"] = "no-store"
    missing = [
        name
        for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
        if not getattr(settings, name)
    ]
    if missing:
        raise HTTPException(500, detail=f"Missing settings: {', '.join(missing)}")

    suffix = secrets.token_hex(4)
    participant_identity = f"{current_user.username}_{suffix}"
    room_name = f"voice_assistant_room_{suffix}"

    token = (
        api.AccessToken()
        .with_identity(participant_identity)
        .with_name(participant_identity)
        .with_grants(
            api.VideoGrants(
                room=room_name, room_join=True, can_publish=True, can_publish_data=True
            )
        )
        .with_ttl(timedelta(minutes=15))
    )
    jwt_token = token.to_jwt()
    return ConnectionDetails(
        server_url=settings.LIVEKIT_URL,
        room_name=room_name,
        participant_name=participant_identity,
        participant_token=jwt_token,
    )
