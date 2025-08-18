from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database as PymongoDatabase
from typing import List
from app.api import deps
from app.crud import crud_user
from app.db.mongodb import get_database
from app.models import user as user_models

router = APIRouter()


@router.get("/me", response_model=user_models.User)
async def read_current_user(
    current_user_indb: user_models.UserInDB = Depends(deps.get_current_active_user),
):
    """
    Get current authenticated user's details.
    """
    return user_models.User.model_validate(current_user_indb)


@router.put("/me", response_model=user_models.User)
async def update_current_user(
    user_update_data: user_models.UserUpdate,
    db: PymongoDatabase = Depends(get_database),
    current_user_indb: user_models.UserInDB = Depends(deps.get_current_active_user),
):
    """
    Update current authenticated user's details.
    """
    if user_update_data.email and user_update_data.email != current_user_indb.email:
        existing_user = await crud_user.get_user_by_email(
            db, email=user_update_data.email
        )
        if existing_user and existing_user.id != current_user_indb.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered by another user.",
            )
    updated_user_indb = await crud_user.update_user(
        db, user_id=str(current_user_indb.id), user_in=user_update_data
    )
    if not updated_user_indb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or update failed.",
        )
    return user_models.User.model_validate(updated_user_indb)


@router.get("/me/avatars", response_model=List[user_models.AvatarMetadata])
async def get_current_user_avatars(
    current_user_indb: user_models.UserInDB = Depends(deps.get_current_active_user),
):
    """
    Get current user's list of avatar metadata.
    """
    return current_user_indb.avatars


@router.post("/me/avatars", response_model=user_models.User)
async def add_current_user_avatar(
    avatar_data: user_models.AvatarMetadata,
    db: PymongoDatabase = Depends(get_database),
    current_user_indb: user_models.UserInDB = Depends(deps.get_current_active_user),
):
    """
    Add new avatar metadata for the current user. Returns the full updated user.
    """
    updated_user_indb = await crud_user.add_avatar_to_user(
        db, user_id=str(current_user_indb.id), avatar_data=avatar_data
    )
    if not updated_user_indb:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Failed to add avatar or user not found.",
        )
    return user_models.User.model_validate(updated_user_indb)


@router.get("/me/rpm-auth-token", response_model=user_models.RPMToken)
async def get_rpm_auth_token_for_current_user(
    db: PymongoDatabase = Depends(get_database),
    current_user_indb: user_models.UserInDB = Depends(deps.get_current_active_user),
):
    """
    Get a short-lived Ready Player Me authentication token for the current user.
    This token is used for iFrame session restoration.
    """
    if not current_user_indb.rpm_guest_user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="RPM user ID not found for this user. Cannot generate auth token.",
        )
    user_doc = current_user_indb.model_dump()
    if not user_doc.get("_id"):
        user_doc["_id"] = user_doc.get("id")
    token = await crud_user.ensure_valid_rpm_guest_user_id(db, user_doc=user_doc)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve Ready Player Me authentication token.",
        )
    return {"token": token}
