from datetime import timedelta, datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Body
from pymongo.database import Database as PymongoDatabase
from typing import Dict
from app.core import security
from app.core.config import settings
from app.crud import crud_user
from app.db.mongodb import get_database
from app.models import token as token_models
from app.models import user as user_models
from app.models.auth import LoginRequest, EmailVerificationRequest
from app.utils.email_utils import send_verification_email

router = APIRouter()
email_verification_requests_timestamps: Dict[str, datetime] = {}
EMAIL_VERIFICATION_COOLDOWN = timedelta(minutes=1)


@router.post("/register", response_model=user_models.User)
async def register_user(
    user_in: user_models.UserCreate, db: PymongoDatabase = Depends(get_database)
):
    """
    Create new user.
    """
    existing_user = await crud_user.get_user_by_email(db, email=user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The user with this email already exists.",
        )
    created_user_indb = await crud_user.create_user(db, user_in=user_in)

    verification_token = security.create_email_verification_token(
        email=created_user_indb.email
    )
    verification_link = f"{settings.SERVER_HOST}{settings.API_V1_STR}/auth/verify-email/{verification_token}"

    await send_verification_email(
        email_to=created_user_indb.email,
        username=created_user_indb.username,
        verification_link=verification_link,
    )

    return user_models.User.model_validate(created_user_indb)


@router.post("/login", response_model=token_models.Token)
async def login_for_access_token(
    db: PymongoDatabase = Depends(get_database), login_data: LoginRequest = Body(...)
):
    """
    Authenticate user and return JWT token along with user details.
    """
    user = await crud_user.get_user_by_email(db, email=login_data.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not security.verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = security.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    user_response = user_models.User.model_validate(user)
    return token_models.Token(access_token=access_token, user=user_response)


@router.post("/request-verification-email", status_code=status.HTTP_202_ACCEPTED)
async def request_email_verification(
    request_data: EmailVerificationRequest, db: PymongoDatabase = Depends(get_database)
):
    """
    Request a new email verification link.
    Implements a 1-minute cooldown per email address.
    """
    email_lower = request_data.email.lower()

    now = datetime.now(timezone.utc)
    if email_lower in email_verification_requests_timestamps:
        last_request_time = email_verification_requests_timestamps[email_lower]
        if now < last_request_time + EMAIL_VERIFICATION_COOLDOWN:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {EMAIL_VERIFICATION_COOLDOWN.total_seconds() / 60:.0f} minute(s) before requesting another verification email.",
            )

    user = await crud_user.get_user_by_email(db, email=email_lower)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User with this email not found.",
        )

    if user.is_email_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Email is already verified."
        )
    email_verification_requests_timestamps[email_lower] = now
    verification_token = security.create_email_verification_token(email=user.email)
    verification_link = f"{settings.SERVER_HOST}{settings.API_V1_STR}/auth/verify-email/{verification_token}"

    await send_verification_email(
        email_to=user.email, username=user.username, verification_link=verification_link
    )
    return {"msg": "Verification email sent. Please check your inbox."}


@router.get("/verify-email/{token}", response_model=user_models.User)
async def verify_email(token: str, db: PymongoDatabase = Depends(get_database)):
    """
    Verify user's email address using the token sent to their email.
    """
    email = security.verify_email_verification_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )
    user = await crud_user.get_user_by_email(db, email=email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found for this token.",
        )
    if user.is_email_verified:
        return user_models.User.model_validate(user)
    updated_user = await crud_user.update_user(
        db, user_id=str(user.id), user_in=user_models.UserUpdate(is_email_verified=True)
    )
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update email verification status.",
        )
    return user_models.User.model_validate(updated_user)
