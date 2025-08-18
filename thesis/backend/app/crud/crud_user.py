import asyncio
from bson import ObjectId
from pymongo.database import Database as PymongoDatabase
from typing import Optional, Dict, Any
import uuid
import httpx
from app.core.security import get_password_hash
from app.core.config import settings
from app.models.user import UserCreate, UserUpdate, UserInDB, AvatarMetadata

USER_COLLECTION = "users"
RPM_BASE_URL = "https://api.readyplayer.me/v1"


async def _create_rpm_guest_user() -> Optional[str]:
    if not settings.RPM_API_KEY or not settings.RPM_APPLICATION_ID:
        print(
            "Warning: RPM_API_KEY or RPM_APPLICATION_ID not configured. Skipping RPM guest user creation."
        )
        return None

    headers = {"x-api-key": settings.RPM_API_KEY, "Content-Type": "application/json"}
    payload = {"data": {"applicationId": settings.RPM_APPLICATION_ID}}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{RPM_BASE_URL}/users", json=payload, headers=headers
            )
            response.raise_for_status()
            rpm_user_data = response.json()
            return rpm_user_data.get("data", {}).get("id")
        except httpx.HTTPStatusError as e:
            print(
                f"RPM guest user creation failed: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            print(f"Error creating RPM guest user: {e}")
            return None


async def get_rpm_auth_token(
    rpm_user_id: str, partner_name: str = None
) -> Optional[str]:
    """
    Requests an authentication token from Ready Player Me for iFrame session restoration.
    The token has a very short lifespan (15 seconds) and should be requested
    right before the iFrame call.
    Args:
        rpm_user_id (str): The User ID for which to get the access token.
                           Note: This user needs to have authorized your app first.
        partner_name (str): Your partner name / subdomain.
    Returns:
        Optional[str]: The authentication token string if successful, None otherwise.
    """
    if not partner_name:
        partner_name = settings.RPM_PARTNER_SUBDOMAIN
    if not settings.RPM_API_KEY:
        print("Warning: RPM_API_KEY not configured. Cannot retrieve auth token.")
        return None
    headers = {"x-api-key": settings.RPM_API_KEY}

    token_url = f"{RPM_BASE_URL}/auth/token?userId={rpm_user_id}&partner={partner_name}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(token_url, headers=headers)
            print(response.text)
            response.raise_for_status()

            response_data = response.json()
            token = response_data.get("data", {}).get("token")
            if token:
                print(f"Successfully retrieved RPM auth token for user {rpm_user_id}.")
                return token
            else:
                print(
                    f"RPM auth token not found in response for user {rpm_user_id}: {response.text}"
                )
                return None
        except httpx.HTTPStatusError as e:
            print(
                f"RPM auth token HTTP error for user {rpm_user_id}: {e.response.status_code} - {e.response.text}"
            )
            return None
        except httpx.RequestError as e:
            print(f"RPM auth token request error for user {rpm_user_id}: {e}")
            return None
        except Exception as e:
            print(
                f"An unexpected error occurred while retrieving RPM auth token for user {rpm_user_id}: {e}"
            )
            return None


async def ensure_valid_rpm_guest_user_id(db: PymongoDatabase, user_doc: Dict[str, Any]):
    """
    Checks if the user_doc has a valid rpm_guest_user_id.
    If it's missing or invalid, attempts to create a new one and updates the database and user_doc.
    """
    rpm_id_from_db = user_doc.get("rpm_guest_user_id")
    if rpm_id_from_db:
        is_valid = await get_rpm_auth_token(rpm_id_from_db)
        if is_valid:
            return is_valid
        print(
            f"RPM guest user ID {rpm_id_from_db} for user {user_doc.get('_id')} was found invalid."
        )
    else:
        pass
    new_rpm_user_id = await _create_rpm_guest_user()
    await asyncio.to_thread(
        db[USER_COLLECTION].update_one,
        {"_id": user_doc["_id"]},
        {"$set": {"rpm_guest_user_id": new_rpm_user_id}},
    )
    user_doc["rpm_guest_user_id"] = new_rpm_user_id
    return await get_rpm_auth_token(new_rpm_user_id)


async def get_user_by_email(db: PymongoDatabase, email: str) -> Optional[UserInDB]:

    user_doc = await asyncio.to_thread(db[USER_COLLECTION].find_one, {"email": email})
    if user_doc:
        await ensure_valid_rpm_guest_user_id(db, user_doc)
        return UserInDB(**user_doc)
    return None


async def get_user_by_id(db: PymongoDatabase, user_id: str) -> Optional[UserInDB]:
    if not ObjectId.is_valid(user_id):
        return None
    user_doc = await asyncio.to_thread(
        db[USER_COLLECTION].find_one, {"_id": ObjectId(user_id)}
    )
    if user_doc:
        await ensure_valid_rpm_guest_user_id(db, user_doc)
        return UserInDB(**user_doc)
    return None


async def create_user(db: PymongoDatabase, user_in: UserCreate) -> UserInDB:
    hashed_password = get_password_hash(user_in.password)
    user_in.email = user_in.email.lower()
    user_db_data = user_in.model_dump(exclude={"password"})
    user_db_data["hashed_password"] = hashed_password

    user_db_data.setdefault("avatars", [])
    result = await asyncio.to_thread(db[USER_COLLECTION].insert_one, user_db_data)
    inserted_id = result.inserted_id

    rpm_user_id = await _create_rpm_guest_user()
    if rpm_user_id:
        await asyncio.to_thread(
            db[USER_COLLECTION].update_one,
            {"_id": inserted_id},
            {"$set": {"rpm_guest_user_id": rpm_user_id}},
        )
    created_user_doc = await asyncio.to_thread(
        db[USER_COLLECTION].find_one, {"_id": inserted_id}
    )
    if not created_user_doc:
        raise Exception("User retrieval failed unexpectedly after insert.")
    return UserInDB(**created_user_doc)


async def update_user(
    db: PymongoDatabase, user_id: str, user_in: UserUpdate
) -> Optional[UserInDB]:
    if not ObjectId.is_valid(user_id):
        return None
    update_data = user_in.model_dump(exclude_unset=True)
    if "email" in update_data and update_data["email"]:
        update_data["email"] = update_data["email"].lower()
    if "password" in update_data and update_data["password"]:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))

    if not update_data:
        return await get_user_by_id(db, user_id)
    result = await asyncio.to_thread(
        db[USER_COLLECTION].update_one,
        {"_id": ObjectId(user_id)},
        {"$set": update_data},
    )

    if result.modified_count == 1 or result.matched_count == 1:
        updated_user_doc = await asyncio.to_thread(
            db[USER_COLLECTION].find_one, {"_id": ObjectId(user_id)}
        )
        if updated_user_doc:
            return UserInDB(**updated_user_doc)
    return None


async def add_avatar_to_user(
    db: PymongoDatabase, user_id: str, avatar_data: AvatarMetadata
) -> Optional[UserInDB]:
    if not ObjectId.is_valid(user_id):
        return None

    avatar_dict = avatar_data.model_dump()
    update_result = await asyncio.to_thread(
        db[USER_COLLECTION].update_one,
        {"_id": ObjectId(user_id)},
        {"$push": {"avatars": avatar_dict}},
    )
    if update_result.modified_count == 1:
        return await get_user_by_id(db, user_id)
    return None
