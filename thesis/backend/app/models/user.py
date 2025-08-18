from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, Type
from bson import ObjectId
from pydantic_core import core_schema, CoreSchema
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler


class ObjectIdStr(str):
    @classmethod
    def _validate_value(cls, v: Any) -> str:
        """Internal method to validate and convert a value to an ObjectId string."""
        if isinstance(v, ObjectId):
            return str(v)
        if isinstance(v, str):
            if ObjectId.is_valid(v):

                return v
            raise ValueError(f"'{v}' is not a valid ObjectId string.")
        raise TypeError(f"Expected ObjectId or str, got {type(v).__name__}.")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Defines the Pydantic V2 core schema for ObjectIdStr."""
        return core_schema.no_info_plain_validator_function(cls._validate_value)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, current_core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ):
        """Defines the JSON schema (e.g., for OpenAPI) for ObjectIdStr."""

        return handler(core_schema.str_schema(pattern="^[0-9a-fA-F]{24}$"))


class AvatarMetadata(BaseModel):
    glb_url: str
    rpm_avatar_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class UserBase(BaseModel):
    email: EmailStr
    username: str
    is_active: bool = True
    is_superuser: bool = False
    is_email_verified: bool = False
    avatars: list[AvatarMetadata] = Field(default_factory=list)
    rpm_guest_user_id: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None

    avatars: Optional[list[AvatarMetadata]] = None
    is_email_verified: Optional[bool] = None


class UserInDB(UserBase):
    id: ObjectIdStr = Field(alias="_id")
    hashed_password: str

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class User(UserBase):
    id: ObjectIdStr = Field(alias="_id")

    class Config:
        from_attributes = True
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class RPMToken(BaseModel):
    token: str
