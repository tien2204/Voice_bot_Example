from pydantic import BaseModel, HttpUrl


class ConnectionDetails(BaseModel):
    server_url: str
    room_name: str
    participant_name: str
    participant_token: str
