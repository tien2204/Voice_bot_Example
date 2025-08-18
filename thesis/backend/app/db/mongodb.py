from pymongo import MongoClient
from pymongo.database import Database as PymongoDatabase
from app.core.config import settings

client = MongoClient(settings.MONGODB_URL)
database: PymongoDatabase = client[settings.DATABASE_NAME]


async def get_database() -> PymongoDatabase:
    return database
