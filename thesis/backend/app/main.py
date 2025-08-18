from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title="LiveKit App Backend", openapi_url=f"{settings.API_V1_STR}/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the LiveKit App Backend API"}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run FastAPI server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (dev mode)"
    )
    args = parser.parse_args()
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)
