import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.api import router as api_router
from src.utils.settings import settings
from src.utils.gpu_utils import print_gpu_info
from src.utils.db_init import init_database

origins = [
    settings.CLIENT_ORIGIN,
    settings.CLIENT_ORIGIN_ONLINE
]

app = FastAPI(title="Medical Assistant API")

# Print GPU info at startup
@app.on_event("startup")
async def startup_event():
    print_gpu_info()
    init_database()
@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical Assistant Application!"}

app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("main:app", log_level="info", port = 3000, reload=True)