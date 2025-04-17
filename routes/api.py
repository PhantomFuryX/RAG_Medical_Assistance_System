from fastapi import APIRouter
from src.api import chatAPI, ragAPI, documentAPI, systemInfo, medicalAPI, feedbackAPI, diagonosisAPI, imageAPI

router = APIRouter()
router.include_router(chatAPI.router)
router.include_router(ragAPI.router)
router.include_router(documentAPI.router)
router.include_router(systemInfo.router)
router.include_router(medicalAPI.router)
router.include_router(feedbackAPI.router)
router.include_router(diagonosisAPI.router)
router.include_router(imageAPI.router)
#i am here
#im here 2
# router.include_router(addresses.router)
# router.include_router(transaction_records.router)
# router.include_router(daily_table.router)
# router.include_router(general_updates.router)
# router.include_router(auth.router)