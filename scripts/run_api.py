import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "spamdet.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("SPAMDET_RELOAD", "false").lower() == "true",
    )
