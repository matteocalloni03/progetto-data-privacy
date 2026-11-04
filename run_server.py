"""
Web Dashboard and REST API Server Runner.
Launches the FastAPI application for the Health Data Privacy Gateway.
"""

import uvicorn
from src.config import HOST, PORT

if __name__ == "__main__":
    print(f"[*] Starting Health Data Privacy Gateway Web Dashboard...")
    print(f"[*] Serving on http://{HOST}:{PORT}")
    print(f"[*] API Documentation available at http://{HOST}:{PORT}/docs")
    uvicorn.run("src.web.app:app", host=HOST, port=PORT, reload=False)
