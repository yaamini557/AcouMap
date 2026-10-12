"""
AcouMap - FastAPI Backend Application Entry Point
Mounts API endpoints and serves the compiled industrial React dashboard.
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from backend.api.routes import router as api_router
from backend.database.db import DatabaseManager

app = FastAPI(
    title="AcouMap — Acoustic Fan Health & Anomaly Monitoring System",
    description="Industrial AI/ML predictive acoustic monitoring system for industrial fans.",
    version="1.0.0"
)

# Enable CORS for React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

@app.on_event("startup")
def on_startup():
    db = DatabaseManager()
    print("[AcouMap Backend] Database verified at:", db.db_path)

@app.get("/api/status")
@app.get("/api/health")
def api_status():
    return {
        "system": "AcouMap — Acoustic Fan Health & Anomaly Monitoring System",
        "status": "Operational",
        "version": "1.0.0",
        "api_docs": "/docs"
    }

# Static file serving for production React build
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_dist = os.path.join(base_dir, "frontend", "dist")

if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str, request: Request):
        # Do not catch /api calls
        if full_path.startswith("api"):
            return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
        # Check if caller wants JSON at root
        if full_path == "" and "application/json" in request.headers.get("accept", ""):
            return JSONResponse(content={
                "system": "AcouMap — Acoustic Fan Health & Anomaly Monitoring System",
                "status": "Operational",
                "version": "1.0.0",
                "api_docs": "/docs"
            })
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "system": "AcouMap — Acoustic Fan Health & Anomaly Monitoring System",
            "status": "Operational",
            "version": "1.0.0",
            "api_docs": "/docs"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
