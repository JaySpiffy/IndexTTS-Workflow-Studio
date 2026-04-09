"""
Frontend serving endpoints for IndexTTS2 API.
"""

from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the frontend HTML file.
    
    Returns:
        HTMLResponse: Frontend HTML content
    """
    frontend_path = Path(__file__).parent.parent.parent.parent / "frontend" / "index.html"
    
    if not frontend_path.exists():
        return HTMLResponse(
            content="""
            <html>
                <head><title>IndexTTS2 - Frontend Not Found</title></head>
                <body>
                    <h1>Frontend Not Found</h1>
                    <p>The frontend files could not be found. Please ensure the frontend directory exists.</p>
                </body>
            </html>
            """,
            status_code=404
        )
    
    return FileResponse(
        path=frontend_path,
        media_type="text/html"
    )

@router.get("/assets/{file_path:path}")
async def serve_static_assets(file_path: str):
    """
    Serve static assets (CSS, JS, images).
    
    Args:
        file_path: Path to the static file
        
    Returns:
        FileResponse: Static file content
    """
    asset_path = Path(__file__).parent.parent.parent.parent / "frontend" / "assets" / file_path
    
    if not asset_path.exists():
        return HTMLResponse(
            content="Asset not found",
            status_code=404
        )
    
    # Determine media type based on file extension
    if file_path.endswith('.css'):
        media_type = "text/css"
    elif file_path.endswith('.js'):
        media_type = "application/javascript"
    elif file_path.endswith('.ico'):
        media_type = "image/x-icon"
    elif file_path.endswith('.png'):
        media_type = "image/png"
    elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
        media_type = "image/jpeg"
    elif file_path.endswith('.svg'):
        media_type = "image/svg+xml"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(
        path=asset_path,
        media_type=media_type
    )

@router.get("/src/{file_path:path}")
async def serve_src_files(file_path: str):
    """
    Serve JavaScript files from src directory.
    
    Args:
        file_path: Path to the JavaScript file
        
    Returns:
        FileResponse: JavaScript file content
    """
    src_path = Path(__file__).parent.parent.parent.parent / "frontend" / "src" / file_path
    
    if not src_path.exists():
        return HTMLResponse(
            content="JavaScript file not found",
            status_code=404
        )
    
    return FileResponse(
        path=src_path,
        media_type="application/javascript"
    )