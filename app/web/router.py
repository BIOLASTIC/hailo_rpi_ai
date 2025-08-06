from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ..services.websocket_manager import manager as websocket_manager

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Serves the main dashboard page."""
    templates = request.app.state.templates
    # Pass the camera ID from config to the template
    # NOTE: In a real app, this would come from a config object
    camera_id = "usb" 
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "camera_id": camera_id
    })

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        print("Client disconnected.")