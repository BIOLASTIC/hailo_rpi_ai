from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
import redis.asyncio as redis

router = APIRouter()

async def redis_frame_generator(redis_client, channel_name: str):
    """Subscribes to a Redis channel and yields JPEG frame bytes."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel_name)
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0)
            if message and message.get("type") == "message":
                frame_bytes = message['data']
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.001) # Yield control to the event loop
    finally:
        await pubsub.close()

@router.get("/camera/stream/{camera_id}")
async def get_camera_stream(camera_id: str, request: Request):
    """Provides the live, RAW video feed from the specified camera."""
    redis_client = request.app.state.redis_client
    channel = f"camera:frames:{camera_id}"
    return StreamingResponse(redis_frame_generator(redis_client, channel), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/ai/stream/{camera_id}")
async def get_ai_stream(camera_id: str, request: Request):
    """Provides the live, AI-ANNOTATED video feed for the specified camera."""
    redis_client = request.app.state.redis_client
    channel = f"ai_stream:frames:{camera_id}"
    return StreamingResponse(redis_frame_generator(redis_client, channel), media_type="multipart/x-mixed-replace; boundary=frame")