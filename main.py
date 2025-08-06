import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import redis.asyncio as redis
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import asynccontextmanager

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', case_sensitive=False)
    REDIS_HOST: str
    REDIS_PORT: int
    ANNOTATED_FRAME_OUTPUT_CHANNEL: str
    SERVER_PORT: int

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- AI Web Application Starting Up ---")
    app.state.redis_client = redis.from_url(f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}")
    await app.state.redis_client.ping()
    print("[INFO] Redis connection successful.")
    yield
    print("--- AI Web Application Shutting Down ---")
    await app.state.redis_client.close()

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    html_content = """
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
    <title>Live Hailo AI Detection</title>
    <style>body{font-family:sans-serif;background-color:#121212;color:#e0e0e0;margin:0;padding:20px;text-align:center;}img{display:block;margin:20px auto;max-width:90%;width:1280px;border-radius:8px;box-shadow:0 4px 15px rgba(0,0,0,0.5);}</style>
    </head><body><h1>Live AI Processed Video Feed</h1><img src="/video_feed" alt="Live AI Video Feed"></body></html>
    """
    return HTMLResponse(content=html_content)

async def frame_generator(redis_client, channel_name):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel_name)
    print(f"[INFO] Streaming from Redis channel: '{channel_name}'")
    while True:
        try:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
            if message:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + message['data'] + b'\r\n')
            else:
                # If no message, just continue to keep the connection open
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            print("[INFO] Client disconnected. Stopping stream.")
            break
        except Exception as e:
            print(f"[ERROR] Error in stream generator: {e}")
            break

@app.get("/video_feed")
async def video_feed(request: Request):
    return StreamingResponse(
        frame_generator(request.app.state.redis_client, settings.ANNOTATED_FRAME_OUTPUT_CHANNEL),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )