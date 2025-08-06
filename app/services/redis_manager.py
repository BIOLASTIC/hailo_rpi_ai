"""
Enhanced Redis connection manager with robust reconnection handling
"""
import asyncio
import logging
import time
from typing import AsyncGenerator
import redis.asyncio as redis
from app.config.settings import settings

logger = logging.getLogger(__name__)


class EnhancedRedisManager:
    def __init__(self):
        self.client = None
        self.connection_pool = None
        self.is_connected = False
        self.reconnect_count = 0
        self.last_connection_time = 0
        
    async def connect(self):
        """Connect to Redis with progressive fallback for async client"""
        try:
            # Method 1: Basic async connection (no complex options)
            logger.info("Trying basic async Redis connection...")
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=False
            )
            
            # Test connection
            await self.client.ping()
            
            self.is_connected = True
            self.last_connection_time = time.time()
            self.reconnect_count = 0
            
            logger.info("✅ Basic async Redis connection established")
            return True
            
        except Exception as e:
            logger.error(f"Basic connection failed: {e}")
            
        try:
            # Method 2: URL-based async connection
            logger.info("Trying URL-based async Redis connection...")
            self.client = redis.from_url(
                'redis://localhost:6379/0',
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            await self.client.ping()
            
            self.is_connected = True
            self.last_connection_time = time.time()
            self.reconnect_count = 0
            
            logger.info("✅ URL-based async Redis connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ All async Redis connection methods failed: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        try:
            if self.client:
                await self.client.close()
            self.is_connected = False
            logger.info("✅ Redis connection closed")
        except Exception as e:
            logger.error(f"❌ Error during disconnect: {e}")
    
    async def ensure_connection(self):
        """Ensure Redis connection is active"""
        try:
            if not self.is_connected or not self.client:
                return await self.connect()
            
            await asyncio.wait_for(self.client.ping(), timeout=5.0)
            return True
            
        except (redis.ConnectionError, redis.TimeoutError, asyncio.TimeoutError) as e:
            logger.warning(f"⚠️ Connection health check failed: {e}")
            self.is_connected = False
            self.reconnect_count += 1
            
            wait_time = min(2 ** min(self.reconnect_count, 5), 30)
            logger.info(f"🔄 Reconnecting in {wait_time}s (attempt #{self.reconnect_count})")
            await asyncio.sleep(wait_time)
            
            return await self.connect()
        
        except Exception as e:
            logger.error(f"❌ Unexpected error in ensure_connection: {e}")
            return False
    
    async def frame_generator(self, channel_name: str) -> AsyncGenerator[bytes, None]:
        """Generate video frames from Redis channel"""
        reconnect_attempts = 0
        max_reconnects = 10
        
        while reconnect_attempts < max_reconnects:
            pubsub = None
            try:
                if not await self.ensure_connection():
                    raise redis.ConnectionError("Failed to establish connection")
                
                logger.info(f"🔗 Subscribing to channel: {channel_name}")
                
                pubsub = self.client.pubsub()
                await pubsub.subscribe(channel_name)
                
                logger.info(f"✅ Successfully subscribed to {channel_name}")
                reconnect_attempts = 0
                
                last_keepalive = time.time()
                keepalive_interval = 30
                
                while True:
                    try:
                        message = await asyncio.wait_for(
                            pubsub.get_message(ignore_subscribe_messages=True),
                            timeout=2.0
                        )
                        
                        if message and message['type'] == 'message':
                            frame_data = message['data']
                            
                            yield (
                                b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n'
                                b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n'
                                b'\r\n' + frame_data + b'\r\n'
                            )
                            
                            last_keepalive = time.time()
                        
                        # Keepalive check
                        current_time = time.time()
                        if current_time - last_keepalive > keepalive_interval:
                            try:
                                await asyncio.wait_for(self.client.ping(), timeout=2.0)
                                last_keepalive = current_time
                            except Exception:
                                raise redis.ConnectionError("Keepalive failed")
                        
                        await asyncio.sleep(0.001)
                        
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        logger.info("🛑 Frame generator cancelled")
                        break
                    except (redis.ConnectionError, redis.TimeoutError):
                        logger.error("❌ Redis connection error in frame generation")
                        self.is_connected = False
                        raise
                    except Exception as e:
                        logger.error(f"❌ Unexpected error: {e}")
                        raise
                        
            except asyncio.CancelledError:
                logger.info("🛑 Frame generator cancelled during setup")
                break
            except (redis.ConnectionError, redis.TimeoutError) as e:
                reconnect_attempts += 1
                logger.error(f"❌ Connection error (attempt {reconnect_attempts}): {e}")
                
                if reconnect_attempts < max_reconnects:
                    wait_time = min(2 ** reconnect_attempts, 30)
                    logger.info(f"🔄 Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("💥 Max reconnection attempts reached")
                    break
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                reconnect_attempts += 1
                await asyncio.sleep(2)
            finally:
                if pubsub:
                    try:
                        await pubsub.unsubscribe(channel_name)
                        await pubsub.close()
                    except Exception as e:
                        logger.error(f"❌ Error cleaning up pubsub: {e}")


# Global instance
redis_manager = EnhancedRedisManager()
