from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from pathlib import Path
import subprocess
import os
import re
import aiohttp
from pathlib import Path
from fastapi import HTTPException
from vidaio_subnet_core import CONFIG
from typing import Optional
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video
from loguru import logger
import traceback
import asyncio
from urllib.parse import quote

app = FastAPI()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    chain_strategy: Optional[str] = None
    

def get_frame_rate(input_file: Path) -> float:
    """
    Extracts the frame rate of the input video using FFmpeg.

    Args:
        input_file (Path): The path to the video file.

    Returns:
        float: The frame rate of the video.
    """
    frame_rate_command = [
        "ffmpeg",
        "-i", str(input_file),
        "-hide_banner"
    ]
    process = subprocess.run(frame_rate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stderr  # Frame rate is usually in stderr

    # Extract frame rate using regex
    match = re.search(r"(\d+(?:\.\d+)?) fps", output)
    if match:
        return float(match.group(1))
    else:
        raise HTTPException(status_code=500, detail="Unable to determine frame rate of the video.")


def upscale_video(payload_video_path: str, task_type: str):
    """
    Upscales a video using the video2x tool and returns the full paths of the upscaled video and the converted mp4 file.

    Args:
        payload_video_path (str): The path to the video to upscale.
        task_type (str): The type of upscaling task to perform.

    Returns:
        str: The full path to the upscaled video.
    """
    try:
        input_file = Path(payload_video_path)

        scale_factor = "2"

        if task_type == "SD24K":
            scale_factor = "4"

        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        # Get the frame rate of the video
        frame_rate = get_frame_rate(input_file)
        print(f"Frame rate detected: {frame_rate} fps")

        # Calculate the duration to duplicate 2 frames
        stop_duration = 2 / frame_rate

        # Generate output file paths
        output_file_with_extra_frames = input_file.with_name(f"{input_file.stem}_extra_frames.mp4")
        output_file_upscaled = input_file.with_name(f"{input_file.stem}_upscaled.mp4")

        # Step 1: Duplicate the last frame two times
        print("Step 1: Duplicating the last frame two times...")
        start_time = time.time()

        duplicate_last_frame_command = [
            "ffmpeg",
            "-i", str(input_file),
            "-vf", f"tpad=stop_mode=clone:stop_duration={stop_duration}",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            str(output_file_with_extra_frames)
        ]

        duplicate_last_frame_process = subprocess.run(
            duplicate_last_frame_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        elapsed_time = time.time() - start_time
        if duplicate_last_frame_process.returncode != 0:
            print(f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
        if not output_file_with_extra_frames.exists():
            print("MP4 video file with extra frames was not created.")
            raise HTTPException(status_code=500, detail="MP4 video file with extra frames was not created.")
        print(f"Step 1 completed in {elapsed_time:.2f} seconds. File with extra frames: {output_file_with_extra_frames}")

        # Step 2: Upscale video using video2x
        print("Step 2: Upscaling video using video2x...")
        start_time = time.time()
        video2x_command = [
            "video2x",
            "-i", str(output_file_with_extra_frames),
            "-o", str(output_file_upscaled),
            "-p", "realesrgan",  
            "-s", scale_factor,  # Scale factor of 2 or 4
            "-c", "libx264",  
            "-e", "preset=slow",  
            "-e", "crf=28"
        ]
        video2x_process = subprocess.run(video2x_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed_time = time.time() - start_time
        if video2x_process.returncode != 0:
            print(f"Upscaling failed: {video2x_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Upscaling failed: {video2x_process.stderr.strip()}")
        if not output_file_upscaled.exists():
            print("Upscaled MP4 video file was not created.")
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")
        print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

        # Cleanup intermediate files if needed
        if output_file_with_extra_frames.exists():
            output_file_with_extra_frames.unlink()
            print(f"Intermediate file {output_file_with_extra_frames} deleted.")
            
        if input_file.exists():
            input_file.unlink()
            print(f"Original file {input_file} deleted.")
        
        print(f"Returning from FastAPI: {output_file_upscaled}")
        return output_file_upscaled
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# New: configurable external upscaler URL (env override)
EXTERNAL_UPSCALER_URL = os.environ.get("EXTERNAL_UPSCALER_URL", "http://localhost:8006/upscale-only/")

def _scale_from_task(task_type: str) -> int:
    """
    Map task_type to numeric scale factor used by app.py.
    SD24K -> 4 (SD -> 4K)
    HD24K -> 2 (HD -> 4K)
    SD2HD -> 2 (SD -> HD)
    Fallback: try to extract digits, else 2
    """
    if not task_type:
        return 2
    t = str(task_type).strip().upper()
    if t == "SD24K":
        return 4
    if t == "HD24K":
        return 2
    if t == "SD2HD":
        return 2
    # try to extract numeric scale (e.g., "4", "4X", "8")
    m = re.search(r"(\d+)", t)
    if m:
        try:
            val = int(m.group(1))
            return val if val >= 2 else 2
        except:
            pass
    return 2

async def call_external_upscaler(payload_video_path: str, task_type: str, chain_strategy: Optional[str] = None) -> str:
    """
    POST the downloaded file to external upscaler and save returned MP4 locally.
    Returns path to the upscaled file.
    """
    scale = _scale_from_task(task_type)
    try:
        async with aiohttp.ClientSession() as session:
            with open(payload_video_path, "rb") as fh:
                data = aiohttp.FormData()
                data.add_field("file", fh, filename=Path(payload_video_path).name, content_type="video/mp4")
                data.add_field("scale_factor", str(scale))
                data.add_field("model_type", "general")
                if chain_strategy:
                    data.add_field("chain_strategy", chain_strategy)

                async with session.post(EXTERNAL_UPSCALER_URL, data=data, timeout=3600) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise HTTPException(status_code=500, detail=f"External upscaler failed: {resp.status} - {text}")

                    content = await resp.read()
                    out_path = str(Path(payload_video_path).with_name(f"{Path(payload_video_path).stem}_upscaled_external.mp4"))
                    with open(out_path, "wb") as out_f:
                        out_f.write(content)
                    return out_path
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"External upscaler error: {e}")

@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    """
    Download payload_url, forward to external upscaler, upload result to storage, schedule deletion.
    """
    try:
        payload_url = request.payload_url
        task_type = request.task_type
        chain_strategy = request.chain_strategy

        logger.info(f"📻 Downloading video.... (chain_strategy={chain_strategy})")
        payload_video_path: str = await download_video(payload_url)
        logger.info(f"Download finished: {payload_video_path}")

        # Forward to external service (now passing optional chain_strategy)
        processed_video_path = await call_external_upscaler(payload_video_path, task_type, chain_strategy)

        # remove original downloaded file
        try:
            if os.path.exists(payload_video_path):
                os.remove(payload_video_path)
                logger.info(f"Removed original downloaded file: {payload_video_path}")
        except Exception:
            logger.warning(f"Could not delete original: {payload_video_path}")

        if not processed_video_path or not os.path.exists(processed_video_path):
            logger.error("No processed file returned from external upscaler")
            return JSONResponse({"uploaded_video_url": None})

        object_name = Path(processed_video_path).name
        await storage_client.upload_file(object_name, processed_video_path)
        logger.info("Uploaded processed video to storage")

        # remove local processed file
        try:
            if os.path.exists(processed_video_path):
                os.remove(processed_video_path)
                logger.info(f"Removed local processed file: {processed_video_path}")
        except Exception:
            logger.warning(f"Could not delete processed file: {processed_video_path}")

        try:
            sharing_link: str | None = await storage_client.get_presigned_url(object_name)
        except Exception as e:
            # Fallback: construct direct bucket URL (requires public bucket/policy)
            logger.exception("Presigned URL generation failed; falling back to direct bucket URL")
            endpoint = getattr(CONFIG, "storage").endpoint if hasattr(CONFIG, "storage") else os.environ.get("BUCKET_COMPATIBLE_ENDPOINT", "")
            bucket = getattr(CONFIG, "storage").bucket_name if hasattr(CONFIG, "storage") else os.environ.get("BUCKET_NAME", "")
            endpoint = endpoint.rstrip("/")
            if not endpoint or not bucket:
                logger.error("Cannot construct fallback URL: missing endpoint or bucket name")
                return JSONResponse({"uploaded_video_url": None})
            sharing_link = f"{endpoint}/{bucket}/{quote(object_name)}"
            logger.info(f"Using fallback direct URL: {sharing_link}")

        if not sharing_link:
            logger.error("Upload failed, no presigned URL or fallback URL")
            return JSONResponse({"uploaded_video_url": None})

        # schedule deletion
        schedule_file_deletion(object_name)

        return JSONResponse({"uploaded_video_url": sharing_link})

    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        return JSONResponse({"uploaded_video_url": None})


if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)