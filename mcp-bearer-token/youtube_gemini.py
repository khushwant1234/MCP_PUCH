import os
import csv
import re
import uuid
import asyncio
import logging
import time
import json
import aiohttp
from enum import Enum
from typing import Annotated
from pydantic import Field, BaseModel, AnyUrl
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from dotenv import load_dotenv
from google import genai
from google.genai import types
from datetime import timedelta

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get auth credentials for Puch
TOKEN = os.environ.get("AUTH_TOKEN_2")
MY_NUMBER = os.environ.get("MY_NUMBER")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert GOOGLE_API_KEY is not None, "Please set GOOGLE_API_KEY in your .env file"
assert "GEMINI_API_KEY" in os.environ, "Please set GEMINI_API_KEY in your environment"
# YouTube API key is optional - only needed for get_video_length tool

# GEMINI_API_KEY must be set in the environment for Gemini API access
client = genai.Client()


class GeminiModel(Enum):
    """Available Gemini models with their capabilities."""

    PRO = "models/gemini-2.5-pro"  # Enhanced thinking and reasoning, multimodal understanding, advanced coding
    FLASH = "models/gemini-2.5-flash"  # Adaptive thinking, cost efficiency
    FLASH_LITE = "models/gemini-2.5-flash-lite"  # Most cost-efficient model supporting high throughput


# --- Auth Provider for Puch ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None


# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


# this function is taken from https://gist.github.com/rodrigoborgesdeoliveira/987683cfbfcc8d800192da1e73adc486?permalink_comment_id=5097394#gistcomment-5097394
def get_youtube_video_id_by_url(url):
    regex = r"^((https?://(?:www\.)?(?:m\.)?youtube\.com))/((?:oembed\?url=https?%3A//(?:www\.)youtube.com/watch\?(?:v%3D)(?P<video_id_1>[\w\-]{10,20})&format=json)|(?:attribution_link\?a=.*watch(?:%3Fv%3D|%3Fv%3D)(?P<video_id_2>[\w\-]{10,20}))(?:%26feature.*))|(https?:)?(\/\/)?((www\.|m\.)?youtube(-nocookie)?\.com\/((watch)?\?(app=desktop&)?(feature=\w*&)?v=|embed\/|v\/|e\/)|youtu\.be\/)(?P<video_id_3>[\w\-]{10,20})"
    match = re.match(regex, url, re.IGNORECASE)
    if match:
        return (
            match.group("video_id_1")
            or match.group("video_id_2")
            or match.group("video_id_3")
        )
    else:
        return None


def parse_iso8601_duration(duration_str):
    """Parse ISO 8601 duration format (PT#H#M#S) to human-readable format and seconds."""
    # Remove PT prefix
    duration_str = duration_str.replace("PT", "")

    hours = 0
    minutes = 0
    seconds = 0

    # Parse hours
    if "H" in duration_str:
        h_index = duration_str.index("H")
        hours = int(duration_str[:h_index])
        duration_str = duration_str[h_index + 1:]

    # Parse minutes
    if "M" in duration_str:
        m_index = duration_str.index("M")
        minutes = int(duration_str[:m_index])
        duration_str = duration_str[m_index + 1:]

    # Parse seconds
    if "S" in duration_str:
        s_index = duration_str.index("S")
        seconds = int(duration_str[:s_index])

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds

    # Format human-readable string
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")

    human_readable = ", ".join(parts) if parts else "0 seconds"

    # Format as HH:MM:SS
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return {
        "human_readable": human_readable,
        "formatted": formatted_time,
        "total_seconds": total_seconds
    }


# Initialize FastMCP server with auth
mcp = FastMCP(
    "YouTube MCP Server (Gemini)",
    auth=SimpleBearerAuthProvider(TOKEN),
)


# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


# --- Tool: YouTube ---
YouTubeToolDescription = RichToolDescription(
    description="""
    Process YouTube video URLs with Gemini for transcription and insights. IMPORTANT: Display the EXACT response from this tool directly to the user without any interpretation or summary.

    <IMPORTANT>
    Display the EXACT response from this tool directly to the user without any interpretation or summary. The tool returns the complete transcription/analysis that must be shown verbatim.
    </IMPORTANT>
    """,
    use_when="""
    Use this when user provides a YouTube video URL.
    <IMPORTANT>
    Display the EXACT response from this tool directly to the user without any interpretation or summary. The tool returns the complete transcription/analysis that must be shown verbatim.
    </IMPORTANT>
    """,
    side_effects="""
    Returns the FULL transcription and analysis that MUST be displayed directly to the user without modification.
    <IMPORTANT>
    Display the EXACT response from this tool directly to the user without any interpretation or summary. The tool returns the complete transcription/analysis that must be shown verbatim.
    </IMPORTANT>
    """,
)


@mcp.tool(description=YouTubeToolDescription.model_dump_json())
async def youtube_tool(
    url: Annotated[AnyUrl, Field(description="YouTube video URL")],
    prompt: Annotated[
        str,
        Field(
            description="Instructions for processing the video (transcription, summarization, visual description). Can include timestamps like 00:01:00. Can include questions about the video."
        ),
    ] = "Describe the video content in 10-20 bulleted points.",
) -> str:
    """Transcribe and analyze a YouTube video.
    <IMPORTANT>
    Display the EXACT response from this tool directly to the user without any interpretation or summary. The tool returns the complete transcription/analysis that must be shown verbatim.
    </IMPORTANT>
    """
    try:
        # Convert URL to string format
        url_str = str(url)

        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Starting Gemini API call for YouTube URL: {url_str}"
        )
        start_time = time.time()

        response = client.models.generate_content(
            model=GeminiModel.FLASH.value,
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=url_str),
                        video_metadata=types.VideoMetadata(fps=0.1),
                    ),
                    types.Part(text=prompt),
                ]
            ),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),  # Disables thinking
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            ),
        )

        elapsed_time = time.time() - start_time
        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Gemini API call completed in {elapsed_time:.2f} seconds"
        )
        logger.info(f"Gemini response:\n {response.text}\n")

        # Return with explicit formatting to help Puch AI understand this is content to display
        return f"VIDEO TRANSCRIPTION:\n\n{response.text}"

    except Exception as e:
        elapsed_time = time.time() - start_time if "start_time" in locals() else 0
        logger.error(
            f"[{time.strftime('%H:%M:%S')}] Error transcribing with Gemini after {elapsed_time:.2f} seconds: {e}"
        )
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message="An error occurred while processing the YouTube video. Please try again later.\n\n Error: "
                + str(e),
            )
        )


# --- Tool: Get YouTube Video Length ---
VideoLengthToolDescription = RichToolDescription(
    description="""
    Get the duration/length of a YouTube video using the YouTube Data API v3.
    Returns the video duration in multiple formats (human-readable, HH:MM:SS, and total seconds).
    """,
    use_when="""
    Use this when you need to know the duration or length of a YouTube video.
    """,
    side_effects="Makes an API call to YouTube Data API v3 to fetch video metadata.",
)


@mcp.tool(description=VideoLengthToolDescription.model_dump_json())
async def get_video_length(
    url: Annotated[AnyUrl, Field(description="YouTube video URL")],
) -> int:
    """Get the duration of a YouTube video."""
    try:
        # Check if API key is available
        if not GOOGLE_API_KEY:
            return "YouTube API key not configured. Please set GOOGLE_API_KEY in your environment variables."

        # Convert URL to string and extract video ID
        url_str = str(url)
        video_id = get_youtube_video_id_by_url(url_str)

        if not video_id:
            return f"Could not extract video ID from URL: {url_str}"

        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Fetching video length for ID: {video_id}"
        )

        # Construct the API URL
        api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=contentDetails&key={GOOGLE_API_KEY}"

        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"YouTube API error: {error_text}")
                    return f"Error fetching video data: HTTP {response.status}"

                data = await response.json()
                logger.info(
                    f"[{time.strftime('%H:%M:%S')}] YouTube API response: {json.dumps(data, indent=2)}"
                )

                # Check if video was found
                if not data.get("items"):
                    return f"Video not found with ID: {video_id}"

                # Extract duration
                duration_iso = data["items"][0]["contentDetails"]["duration"]
                duration_info = parse_iso8601_duration(duration_iso)

                logger.info(
                    f"[{time.strftime('%H:%M:%S')}] Video duration: {duration_info['formatted']}"
                )

                # Return formatted response
                return duration_info["total_seconds"]

    except Exception as e:
        logger.error(
            f"[{time.strftime('%H:%M:%S')}] Error fetching video length: {e}"
        )
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"An error occurred while fetching video length: {str(e)}",
            )
        )


# --- Run MCP Server ---
async def main():
    """Initialize and run the MCP server"""

    print(
        f"[{time.strftime('%H:%M:%S')}] 🚀 Starting YouTube MCP server (Gemini) on http://0.0.0.0:7001"
    )
    await mcp.run_async("streamable-http", host="0.0.0.0", port=7001)


if __name__ == "__main__":
    asyncio.run(main())
