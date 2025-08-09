import os
import csv
import uuid
import asyncio
import logging
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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get auth credentials for Puch
TOKEN = os.environ.get("AUTH_TOKEN_2")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert "GEMINI_API_KEY" in os.environ, "Please set GEMINI_API_KEY in your environment"

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
    description="Process YouTube video URLs with Gemini for transcription and insights. The prompt can contain any instructions (transcription, summarization, visual description). It can also understand timestamps (e.g. 00:01:00).",
    use_when="Use this when user provides a YouTube video URL.",
    side_effects="Returns insights, transcriptions, and visual descriptions of the video.",
)


@mcp.tool(description=YouTubeToolDescription.model_dump_json())
async def youtube_tool(
    url: Annotated[AnyUrl, Field(description="YouTube video URL")],
    prompt: Annotated[
        str,
        Field(
            description="Instructions for processing the video (transcription, summarization, visual description). Can include timestamps like 00:01:00. Can include questions about the video."
        ),
    ] = "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions.",
) -> str:

    try:
        # Convert URL to string format
        url_str = str(url)

        response = client.models.generate_content(
            model=GeminiModel.FLASH_LITE.value,
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=url_str),
                        video_metadata=types.VideoMetadata(fps=1),
                    ),
                    types.Part(text=prompt),
                ]
            ),
        )
        logger.info(f"Gemini response:\n {response.text}\n")
        return response.text

    except Exception as e:
        logger.error(f"Error transcribing with Gemini: {e}")
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message="An error occurred while processing the YouTube video. Please try again later.\n\n Error: "
                + str(e),
            )
        )


# --- Run MCP Server ---
async def main():
    """Initialize and run the MCP server"""

    print(f"🚀 Starting YouTube MCP server (Gemini) on http://0.0.0.0:7001")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=7001)


if __name__ == "__main__":
    asyncio.run(main())
