import os
import logging
import asyncio
from textwrap import dedent
from enum import Enum
from typing import Annotated, List
from pydantic import Field, BaseModel, AnyUrl
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR, ImageContent
from dotenv import load_dotenv

# Import the music memes functionality
try:
    from app import (
        generate_meme_image_from_url,
        generate_meme_image_from_urls,
        generate_meme_image_from_ytmusic_url,
        generate_meme_image_from_ytmusic_urls,
        generate_meme_image_from_spotify_url,
        generate_meme_image_from_spotify_urls,
        generate_meme_image_from_youtube_url,
        generate_meme_image_from_youtube_urls,
        detect_platform,
    )
except ImportError:
    # If running from different directory, try absolute imports
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from app import (
        generate_meme_image_from_url,
        generate_meme_image_from_urls,
        generate_meme_image_from_ytmusic_url,
        generate_meme_image_from_ytmusic_urls,
        generate_meme_image_from_spotify_url,
        generate_meme_image_from_spotify_urls,
        generate_meme_image_from_youtube_url,
        generate_meme_image_from_youtube_urls,
        detect_platform,
    )

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get auth credentials for Puch
TOKEN = os.environ.get("AUTH_TOKEN_2")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN_2 in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"


def _encode_image(image) -> ImageContent:
    """
    Encodes a PIL Image to a format compatible with ImageContent.
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_obj = Image(data=img_bytes, format="png")
    return img_obj.to_image_content()


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
    "Music Memes MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)


# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


# --- Tool: about (recommended by Puch) ---
@mcp.tool
async def about() -> dict[str, str]:
    server_name = "Music Memes MCP Server"
    server_description = dedent(
        """
        This MCP server creates music memes by generating composite images from music URLs.
        It automatically detects the platform and downloads thumbnails from YouTube Music, Spotify, and YouTube,
        then overlays them onto background templates to create meme-style images.

        Features:
        - Generate memes from single music URLs (YT Music, Spotify, YouTube)
        - Generate memes from multiple music URLs (up to 5, mixed platforms supported)
        - Automatic platform detection and thumbnail extraction
        - Image composition with random background selection
        - Support for mixed platform URLs in a single meme
        """
    )

    return {"name": server_name, "description": server_description}


# --- Tool: Single Meme Generation ---
SingleMemeToolDescription = RichToolDescription(
    description="""
    Generate a meme image from a single music URL. Automatically detects platform (YouTube Music, Spotify, YouTube) 
    and downloads the thumbnail, then overlays it onto a random background template.
    """,
    use_when="""
    Use this when the user provides a single music URL (from YouTube Music, Spotify, or YouTube) 
    and wants to create a meme image from it.
    """,
    side_effects="""
    Downloads thumbnail image, creates composite image, saves to outputs directory.
    """,
)


@mcp.tool(description=SingleMemeToolDescription.model_dump_json())
async def generate_single_meme(
    url: Annotated[
        AnyUrl, Field(description="Music URL for a song, album, or playlist (YouTube Music, Spotify, or YouTube)")
    ],
) -> ImageContent:
    """Generate a meme image from a single music URL (auto-detects platform)."""
    try:
        logger.info(f"Generating meme from single URL: {url}")

        # Convert URL to string format
        url_str = str(url)
        
        # Detect platform
        platform = detect_platform(url_str)
        logger.info(f"Detected platform: {platform}")

        # Generate meme using the generic function
        meme_image = generate_meme_image_from_url(url_str)

        if meme_image:
            logger.info(f"Successfully generated meme from URL: {url_str}")
            return _encode_image(meme_image)
        else:
            logger.error(f"Failed to generate meme from URL: {url_str}")
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to generate meme from {platform} URL. This could be due to:\n- Invalid or unsupported URL\n- Thumbnail download failure\n- Missing background templates\n- File system issues",
                )
            )

    except Exception as e:
        logger.error(f"Error generating single meme: {e}")
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"An error occurred while generating the meme: {str(e)}",
            )
        )


# --- Tool: Multiple Memes Generation ---
MultipleMemeToolDescription = RichToolDescription(
    description="""
    Generate a single meme image from multiple music URLs (maximum 5). Automatically detects platforms 
    (YouTube Music, Spotify, YouTube) for each URL, downloads thumbnails, and composites them onto 
    a background template designed for multiple images. Mixed platforms are supported.
    """,
    use_when="""
    Use this when the user provides multiple music URLs (2-5) from any supported platform 
    and wants to create a single meme image containing all of them.
    """,
    side_effects="""
    Downloads multiple thumbnail images, creates single composite image with all thumbnails, saves to outputs directory.
    """,
)


@mcp.tool(description=MultipleMemeToolDescription.model_dump_json())
async def generate_multiple_memes(
    urls: Annotated[
        List[AnyUrl],
        Field(
            description="List of music URLs (maximum 5) from YouTube Music, Spotify, or YouTube",
            min_length=1,
            max_length=5,
        ),
    ],
) -> ImageContent:
    """Generate a single meme image from multiple music URLs (max 5, mixed platforms supported)."""
    try:
        logger.info(f"Generating meme from {len(urls)} URLs: {urls}")

        # Convert URLs to string format and detect platforms
        url_strs = [str(url) for url in urls]
        platforms = [detect_platform(url) for url in url_strs]
        logger.info(f"Detected platforms: {platforms}")

        # Generate meme using the generic function
        meme_image = generate_meme_image_from_urls(url_strs)

        if meme_image:
            logger.info(f"Successfully generated meme from {len(url_strs)} URLs")
            return _encode_image(meme_image)
        else:
            logger.error(f"Failed to generate meme from URLs: {url_strs}")
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to generate meme from multiple URLs. This could be due to:\n- Invalid or unsupported URLs\n- Thumbnail download failures\n- Missing background templates for {len(url_strs)} images\n- File system issues",
                )
            )

    except ValueError as e:
        logger.error(f"Invalid input for multiple memes: {e}")
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=str(e),
            )
        )
    except Exception as e:
        logger.error(f"Error generating multiple memes: {e}")
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"An error occurred while generating the meme: {str(e)}",
            )
        )


# --- Run MCP Server ---
async def main():
    """Initialize and run the MCP server"""
    print(f"🚀 Starting Music Memes MCP server on http://0.0.0.0:7002")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=7002)


if __name__ == "__main__":
    asyncio.run(main())
