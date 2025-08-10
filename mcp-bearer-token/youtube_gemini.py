import os
import random
import re
import asyncio
import logging
import time
import aiohttp
import yt_dlp
from enum import Enum
from typing import Annotated
from pydantic import Field, BaseModel, AnyUrl
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get auth credentials for Puch
TOKEN = os.environ.get("AUTH_TOKEN_2")
MY_NUMBER = os.environ.get("MY_NUMBER")
GOOGLE_API_KEY_1 = os.environ.get("GOOGLE_API_KEY_1")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
# YouTube API key is optional - only needed for get_video_length tool

# GEMINI_API_KEY must be set in the environment for Gemini API access
# Support for multiple Gemini API keys for parallel processing
GEMINI_API_KEYS = [
    os.environ.get("GEMINI_API_KEY_1"),
    os.environ.get("GEMINI_API_KEY_2"),
    os.environ.get("GEMINI_API_KEY_3"),
    os.environ.get("GEMINI_API_KEY_4"),
]

# Filter out None values and ensure we have at least one key
GEMINI_API_KEYS = [key for key in GEMINI_API_KEYS if key is not None]
assert (
    len(GEMINI_API_KEYS) >= 1
), "At least GEMINI_API_KEY must be set in your environment"

# Primary client for single API calls
client = genai.Client(
    api_key=GEMINI_API_KEYS[random.randint(0, len(GEMINI_API_KEYS) - 1)]
)

# Create clients for each API key for parallel processing
gemini_clients = [genai.Client(api_key=key) for key in GEMINI_API_KEYS]

# Configuration for parallel processing
PARALLEL_SEGMENTS = 3  # Number of parallel segments to process videos


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
        duration_str = duration_str[h_index + 1 :]

    # Parse minutes
    if "M" in duration_str:
        m_index = duration_str.index("M")
        minutes = int(duration_str[:m_index])
        duration_str = duration_str[m_index + 1 :]

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
        "total_seconds": total_seconds,
    }


async def get_video_length(
    url: Annotated[AnyUrl, Field(description="YouTube video URL")],
) -> int:
    """Get the duration of a YouTube video."""
    try:
        # Check if API key is available
        if not GOOGLE_API_KEY_1:
            logger.error("YouTube API key not configured.")
            raise ValueError(
                "YouTube API key not configured. Please set GOOGLE_API_KEY_1 in your environment variables."
            )

        # Convert URL to string and extract video ID
        url_str = str(url)
        video_id = get_youtube_video_id_by_url(url_str)

        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url_str}")
            raise ValueError(f"Could not extract video ID from URL: {url_str}")

        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Fetching video length for ID: {video_id}"
        )

        # Construct the API URL
        api_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=contentDetails&key={GOOGLE_API_KEY_1}"

        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"YouTube API error: {error_text}")
                    raise ValueError(
                        f"Error fetching video data: HTTP {response.status}"
                    )

                data = await response.json()
                logger.info(
                    f"[{time.strftime('%H:%M:%S')}] Successfully fetched video length from YouTube API"
                )

                # Check if video was found
                if not data.get("items"):
                    raise ValueError(f"Video not found with ID: {video_id}")

                # Extract duration
                duration_iso = data["items"][0]["contentDetails"]["duration"]
                duration_info = parse_iso8601_duration(duration_iso)

                logger.info(
                    f"[{time.strftime('%H:%M:%S')}] Video duration: {duration_info['formatted']}"
                )

                # Return formatted response
                return duration_info["total_seconds"]

    except Exception as e:
        logger.error(f"[{time.strftime('%H:%M:%S')}] Error fetching video length: {e}")
        raise e


async def get_video_subtitles(
    url: Annotated[AnyUrl, Field(description="YouTube video URL")],
    language: Annotated[
        str,
        Field(
            description="Language code for subtitles (e.g., 'en' for English, 'es' for Spanish). Use 'auto' to get the first available subtitle."
        ),
    ] = "en",
) -> str:
    """Extract subtitles from a YouTube video using yt-dlp Python API."""
    try:
        # Convert URL to string
        url_str = str(url)

        start_time = time.time()
        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Fetching subtitles for URL: {url_str}"
        )

        # Configure yt-dlp options
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [language] if language != "auto" else None,
            "subtitlesformat": "srt/vtt/best",
        }

        # Extract video info without downloading
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_str, download=False)

            # Get video title
            video_title = info.get("title", "Unknown Title")

            # Check for subtitles
            subtitles = info.get("subtitles", {})
            automatic_captions = info.get("automatic_captions", {})

            # Determine which subtitles to use
            selected_subs = None
            selected_lang = None
            is_auto = False

            if language == "auto":
                # Try manual subtitles first
                if subtitles:
                    # Get the first available language
                    selected_lang = next(iter(subtitles.keys()))
                    selected_subs = subtitles[selected_lang]
                elif automatic_captions:
                    # Fall back to auto-generated
                    selected_lang = next(iter(automatic_captions.keys()))
                    selected_subs = automatic_captions[selected_lang]
                    is_auto = True
            else:
                # Try the requested language
                if language in subtitles:
                    selected_subs = subtitles[language]
                    selected_lang = language
                elif language in automatic_captions:
                    selected_subs = automatic_captions[language]
                    selected_lang = language
                    is_auto = True
                else:
                    # Try variations of the language code
                    for lang_code in automatic_captions:
                        if lang_code.startswith(language):
                            selected_subs = automatic_captions[lang_code]
                            selected_lang = lang_code
                            is_auto = True
                            break

            if not selected_subs:
                # List available subtitles
                available = []
                if subtitles:
                    available.append(f"Manual subtitles: {', '.join(subtitles.keys())}")
                if automatic_captions:
                    available.append(
                        f"Auto-generated: {', '.join(automatic_captions.keys())}"
                    )

                if available:
                    return (
                        f"No subtitles found for language '{language}'. Available:\n"
                        + "\n".join(available)
                    )
                else:
                    return "No subtitles available for this video."

            # Find the best format (prefer vtt or srt)
            subtitle_url = None
            for sub in selected_subs:
                if sub.get("ext") in ["vtt", "srt"]:
                    subtitle_url = sub.get("url")
                    break

            if not subtitle_url and selected_subs:
                # Use the first available format
                subtitle_url = selected_subs[0].get("url")

            if not subtitle_url:
                return f"Could not find subtitle URL for language '{selected_lang}'"

            # Download the subtitle content
            async with aiohttp.ClientSession() as session:
                async with session.get(subtitle_url) as response:
                    if response.status != 200:
                        return f"Error downloading subtitles: HTTP {response.status}"

                    subtitle_content = await response.text()

            logger.info(
                f"[{time.strftime('%H:%M:%S')}] Successfully downloaded subtitles for language `{selected_lang}`"
            )
            logger.info(
                f"Subtitle Format: {selected_subs[0].get('ext', 'unknown')}, URL: {subtitle_url}"
            )
            logger.info(
                f"[{time.strftime('%H:%M:%S')}] Subtitle content (first 1000 chars): {subtitle_content[:1000]}"
            )

            # Check if we got an M3U8 playlist instead of actual subtitles
            if subtitle_content.startswith("#EXTM3U") or subtitle_content.startswith(
                "#EXT-X-"
            ):
                logger.info(
                    f"[{time.strftime('%H:%M:%S')}] Detected M3U8 playlist, extracting VTT URLs..."
                )

                # Extract VTT URLs from the M3U8 playlist
                vtt_urls = []
                lines = subtitle_content.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("https://") and "fmt=vtt" in line:
                        vtt_urls.append(line)

                if not vtt_urls:
                    return "No VTT URLs found in the subtitle playlist."

                logger.info(
                    f"[{time.strftime('%H:%M:%S')}] Found {len(vtt_urls)} VTT chunks, downloading in parallel..."
                )

                # Download all VTT chunks in parallel
                async def download_chunk(session, url, index):
                    """Download a single VTT chunk."""
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()
                                return (index, content)
                            else:
                                logger.warning(
                                    f"Failed to download chunk {index+1}: HTTP {response.status}"
                                )
                                return (index, None)
                    except Exception as e:
                        logger.warning(f"Error downloading chunk {index+1}: {e}")
                        return (index, None)

                # Create semaphore to limit concurrent downloads (avoid overwhelming the server)
                semaphore = asyncio.Semaphore(10)  # Max 10 concurrent downloads

                async def download_with_semaphore(session, url, index):
                    async with semaphore:
                        return await download_chunk(session, url, index)

                async with aiohttp.ClientSession() as session:
                    # Create download tasks for all chunks
                    tasks = [
                        download_with_semaphore(session, url, i)
                        for i, url in enumerate(vtt_urls)
                    ]

                    # Execute all downloads concurrently
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                # Sort results by index and combine content
                successful_chunks = []
                for result in results:
                    if isinstance(result, tuple) and result[1] is not None:
                        successful_chunks.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Download task failed: {result}")

                # Sort by index to maintain proper order
                successful_chunks.sort(key=lambda x: x[0])

                # Combine all chunk content
                all_subtitle_content = ""
                for index, content in successful_chunks:
                    all_subtitle_content += content + "\n"

                subtitle_content = all_subtitle_content
                logger.info(
                    f"[{time.strftime('%H:%M:%S')}] Successfully downloaded and combined {len(successful_chunks)}/{len(vtt_urls)} VTT chunks"
                )

            # Clean up the subtitle content
            # Remove WEBVTT header
            subtitle_content = re.sub(r"^WEBVTT\n+", "", subtitle_content)

            # Remove timestamp lines (matches both VTT and SRT format)
            subtitle_content = re.sub(
                r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}.*\n",
                "",
                subtitle_content,
            )

            # Remove subtitle numbers (for SRT format)
            subtitle_content = re.sub(
                r"^\d+\n", "", subtitle_content, flags=re.MULTILINE
            )

            # Remove HTML tags
            subtitle_content = re.sub(r"<[^>]+>", "", subtitle_content)

            # Remove duplicate blank lines
            subtitle_content = re.sub(r"\n{3,}", "\n\n", subtitle_content)

            logger.info(
                f"[{time.strftime('%H:%M:%S')}] Successfully extracted subtitles for language `{selected_lang}` in {time.time() - start_time:.2f} seconds"
            )

            return f"""SUBTITLES FOR: {video_title}
Language: {selected_lang}{' (auto-generated)' if is_auto else ''}

{subtitle_content.strip()}"""

    except Exception as e:
        logger.error(f"[{time.strftime('%H:%M:%S')}] Error fetching subtitles: {e}")
        raise e


async def make_parallel_gemini_calls(
    url_str: str, prompt: str, video_length: int
) -> str:
    """Make parallel API calls to Gemini with different API keys and combine results."""

    async def make_single_call(client_gemini, client_index, offset_start, offset_end):
        """Make a single Gemini API call with specific time offsets."""
        try:
            # Add unique timestamp to track when each call actually starts
            unique_start_time = time.time()
            logger.info(
                f"[{time.strftime('%H:%M:%S')}.{int((unique_start_time % 1) * 1000):03d}] STARTING parallel Gemini call {client_index + 1}/{PARALLEL_SEGMENTS} with offsets {offset_start}-{offset_end}"
            )

            # Run the synchronous API call in a thread pool to achieve true parallelism
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client_gemini.models.generate_content(
                    model=GeminiModel.FLASH_LITE.value,
                    contents=types.Content(
                        parts=[
                            types.Part(
                                file_data=types.FileData(file_uri=url_str),
                                video_metadata=types.VideoMetadata(
                                    fps=0.1,
                                    start_offset=offset_start,
                                    end_offset=offset_end,
                                ),
                            ),
                            types.Part(
                                text=f"{prompt} (Segment {client_index + 1}/{PARALLEL_SEGMENTS}: {offset_start} to {offset_end})"
                            ),
                        ]
                    ),
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
                    ),
                ),
            )

            elapsed_time = time.time() - unique_start_time
            logger.info(
                f"[{time.strftime('%H:%M:%S')}.{int((time.time() % 1) * 1000):03d}] COMPLETED parallel Gemini call {client_index + 1}/{PARALLEL_SEGMENTS} in {elapsed_time:.2f} seconds"
            )

            return f"=== SEGMENT {client_index + 1} ({offset_start} to {offset_end}) ===\n{response.text}"

        except Exception as e:
            logger.error(
                f"[{time.strftime('%H:%M:%S')}] Error in parallel call {client_index + 1}: {e}"
            )
            return f"=== SEGMENT {client_index + 1} ({offset_start} to {offset_end}) ===\nError: {str(e)}"

    # Calculate time segments for parallel calls
    segment_duration = video_length // PARALLEL_SEGMENTS

    # Define time offsets for each segment
    segments = []
    for i in range(PARALLEL_SEGMENTS):
        start_time = i * segment_duration
        end_time = (
            (i + 1) * segment_duration if i < PARALLEL_SEGMENTS - 1 else video_length
        )
        segments.append((f"{start_time}s", f"{end_time}s"))

    # Randomly select clients from available ones
    num_clients = len(gemini_clients)
    if num_clients >= PARALLEL_SEGMENTS:
        selected_clients = random.sample(gemini_clients, PARALLEL_SEGMENTS)
        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Randomly selected {PARALLEL_SEGMENTS} API keys out of {num_clients} available"
        )
    else:
        selected_clients = gemini_clients
        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Using all {num_clients} available API keys (fewer than {PARALLEL_SEGMENTS})"
        )

    logger.info(
        f"[{time.strftime('%H:%M:%S')}] About to start {PARALLEL_SEGMENTS} parallel Gemini API calls using randomly selected clients"
    )

    # Create coroutines first, then convert to tasks
    coroutines = []
    for i in range(PARALLEL_SEGMENTS):
        if num_clients >= PARALLEL_SEGMENTS:
            # Use the randomly selected clients
            client_to_use = selected_clients[i]
        else:
            # Cycle through available clients if we have fewer than PARALLEL_SEGMENTS
            client_index = i % num_clients
            client_to_use = gemini_clients[client_index]

        offset_start, offset_end = segments[i]

        # Create coroutine
        coro = make_single_call(client_to_use, i, offset_start, offset_end)
        coroutines.append(coro)

    # Execute all calls in parallel using asyncio.gather
    parallel_start_time = time.time()
    logger.info(
        f"[{time.strftime('%H:%M:%S')}] Executing gather() for parallel calls..."
    )

    results = await asyncio.gather(*coroutines, return_exceptions=True)

    parallel_elapsed_time = time.time() - parallel_start_time
    logger.info(
        f"[{time.strftime('%H:%M:%S')}] All parallel calls completed in {parallel_elapsed_time:.2f} seconds"
    )

    # Combine results
    combined_result = "\n\n".join(
        [
            str(result) if not isinstance(result, Exception) else f"Error: {result}"
            for result in results
        ]
    )

    return f"VIDEO TRANSCRIPTION ({PARALLEL_SEGMENTS} Parallel Segments):\n\n{combined_result}"


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
    ] = "Describe the video content in 5-10 bulleted points. Display the EXACT response from this tool directly to the user without any interpretation or summary. The tool returns the complete transcription/analysis that must be shown verbatim.",
) -> str:
    """Transcribe and analyze a YouTube video.
    <IMPORTANT>
    Display the EXACT response from this tool directly to the user without any interpretation or summary. The tool returns the complete transcription/analysis that must be shown verbatim.
    </IMPORTANT>
    """
    tool_start_time = time.time()
    logger.info(f"[{time.strftime('%H:%M:%S')}] Starting YouTube Tool for URL: {url}")
    try:
        video_length = await get_video_length(url)

        # Convert URL to string format
        url_str = str(url)

        if video_length <= 600:
            logger.info(
                f"[{time.strftime('%H:%M:%S')}] Video length is {video_length} seconds (< 600), using single Gemini API call."
            )

            # Use a single Gemini API call for shorter videos
            gemini_call_start_time = time.time()
            response = client.models.generate_content(
                model=GeminiModel.FLASH.value,
                contents=types.Content(
                    parts=[
                        types.Part(file_data=types.FileData(file_uri=url_str)),
                        types.Part(text=prompt),
                    ]
                ),
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )

            elapsed_time = time.time() - gemini_call_start_time
            logger.info(
                f"[{time.strftime('%H:%M:%S')}] Gemini API call completed in {elapsed_time:.2f} seconds"
            )
            logger.info(f"Gemini response:\n {response.text}\n")
            # Return with explicit formatting to help Puch AI understand this is content to display
            return f"VIDEO TRANSCRIPTION:\n\n{response.text} \n\n Display the EXACT response from this tool directly to the user without any interpretation or summary. The tool returns the complete transcription/analysis that must be shown verbatim."

        elif video_length > 600 and video_length <= 1800:
            logger.info(
                f"[{time.strftime('%H:%M:%S')}] Video length is {video_length} seconds (> 600), using {PARALLEL_SEGMENTS} parallel Gemini API calls."
            )

            # Use parallel API calls for better processing
            result = await make_parallel_gemini_calls(url_str, prompt, video_length)
            return result

        else:
            return f"""Display the following to the user verbatim:
            Video length is {video_length} seconds (> 1800). This tool currently does not support videos longer than 30 minutes. Please provide a shorter video URL. (Yeah we're broke and on the free tier of Gemini API. Please try smol videos 👉👈)"""
            # This is not in use but kept for future use cases
    #         logger.info(
    #             f"[{time.strftime('%H:%M:%S')}] Video length is {video_length} seconds, using subtitles extraction."
    #         )
    #         # Use subtitles extraction for longer videos
    #         subtitles = await get_video_subtitles(url)

    #         if not subtitles:
    #             raise McpError(
    #                 ErrorData(
    #                     code=INTERNAL_ERROR,
    #                     message="No subtitles found for this video.",
    #                 )
    #             )

    #         logger.info(
    #             f"[{time.strftime('%H:%M:%S')}] Successfully extracted subtitles for long video."
    #         )

    #         # Now send the subtitles to Gemini for processing
    #         subtitles_prompt_part = f'"""{subtitles}"""'
    #         instructions_prompt_part = f"Answer the questions/prompt on the basis of the above subtitles from a YouTube video:\n\n{prompt}"
    #         gemini_call_start_time = time.time()
    #         response = client.models.generate_content(
    #             model=GeminiModel.FLASH.value,
    #             contents=types.Content(
    #                 parts=[
    #                     types.Part(text=subtitles_prompt_part),
    #                     types.Part(text=instructions_prompt_part),
    #                 ]
    #             ),
    #             config=types.GenerateContentConfig(
    #                 thinking_config=types.ThinkingConfig(thinking_budget=0),
    #             ),
    #         )

    #         elapsed_time = time.time() - gemini_call_start_time
    #         logger.info(
    #             f"[{time.strftime('%H:%M:%S')}] Gemini API call completed in {elapsed_time:.2f} seconds"
    #         )
    #         logger.info(f"Gemini response:\n {response.text}\n")
    #         # Return with explicit formatting to help Puch AI understand this is content to display
    #         return f"VIDEO TRANSCRIPTION:\n\n{response.text}"

    except Exception as e:
        logger.error(f"[{time.strftime('%H:%M:%S')}] Error processing the video: {e}")
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message="An error occurred while processing the YouTube video. Please try again later.\n\n Error: \n"
                + str(e),
            )
        )
    finally:
        logger.info(
            f"[{time.strftime('%H:%M:%S')}] Finished YouTube Tool for URL: {url} in {time.time() - tool_start_time:.2f} seconds"
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
