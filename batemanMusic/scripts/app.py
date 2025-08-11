import subprocess
import os
import yt_dlp
import random

from ytmusic_thumbnail import get_ytmusic_thumbnail
from youtube_thumbnail import get_yt_thumbnail
from spotify_thumbnail import get_spotify_thumbnail


# def generate_output_video_user_upload(bg_image_path: str) -> str:
#     """Generates the final video using a bash script that uses ffmpeg

#     Args:
#         background_temp_file_path (str): path to the background image

#     Returns:
#         str: path to the output video
#     """
#     if bg_image_path in st.session_state["video_state"]:
#         print("Using already generated video")
#         return st.session_state["video_state"][bg_image_path]

#     result = subprocess.run(
#         ["scripts/generateVideoUserUpload.sh", bg_image_path], capture_output=True
#     )
#     output_video_path = str(result.stdout).lstrip("b'").rstrip("\\n'")
#     st.session_state["video_state"][bg_image_path] = output_video_path

#     return output_video_path


# def generate_output_video_spotify(bg_image_path: str) -> str:
#     """Generates the final video using a bash script that uses ffmpeg

#     Args:
#         background_temp_file_path (str): path to the background image

#     Returns:
#         str: path to the output video
#     """
#     if bg_image_path in st.session_state["video_state"]:
#         print("Using already generated video")
#         return st.session_state["video_state"][bg_image_path]

#     result = subprocess.run(
#         ["scripts/generateVideoSpotify.sh", bg_image_path], capture_output=True
#     )
#     output_video_path = str(result.stdout).lstrip("b'").rstrip("\\n'")
#     st.session_state["video_state"][bg_image_path] = output_video_path

#     return output_video_path


def generate_output_video_ytmusic(bg_image_path: str) -> str:
    """Generates the final video using a bash script that uses ffmpeg

    Args:
        background_temp_file_path (str): path to the background image

    Returns:
        str: path to the output video
    """

    result = subprocess.run(
        ["batemanMusic/scripts/generateVideoYTMusic.sh", bg_image_path],
        capture_output=True,
    )
    output_video_path = str(result.stdout).lstrip("b'").rstrip("\\n'")

    return output_video_path


def generate_output_video_youtube(bg_image_path: str) -> str:
    """Generates the final video using a bash script that uses ffmpeg. It uses a landscape bateman video

    Args:
        background_temp_file_path (str): path to the background image

    Returns:
        str: path to the output video
    """

    result = subprocess.run(
        ["batemanMusic/scripts/generateVideoYouTube.sh", bg_image_path],
        capture_output=True,
    )
    output_video_path = str(result.stdout).lstrip("b'").rstrip("\\n'")

    return output_video_path


# def download_song_spotify(url: str) -> str:
#     """downloads song using spotify_dl

#     Args:
#         url (str): url to track on spotify

#     Returns:
#         str: path of the downloaded song
#     """
#     # if the song is already downloaded, no need to download again.
#     if url in st.session_state["song_state"]:
#         print("Using already downloaded song: spotify")
#         return st.session_state["song_state"][url]

#     # spotify_dl downloads the song, and outputs a lot of things on stdout, among them is save location
#     os.makedirs("./batemanMusic/audio/spotify", exist_ok=True)
#     result = subprocess.run(
#         ["spotify_dl", "-l", url, "-o", "./batemanMusic/audio/spotify", "-m"], capture_output=True
#     )
#     pattern = r"\[download\] Destination: (.+?)\n|\[download\] (.+?) has already been downloaded\n"
#     match = re.search(pattern, result.stdout.decode("utf-8"))
#     if match:
#         song_path = match.group(1) if match.group(1) else match.group(2)
#         st.session_state["song_state"][url] = song_path
#         print(song_path)
#         return song_path


def download_song_youtube(url: str) -> str:
    """downloads video using yt_dlp and extracts audio from it.

    Args:
        url (str): url to song on youtube

    Returns:
        str: path of the downloaded song
    """

    output_location = {}

    def hook(d):
        if d["status"] == "finished":
            output_location["filename"] = d["filename"]

    os.makedirs("./batemanMusic/audio/youtube", exist_ok=True)
    urls = [url]

    ydl_opts = {
        "format": "m4a/bestbatemanMusic/audio/best",
        "postprocessors": [
            {  # Extract audio using ffmpeg
                "key": "FFmpegExtractAudio",
                "preferredcodec": "aac",
            }
        ],
        "outtmpl": os.path.join(
            "./batemanMusic/audio/youtube", "%(title)s.%(ext)s"
        ),  # Set the output directory and filename template
        "progress_hooks": [hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(urls)
        print("youtube-dl error code: ", error_code)
    print("output_location: ", output_location)

    # After audio extraction, the file extension changes from mp4 to m4a
    filename = output_location.get("filename", "Download failed or file not found")
    if filename.endswith(".mp4"):
        filename = filename.replace(".mp4", ".m4a")

    return filename


def download_song_ytmusic(url: str) -> str:
    """downloads video using yt_dlp and extracts audio from it.

    Args:
        url (str): url to song on youtube music

    Returns:
        str: path of the downloaded song
    """

    output_location = {}

    def hook(d):
        if d["status"] == "finished":
            output_location["filename"] = d["filename"]

    os.makedirs("./batemanMusic/audio/ytmusic", exist_ok=True)
    urls = [url]

    ydl_opts = {
        "format": "m4a/bestbatemanMusic/audio/best",
        "postprocessors": [
            {  # Extract audio using ffmpeg
                "key": "FFmpegExtractAudio",
                "preferredcodec": "aac",
            }
        ],
        "outtmpl": os.path.join(
            "./batemanMusic/audio/ytmusic", "%(title)s.%(ext)s"
        ),  # Set the output directory and filename template
        "progress_hooks": [hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        error_code = ydl.download(urls)
        print("youtube-dl error code: ", error_code)

    # After audio extraction, the file extension changes from mp4 to m4a
    filename = output_location.get("filename", "Download failed or file not found")
    if filename.endswith(".mp4"):
        filename = filename.replace(".mp4", ".m4a")

    return filename


def get_song_duration(song_path: str) -> float:
    """finds the duration of a song in seconds

    Args:
        song_path (str): path to song

    Returns:
        float: song duration in seconds
    """
    print("song_path: ", song_path)
    result = subprocess.run(
        [
            "ffprobe",
            "-i",
            song_path,
            "-show_entries",
            "format=duration",
            "-v",
            "quiet",
            "-of",
            "csv=p=0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print("ffprobe stderr: ", result)
    duration_in_seconds = result.stdout.strip()
    print("duration: ", duration_in_seconds)
    return float(duration_in_seconds)


def combine_audio_video(
    output_video_path: str, song_path: str, delay_in_seconds: int
) -> str:
    """combines the given audio and video with delay added from start of the audio.

    Args:
        output_video_path (str): path to the video
        song_path (str): path to the audio
        delay_in_seconds (int): a delay of random seconds to make the audio start from the interesting part of the song

    Returns:
        str: _description_
    """
    result = subprocess.run(
        [
            "batemanMusic/scripts/combineAudioVideo.sh",
            song_path,
            output_video_path,
            str(delay_in_seconds),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    final_video_path = result.stdout.strip()
    print("final_path: ", final_video_path)
    return final_video_path


def get_final_video_ytmusic(ytmusic_url: str) -> str:
    """Generates the final video using a bash script that uses ffmpeg

    Args:
        ytmusic_url (str): YouTube Music URL to the song

    Returns:
        str: path to the output video
    """
    bg_image_path = get_ytmusic_thumbnail(ytmusic_url)
    output_video_path = generate_output_video_ytmusic(bg_image_path)
    song_path = download_song_ytmusic(ytmusic_url)

    delay_in_seconds = 0
    if song_path is not None:
        try:
            delay_in_seconds = get_song_duration(song_path) - random.randint(
                0, get_song_duration(song_path)
            )
        except Exception as e:
            delay_in_seconds = 0

    final_video_path = combine_audio_video(
        output_video_path, song_path, delay_in_seconds
    )
    return final_video_path


def get_final_video_youtube(youtube_url: str) -> str:
    """Generates the final video using a bash script that uses ffmpeg

    Args:
        youtube_url (str): YouTube URL to the song

    Returns:
        str: path to the output video
    """
    bg_image_path = get_yt_thumbnail(youtube_url)
    output_video_path = generate_output_video_youtube(bg_image_path)
    song_path = download_song_youtube(youtube_url)

    delay_in_seconds = 0
    if song_path is not None:
        try:
            delay_in_seconds = get_song_duration(song_path) - random.randint(
                0, get_song_duration(song_path)
            )
        except Exception as e:
            delay_in_seconds = 0

    final_video_path = combine_audio_video(
        output_video_path, song_path, delay_in_seconds
    )
    return final_video_path


if __name__ == "__main__":
    # get_song_duration(
    #     "/home/malik/Documents/Programming/puch/emceepee/batemanMusic/audio/youtube/Kanye West - Good Morning.m4a"
    # )
    print(
        get_final_video_ytmusic(
            "https://music.youtube.com/watch?v=qU9mHegkTc4"
        )
    )
