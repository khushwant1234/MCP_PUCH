import os
import random
from PIL import Image
from maker import apply_overlay_transformation, apply_overlay_transformation_v2, apply_overlay_transformation_image, apply_overlay_transformation_v2_image
from ytmusic_thumbnail import get_ytmusic_thumbnail


def generate_meme_from_url(ytmusic_url):
    """
    Generate a meme image from a single YT Music URL.

    Args:
        ytmusic_url (str): YT Music URL for a song or playlist

    Returns:
        str: Path to the generated meme image, or None if generation failed
    """
    # Get thumbnail from URL
    overlay_path = get_ytmusic_thumbnail(ytmusic_url)
    if not overlay_path:
        return None

    # Get random background for single image
    backgrounds_directory = os.path.join("music-memes", "assets", "background", "1")
    if not os.path.exists(backgrounds_directory):
        return None

    background_files = [f for f in os.listdir(backgrounds_directory)
                       if os.path.isfile(os.path.join(backgrounds_directory, f))]

    if not background_files:
        return None

    # Pick random background
    background_filename = random.choice(background_files)
    background_path = os.path.join(backgrounds_directory, background_filename)

    # Generate meme
    return apply_overlay_transformation(background_path, overlay_path)


def generate_meme_from_urls(ytmusic_urls):
    """
    Generate a single meme image from a list of YT Music URLs (max 5).

    Args:
        ytmusic_urls (list): List of YT Music URLs (max length 5)

    Returns:
        str: Path to the generated meme image, or None if generation failed
    """
    if len(ytmusic_urls) > 5:
        raise ValueError("Maximum 5 URLs allowed")

    if len(ytmusic_urls) == 0:
        return None

    # Get thumbnails from URLs
    overlay_paths = [path for url in ytmusic_urls if (path := get_ytmusic_thumbnail(url)) is not None]

    if not overlay_paths:
        return None

    n = len(overlay_paths)
    backgrounds_directory = os.path.join("music-memes", "assets", "background", str(n))

    if not os.path.exists(backgrounds_directory):
        return None

    # Get random background
    background_files = [f for f in os.listdir(backgrounds_directory)
                       if os.path.isfile(os.path.join(backgrounds_directory, f))]

    if not background_files:
        return None

    background_filename = random.choice(background_files)
    background_path = os.path.join(backgrounds_directory, background_filename)

    # Generate single meme
    if n == 1:
        return apply_overlay_transformation(background_path, overlay_paths[0])
    else:
        return apply_overlay_transformation_v2(background_path, overlay_paths)


if __name__ == "__main__":
    # Example usage
    # url = "https://music.youtube.com/watch?v=nyuo9-OjNNg&si=mOMlPjr16WNVxWto"
    # print(generate_meme_from_url(url))

    urls = [
        "https://music.youtube.com/watch?v=nyuo9-OjNNg&si=mOMlPjr16WNVxWto",
        "https://music.youtube.com/watch?v=sEetXo3R-aM&si=zEbt0ZqGo_HHwQJ8",
    ]
    print(generate_meme_from_urls(urls))


def generate_meme_image_from_url(ytmusic_url):
    """
    Generate a meme image from a single YT Music URL.

    Args:
        ytmusic_url (str): YT Music URL for a song or playlist

    Returns:
        PIL.Image: Generated meme image, or None if generation failed
    """
    # Get thumbnail from URL
    overlay_path = get_ytmusic_thumbnail(ytmusic_url)
    if not overlay_path:
        return None

    # Get random background for single image
    backgrounds_directory = os.path.join("music-memes", "assets", "background", "1")
    if not os.path.exists(backgrounds_directory):
        return None

    background_files = [f for f in os.listdir(backgrounds_directory)
                       if os.path.isfile(os.path.join(backgrounds_directory, f))]

    if not background_files:
        return None

    # Pick random background
    background_filename = random.choice(background_files)
    background_path = os.path.join(backgrounds_directory, background_filename)

    # Generate meme and return PIL Image
    return apply_overlay_transformation_image(background_path, overlay_path)


def generate_meme_image_from_urls(ytmusic_urls):
    """
    Generate a single meme image from a list of YT Music URLs (max 5).

    Args:
        ytmusic_urls (list): List of YT Music URLs (max length 5)

    Returns:
        PIL.Image: Generated meme image, or None if generation failed
    """
    if len(ytmusic_urls) > 5:
        raise ValueError("Maximum 5 URLs allowed")

    if len(ytmusic_urls) == 0:
        return None

    # Get thumbnails from URLs
    overlay_paths = [path for url in ytmusic_urls if (path := get_ytmusic_thumbnail(url)) is not None]

    if not overlay_paths:
        return None

    n = len(overlay_paths)
    backgrounds_directory = os.path.join("music-memes", "assets", "background", str(n))

    if not os.path.exists(backgrounds_directory):
        return None

    # Get random background
    background_files = [f for f in os.listdir(backgrounds_directory)
                       if os.path.isfile(os.path.join(backgrounds_directory, f))]

    if not background_files:
        return None

    background_filename = random.choice(background_files)
    background_path = os.path.join(backgrounds_directory, background_filename)

    # Generate single meme and return PIL Image
    if n == 1:
        return apply_overlay_transformation_image(background_path, overlay_paths[0])
    else:
        return apply_overlay_transformation_v2_image(background_path, overlay_paths)
