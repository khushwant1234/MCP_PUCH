import requests
from PIL import Image
import os
from io import BytesIO
import re

# youtube music thumbnails can also be grabbed in this manner


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


# if yt link if provided, generate two videos: square and landscape
def get_yt_thumbnail(url: str) -> None | str:
    """Get the thumbnail of a song using its youtube music url

    Args:
        url (str): YouTube url to the song

    Returns:
        str | None: path to the downloaded thumbail image file
    """
    video_id: str = get_youtube_video_id_by_url(url)
    save_name = video_id
    save_path = os.path.join(
        "./music-memes/assets/thumbnails/youtube", f"{save_name}.jpg"
    )
    os.makedirs("./music-memes/assets/thumbnails/youtube", exist_ok=True)

    # check if this thumbnail is already available, and return it if it exists
    try:
        with open(save_path) as im:
            print("using already generated youtube thumbnail")
        return save_path
    except:
        pass
    thumbnail_url: str = "https://img.youtube.com/vi/" + video_id + "/maxresdefault.jpg"

    rr = requests.get(thumbnail_url)
    if rr.status_code != 200:
        return None
    with Image.open(BytesIO(rr.content)) as im:
        try:
            # Crop from center to make it square, then resize to 512x512
            width, height = im.size
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            right = left + size
            bottom = top + size
            
            cropped_im = im.crop((left, top, right, bottom))
            resized_im = cropped_im.resize((512, 512), Image.Resampling.LANCZOS)
            resized_im.save(save_path, format="JPEG")
            return save_path
        except:
            return None


if __name__ == "__main__":
    get_yt_thumbnail("https://www.youtube.com/watch?v=6CHs4x2uqcQ")
