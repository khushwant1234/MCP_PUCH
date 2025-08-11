#!/usr/bin/env python3
"""
Test script to verify multi-platform music meme generation functionality.
"""

from app import (
    detect_platform,
    generate_meme_image_from_url,
    generate_meme_image_from_urls
)

# Test URLs for different platforms
test_urls = {
    'ytmusic': 'https://music.youtube.com/watch?v=sEetXo3R-aM&si=zEbt0ZqGo_HHwQJ8',
    'spotify': 'https://open.spotify.com/track/1gqkRc9WtOpnGIqxf2Hvzr?si=bd0351766c9b496f',
    'youtube': 'https://www.youtube.com/watch?v=6CHs4x2uqcQ'
}

def test_platform_detection():
    """Test platform detection for different URLs."""
    print("Testing platform detection:")
    for platform, url in test_urls.items():
        detected = detect_platform(url)
        print(f"  {platform}: {url}")
        print(f"  Detected: {detected}")
        print(f"  Correct: {'✓' if detected == platform else '✗'}")
        print()

def test_single_meme_generation():
    """Test single meme generation for each platform."""
    print("Testing single meme generation:")
    for platform, url in test_urls.items():
        print(f"  Testing {platform}: {url}")
        try:
            meme = generate_meme_image_from_url(url)
            if meme:
                print(f"  ✓ Successfully generated meme for {platform}")
                # Optionally save the test image
                # meme.save(f"test_{platform}_meme.png")
            else:
                print(f"  ✗ Failed to generate meme for {platform}")
        except Exception as e:
            print(f"  ✗ Error generating meme for {platform}: {e}")
        print()

def test_mixed_platform_meme():
    """Test generating a meme from mixed platform URLs."""
    print("Testing mixed platform meme generation:")
    mixed_urls = list(test_urls.values())[:3]  # Take first 3 URLs
    print(f"  URLs: {mixed_urls}")
    
    try:
        meme = generate_meme_image_from_urls(mixed_urls)
        if meme:
            print("  ✓ Successfully generated mixed platform meme")
            # Optionally save the test image
            # meme.save("test_mixed_meme.png")
        else:
            print("  ✗ Failed to generate mixed platform meme")
    except Exception as e:
        print(f"  ✗ Error generating mixed platform meme: {e}")
    print()

if __name__ == "__main__":
    print("Music Memes Multi-Platform Test")
    print("=" * 40)
    
    test_platform_detection()
    test_single_meme_generation()
    test_mixed_platform_meme()
    
    print("Test completed!")
