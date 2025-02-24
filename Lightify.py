import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

import board
import neopixel
import os

# LED strip configuration
pixel_pin = board.D18
num_pixels = 300
ORDER = neopixel.GRB
pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=0.2, auto_write=False, pixel_order=ORDER)

def adjust_gamma(color, gamma=0.5):
    # Apply inverse gamma correction to each RGB channel
    return tuple(int((channel / 255.0) ** (1 / gamma) * 255) for channel in color)

def fade_to_color(target_colors, steps, delay):
    #checks if current pixel colors hold values less than or greater to new colors
    #adjusts colors accordingly
    #maps the colors on each increment / decrement
    return

def map_colors(colors):
    """Assigns colors to the LED strip in groups and updates the strip."""
    corrected_colors = [adjust_gamma(tuple(color)) for color in colors]
    group_size = 10  # How many LEDs get the same color in sequence
    num_colors = len(corrected_colors)
    for i in range(num_pixels):
        group_index = (i // group_size) % num_colors
        pixels[i] = corrected_colors[group_index]
    pixels.show()

def get_unique_dominant_colors(image_url, num_colors=3, brightness_threshold=30, uniqueness_threshold=250):
    """
    Downloads an image from a URL, filters out dark pixels, and uses KMeans clustering 
    to extract a set of unique dominant colors.
    """
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download image, status code: {response.status_code}")

    content_type = response.headers.get('Content-Type', '').lower()
    if 'image' not in content_type:
        raise Exception(f"URL did not return an image. Content-Type: {content_type}")

    response.raw.decode_content = True
    try:
        img = Image.open(response.raw)
    except Exception as e:
        raise Exception(f"Error identifying image file: {e}")

    img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    pixels_arr = img_array.reshape(-1, 3)

    # Filter out dark pixels
    bright_pixels = []
    for pixel in pixels_arr:
        brightness = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
        if brightness >= brightness_threshold:
            bright_pixels.append(pixel)
    bright_pixels = np.array(bright_pixels)

    if len(bright_pixels) == 0:
        raise Exception("No bright pixels found after filtering.")

    # Use KMeans to get candidate colors
    num_clusters = max(num_colors * 2, 6)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(bright_pixels)
    centers = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(labels)
    clusters = [{'center': center, 'count': count} for center, count in zip(centers, counts)]
    clusters.sort(key=lambda x: x['count'], reverse=True)

    def color_distance(c1, c2):
        return np.linalg.norm(np.array(c1) - np.array(c2))

    unique_colors = []
    for cluster in clusters:
        candidate = cluster['center']
        if not unique_colors:
            unique_colors.append(candidate)
        else:
            if all(color_distance(candidate, selected) >= uniqueness_threshold for selected in unique_colors):
                unique_colors.append(candidate)
        if len(unique_colors) >= num_colors:
            break

    # Fallback: If not enough unique colors, add more regardless
    if len(unique_colors) < num_colors:
        for cluster in clusters:
            candidate = cluster['center']
            if not any(np.array_equal(candidate, col) for col in unique_colors):
                unique_colors.append(candidate)
            if len(unique_colors) >= num_colors:
                break

    return np.array(unique_colors[:num_colors])

def create_spotify_oauth():
    """
    Returns a SpotifyOAuth instance configured with your credentials.
    The token will be cached in a file (e.g. ".cache-<username>") for reuse.
    """
    return SpotifyOAuth(
        client_id='810dec886bd14815a165fe7390266210',
        client_secret='09f03b66b73e4a548235e05853c70702',
        redirect_uri='http://localhost:8888/callback',
        scope='user-read-currently-playing user-read-playback-state'
    )

def get_token(spotify_oauth):
    """
    Attempts to get a cached token; if none exists, prompts the user to authenticate.
    """
    token_info = spotify_oauth.get_cached_token()
    if not token_info:
        auth_url = spotify_oauth.get_authorize_url()
        print("Please navigate to the following URL and authorize access:")
        print(auth_url)
        # In a headless setup, copy the URL to a browser on another device
        redirect_response = input("Paste the full redirect URL here: ")
        # Extract the code from the URL and fetch the token
        token_info = spotify_oauth.get_access_token(redirect_response.split("code=")[-1])
    now = int(time.time())
    if token_info['expires_at'] - now < 60:
        token_info = spotify_oauth.refresh_access_token(token_info['refresh_token'])
    return token_info

def main_loop():
    spotify_oauth = create_spotify_oauth()
    token_info = get_token(spotify_oauth)
    sp = spotipy.Spotify(auth=token_info['access_token'])
    last_track_id = None

    while True:
        try:
            current = sp.currently_playing()
            if current and current.get('item'):
                track_id = current['item']['id']
                if track_id != last_track_id:
                    last_track_id = track_id
                    album_images = current['item']['album']['images']
                    image_url = None
                    # Prefer the 300x300 image if available
                    for img in album_images:
                        if img.get('height') == 300 and img.get('width') == 300:
                            image_url = img['url']
                            break
                    if not image_url and album_images:
                        image_url = album_images[-1]['url']
                    
                    print("Now playing:", current['item']['name'])
                    try:
                        dominant_colors = get_unique_dominant_colors(image_url)
                        colors = dominant_colors.tolist()
                        map_colors(colors)
                        print("LEDs updated with colors:", colors)
                    except Exception as e:
                        print("Error processing album art:", e)
            else:
                print("No song currently playing or no album art available.")
        except Exception as e:
            print("Error fetching playback data:", e)
        
        time.sleep(0.1)  # Poll every 5 seconds

if __name__ == '__main__':
    main_loop()
