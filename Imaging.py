import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, request, url_for, session, redirect, render_template

import requests
from io import BytesIO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  # optional; used for testing locally

import time

app = Flask(__name__)
app.config['SESSION_COOKIE_NAME'] = '' // fill in yourself
app.secret_key = '' // fill in yourself
TOKEN_INFO = 'token_info'

@app.route('/')
def login():
    auth_url = create_spotify_oauth().get_authorize_url()
    return redirect(auth_url)
    
@app.route('/redirect')
def redirect_page():
    session.clear()
    code = request.args.get('code')
    token_info = create_spotify_oauth().get_access_token(code)
    session[TOKEN_INFO] = token_info
    return redirect(url_for('store_current_song', _external=True))

@app.route('/storeCurrentSong')
def store_current_song():
    try:
        token_info = get_token()
    except Exception as e:
        print('User not logged in:', e)
        return redirect("/")
    
    sp = spotipy.Spotify(auth=token_info['access_token'])
    current = sp.currently_playing()
    if current and current.get('item'):
        # Grab the album art URL (prefer the 300x300 image if available)
        album_images = current['item']['album']['images']
        image_url = None
        for img in album_images:
            if img.get('height') == 300 and img.get('width') == 300:
                image_url = img['url']
                break
        if not image_url and album_images:
            image_url = album_images[-1]['url']
        
        # Process dominant colors
        try:
            dominant_colors = get_unique_dominant_colors(image_url)
            # Convert the numpy array to a list so that it can be JSON serializable in the template
            colors = dominant_colors.tolist()
        except Exception as e:
            print("Error processing image:", e)
            colors = []
        return render_template('index.html', image_url=image_url, colors=colors)

    else:
        return "No song is currently playing or no album art available."

def get_token():
    token_info = session.get(TOKEN_INFO, None)
    if not token_info:
        return redirect(url_for('login', _external=False))
    
    now = int(time.time())
    is_expired = token_info['expires_at'] - now < 60
    if is_expired:
        spotify_oauth = create_spotify_oauth()
        token_info = spotify_oauth.refresh_access_token(token_info['refresh_token'])
        session[TOKEN_INFO] = token_info
    return token_info

def create_spotify_oauth(): 
    return SpotifyOAuth(
        client_id='', // fill in from spotify developer dashboard
        client_secret='', // fill in from spotify developer dashboard
        redirect_uri=url_for('redirect_page', _external=True),
        scope='user-read-currently-playing user-read-playback-state'
    )

def get_unique_dominant_colors(image_url, num_colors=3, brightness_threshold=30, uniqueness_threshold=250):
    """
    Downloads an image from the given URL, filters out dark pixels, and extracts the dominant
    colors ensuring that they are distinct based on a minimum Euclidean distance in RGB space.
    
    Parameters:
      image_url (str): URL of the image.
      num_colors (int): Number of unique dominant colors to extract.
      brightness_threshold (int): Minimum brightness (0-255) required for a pixel to be considered.
      uniqueness_threshold (float): Minimum Euclidean distance in RGB space between colors.
      
    Returns:
      np.ndarray: Array of shape (num_colors, 3) containing the dominant RGB colors.
    """
    # Download the image
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

    # Convert image to RGB and resize for faster processing
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)

    # Filter out dark pixels using the brightness threshold (luminance formula)
    bright_pixels = []
    for pixel in pixels:
        brightness = 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]
        if brightness >= brightness_threshold:
            bright_pixels.append(pixel)
    bright_pixels = np.array(bright_pixels)
    
    if len(bright_pixels) == 0:
        raise Exception("No bright pixels found after filtering.")

    # Run KMeans on extra clusters to provide a pool of candidate colors
    num_clusters = max(num_colors * 2, 6)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(bright_pixels)
    centers = kmeans.cluster_centers_.astype(int)

    # Count frequency for each cluster
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

    # Fallback: if not enough unique colors, add additional candidates regardless
    if len(unique_colors) < num_colors:
        for cluster in clusters:
            candidate = cluster['center']
            if not any(np.array_equal(candidate, col) for col in unique_colors):
                unique_colors.append(candidate)
            if len(unique_colors) >= num_colors:
                break

    return np.array(unique_colors[:num_colors])

if __name__ == '__main__':
    app.run(debug=True)
