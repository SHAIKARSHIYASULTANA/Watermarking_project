from flask import Flask, render_template, request, session, redirect, url_for, send_from_directory
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management

# Folder where uploaded files will be saved
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def resize_images(img1, img2):
    min_height = min(img1.shape[0], img2.shape[0])
    min_width = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))
    return img1_resized, img2_resized

def normalize_lucas(lucas_sequence, length, value_limit=1e5):
    lucas_sequence = np.array(lucas_sequence)
    lucas_sequence = np.clip(lucas_sequence, -value_limit, value_limit)
    if np.ptp(lucas_sequence) == 0:
        return np.ones(length, dtype=float)
    min_val = np.min(lucas_sequence)
    max_val = np.max(lucas_sequence)
    scaled_lucas = 0.1 * (lucas_sequence - min_val) / (max_val - min_val)
    normalized_lucas = np.resize(scaled_lucas, length)
    if len(normalized_lucas) > length:
        normalized_lucas = normalized_lucas[:length]
    return normalized_lucas

def embed_watermark_tlsb(img, watermark):
    img, watermark = resize_images(img, watermark)
    lucas_sequence = [2, 1]
    while len(lucas_sequence) < img.size:
        lucas_sequence.append(lucas_sequence[-1] + lucas_sequence[-2])
    normalized_lucas = normalize_lucas(lucas_sequence, img.size)
    watermarked = img.astype(np.uint32)
    watermarked &= 0b11111111111111111111111111111000
    watermarked |= ((watermark.astype(np.uint32) >> 5) & 0b00000000000000000000000000000111)
    return watermarked.astype(np.uint8)

def extract_watermark_tlsb(watermarked):
    extracted_watermark = (watermarked & 0b00000000000000000000000000000111) << 5 
    
    return extracted_watermark.astype(np.uint8)

# Add this function to convert bytes to KB
def convert_bytes_to_kb(size_in_bytes):
    return size_in_bytes / 1024

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        start_time = time.time()
        cover_image = request.files['cover_image']
        watermark_image = request.files['watermark_image']
        cover_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(cover_image.filename))
        watermark_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(watermark_image.filename))
        cover_image.save(cover_image_path)
        watermark_image.save(watermark_image_path)
        session['cover_image_path'] = cover_image_path
        session['watermark_image_path'] = watermark_image_path
        cover_image_data = cv2.imread(cover_image_path)
        watermark_image_data = cv2.imread(watermark_image_path)
        watermarked_image_data = embed_watermark_tlsb(cover_image_data, watermark_image_data)
        watermarked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'watermarked_image_tlsb.jpg')
        cv2.imwrite(watermarked_image_path, watermarked_image_data)
        session['watermarked_image_path'] = watermarked_image_path
        end_time = time.time()
        embed_time = end_time - start_time
        session['embed_time'] = embed_time
        cover_image_size = os.path.getsize(cover_image_path)
        watermark_image_size = os.path.getsize(watermark_image_path)
        watermarked_image_size = os.path.getsize(watermarked_image_path)
        return render_template('index5.html',
                               watermarked_image_path='uploads/watermarked_image_tlsb.jpg',
                               watermarked_image_size=convert_bytes_to_kb(watermarked_image_size),
                               embed_time=embed_time, 
                               show_details=True)

    return render_template('index5.html',
                           watermarked_image_path=session.get('watermarked_image_path'))

@app.route('/extract', methods=['POST'])
def extract_watermark():
    start_time = time.time()
    # Get the paths of the cover image and watermark image
    cover_image_path = session.get('cover_image_path')
    watermark_image_path = session.get('watermark_image_path')

    if cover_image_path is None or watermark_image_path is None:
        return "Error: Cover image or watermark image path not found."

    # Process the watermarked image
    cover_image_data = cv2.imread(cover_image_path)
    watermarked_image_data = embed_watermark_tlsb(cover_image_data, cv2.imread(watermark_image_path))

    if watermarked_image_data is None:
        return "Error: Unable to embed watermark into the cover image."

    extracted_watermark_data = extract_watermark_tlsb(watermarked_image_data)
    extracted_watermark_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_watermark_image_tlsb.jpg')
    cv2.imwrite(extracted_watermark_image_path, extracted_watermark_data)
    end_time = time.time()
    extract_time = end_time - start_time

    # Get the sizes of the images
    cover_image_size = os.path.getsize(cover_image_path)
    watermark_image_size = os.path.getsize(watermark_image_path)
    watermarked_image_size = os.path.getsize(extracted_watermark_image_path)
    extracted_watermark_size = os.path.getsize(extracted_watermark_image_path)

    # Render the template with the appropriate image paths and sizes
    return render_template('index5.html',
                           watermarked_image_path='uploads/watermarked_image_tlsb.jpg',
                           extracted_watermark_image_path='uploads/extracted_watermark_image_tlsb.jpg',
                           watermarked_image_size=convert_bytes_to_kb(watermarked_image_size),
                           extracted_watermark_size=convert_bytes_to_kb(extracted_watermark_size),
                           embed_time=session.get('embed_time'),
                           extract_time=extract_time,
                           show_details=True)



@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)

