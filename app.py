from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Image processing functions
def apply_gaussian_blur(image, kernel_size=(3, 3), sigma=1):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

def enhance_contrast(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

def sharpening(image):
    blurred = cv2.GaussianBlur(image, (0,0), 1)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading and processing image
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Read original image
        original_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Apply image enhancement algorithms
        blurred_img = apply_gaussian_blur(original_image)
        contrast_enhanced_img = enhance_contrast(blurred_img)
        sharpened_img = sharpening(contrast_enhanced_img)
        
        # Save processed image temporarily
        filename = 'processed_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, sharpened_img)
        
        # Save original image temporarily
        original_filename = 'original_image.jpg'
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        cv2.imwrite(original_filepath,original_image)

        print(filename,original_filename)
        
        return redirect(url_for('result', original_filename=original_filename, processed_filename=filename))

# Route for displaying the result page
@app.route('/result')
def result():
    
    original_filename_img = request.args.get('original_filename')
    processed_filename_img = request.args.get('processed_filename')
    # print(original_filename,processed_filename)
    return render_template('result.html', original_filename=original_filename_img, processed_filename=processed_filename_img)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
