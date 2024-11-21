from flask import Flask, request, render_template, redirect, url_for, session
import cv2
import os
import numpy as np
import joblib
from skimage.feature import greycomatrix, greycoprops

# Inisialisasi aplikasi Flask
app = Flask(__name__)

users = {'admin': 'admin'}
app.secret_key = 'supersecretkey'

# Muat model dan skaler
model = joblib.load('models/knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Fungsi untuk ekstraksi fitur menggunakan GLCM (sama seperti sebelumnya)
def extract_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(gray_image, distances, angles, 256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').flatten()
    dissimilarity = greycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = greycoprops(glcm, 'homogeneity').flatten()
    energy = greycoprops(glcm, 'energy').flatten()
    correlation = greycoprops(glcm, 'correlation').flatten()
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

# Rute untuk halaman utama
@app.route('/')
def home():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid credentials, please try again!'
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

# Rute untuk memproses gambar dan melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Baca gambar dari input pengguna
        image = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # Ekstraksi fitur dari gambar
        features = extract_glcm_features(image)
        features = scaler.transform([features])
        
        # Prediksi menggunakan model
        prediction = model.predict(features)[0]

         # Simpan gambar yang diunggah
        upload_folder = 'static/uploads'  # Pastikan folder ini ada
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        file_path = os.path.join(upload_folder, file.filename)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV menggunakan BGR
        cv2.imwrite(file_path, image_bgr)
        
        # Dekode label (misalnya, 0 -> "bad", 1 -> "good")
        label = "Berkualitas" if prediction == 1 else "Tidak Berkualitas"
        
        return render_template('predict.html', label=label, image_file=file.filename)
    
    return render_template('dashboard.html', label="Error")

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
