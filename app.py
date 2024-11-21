from flask import Flask, request, render_template, redirect, url_for, session, flash
import cv2
import os
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from datetime import datetime
import random
from werkzeug.utils import secure_filename


# Inisialisasi aplikasi Flask
app = Flask(__name__)
# Konfigurasi aplikasi
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'  # Lokasi penyimpanan file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}  # Format file yang diizinkan
# Kamus pengguna: username sebagai kunci, dan tuple (password, role) sebagai nilai
users = {
    'admin': ('admin', 'admin'),
    'user1': ('password1', 'user'),  # Contoh pengguna biasa
    'user2': ('password2', 'user')  # Tambah lebih banyak jika diperlukan
}
app.secret_key = 'supersecretkey'

# Muat model dan skaler
model = joblib.load('models/knn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Fungsi untuk memeriksa ekstensi file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Fungsi untuk ekstraksi fitur menggunakan GLCM (sama seperti sebelumnya)
def extract_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_image, distances, angles, 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

# Rute untuk halaman utama
@app.route('/')
def home():
    if 'username' in session:
        role = session.get('role')
        if role == 'admin':
            return redirect(url_for('dashboard'))
        elif role == 'user':
            return redirect(url_for('site'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username][0] == password:
            session['username'] = username
            session['role'] = users[username][1]  # Simpan role di sesi
            if users[username][1] == 'admin':
                return redirect(url_for('dashboard'))
            elif users[username][1] == 'user':
                return redirect(url_for('site'))
        else:
            return 'Invalid credentials, please try again!'
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session and session.get('role') == 'admin':
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/site')
def site():
    if 'username' in session and session.get('role') == 'user':
        return render_template('site.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
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
        features_glcm = extract_glcm_features(image)
        features = scaler.transform([features_glcm])
        
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
        kode_random = random.randint(00, 99)  # 6-digit random code
        tanggal_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
        
        return render_template('predict.html', features=features_glcm, label=label, image_file=file.filename, kode_random=kode_random, tanggal_sekarang=tanggal_sekarang)
    
    return render_template('dashboard.html', label="Error")

# Fungsi untuk memeriksa format file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Rute untuk halaman utama
@app.route('/dashboard/train', methods=['GET', 'POST'])
def index():
    return render_template('train.html')

# Rute untuk mengunggah file
@app.route('/dashboard/train/upload', methods=['POST'])
def upload_file():
    # Memeriksa apakah ada file dalam permintaan
    if 'file' not in request.files:
        flash('No file part')  # Flash pesan kesalahan
        return redirect(request.url)
    
    file = request.files['file']
    
    # Jika tidak ada file yang dipilih
    if file.filename == '':
        flash('No selected file')  # Flash pesan kesalahan
        return redirect(request.url)

    # Memeriksa apakah input radio telah dipilih
    quality = request.form.get('quality')
    if not quality:
        flash('Please select the quality of the image.')  # Flash pesan kesalahan
        return redirect(request.url)
    
    # Memeriksa apakah file valid
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Mengamankan nama file
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], quality)
        
        # Membuat subfolder berdasarkan kualitas jika belum ada
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        file.save(os.path.join(save_path, filename))  # Simpan file
        flash(f'File successfully uploaded as {quality}')  # Flash pesan keberhasilan
        return redirect(url_for('index'))
    else:
        flash('Invalid file type')  # Flash pesan kesalahan tipe file
        return redirect(request.url)

FOLDER_BERKUALITAS = 'static/uploads/iya'
FOLDER_TIDAKBERKUALITAS = 'static/uploads/tidak'

@app.route('/dashboard/data')
def show_images():
    # Ambil daftar gambar dari folder berkualitas dan tidakberkualitas
    berkualitas_images = os.listdir(FOLDER_BERKUALITAS)
    tidakberkualitas_images = os.listdir(FOLDER_TIDAKBERKUALITAS)

    # Gabungkan semua gambar dengan statusnya
    images = []
    nomor = 1
    
    # Menambahkan gambar berkualitas dengan status 'Berkualitas'
    for img in berkualitas_images:
        images.append({
            'nomor': nomor,
            'subfolder': 'iya',
            'status': 'Berkualitas',
            'image': img
        })
        nomor += 1

    # Menambahkan gambar tidak berkualitas dengan status 'Tidak Berkualitas'
    for img in tidakberkualitas_images:
        images.append({
            'nomor': nomor,
            'subfolder': 'tidak',
            'status': 'Tidak Berkualitas',
            'image': img
        })
        nomor += 1

    return render_template('show.html', images=images)

# Menjalankan aplikasi
if __name__ == '__main__':
    # Membuat folder jika belum ada
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
# # Jalankan aplikasi Flask
# if __name__ == '__main__':
#     app.run(debug=True)