# Laporan Analisis Komprehensif: Model Machine Learning Forex Forecasting (USD/CHF)

## 1. Ringkasan Eksekutif
Proyek Anda bertujuan untuk memprediksi *log-return* 15 menit dari pasangan mata uang USD/CHF menggunakan data OHLC (2020-2026). Memprediksi pasar forex frekuensi tinggi adalah tugas yang sangat rumit karena rasio sinyal-ke-noise (signal-to-noise ratio) yang sangat rendah dan sifat pasar yang sangat efisien (EMH). Pendekatan metodologis yang Anda gunakan sudah sangat terstruktur: pembagian data secara kronologis (mencegah *look-ahead bias*), perancangan fitur teknikal yang runut, serta eksperimen komparatif antara model linear, non-linear, dan tree-based.

Mendapatkan nilai $R^2$ yang dekat dengan 0 (bahkan negatif) adalah **hasil yang wajar dan realistis** dalam akademik *quantitative finance* pada frekuensi intraday. Laporan ini menyajikan evaluasi mendetail untuk setiap *pipeline* serta saran strategis untuk *breakthrough* selanjutnya.

---

## 2. Analisis Data & Feature Engineering

### Kekuatan (Pros):
- **Resampling 15-Menit**: Langkah yang sangat tepat. Data 1 menit terlalu didominasi oleh *micro-market-structure noise* (seperti eksekusi HFT dan *bid-ask bounce*). Resampling membantu memuluskan kurva agar *macro-structure* lebih jelas.
- **Target *Log-Return***: Penggunaan *log-return* jauh lebih robust, skalabel, dan stasioner dibandingkan memprediksi level harga absolut.
- **Split Kronologis**: Menghindari kebocoran data (*data leakage*) yang fatal yang biasa terjadi jika menggunakan `train_test_split` acak pada *time-series*.

### Area Peningkatan & Saran (Cons/Suggestions):
- **Kurangnya Fitur Temporal (Konteks Waktu)**: Pasar Forex memiliki struktur sesi (Sesi Tokyo, London, New York). Volatilitas dan likuiditas berubah drastis tergantung pada jam operasi. 
  - *Saran*: Tambahkan fitur waktu yang di-encode secara siklikal (menggunakan `sin` dan `cos`) untuk `hour_of_day`, dan label kategori untuk `day_of_week`.
- **Standarisasi (*Scaling*) Global yang Rentan**: Anda menerapkan `StandardScaler` pada seluruh data *training*. Di pasar keuangan, varians dan rata-rata selalu berubah (*non-stationary regime*).
  - *Saran*: Gunakan **Rolling/Windowed Z-Score** di mana harga distandarisasi hanya berdasarkan distribusi 100 *candle* ke belakang, bukan seluruh data sejak 2020. Alternatif tingkat lanjut adalah metode **Fractional Differentiation**.
- **Perumusan Target yang Kaku**: Prediksi titik (regresi nilai presisi return) sangat sulit.
  - *Saran*: Ubah problem regresi menjadi **Klasifikasi Biner/Multi-kelas** (misalnya: Harga Naik > ATR vs Harga Turun < ATR vs Sideways). Model ML jauh lebih tangguh dalam mengenali pola kelas (Arah) dibandingkan memprediksi besaran presisi (Regresi).

---

## 3. Analisis Pemodelan & Arsitektur

### A. MLP (Multi-Layer Perceptron)
- **Kondisi Saat Ini**: Arsitektur `33 -> 1024 -> 512 -> 256 -> 128 -> 1` yang Anda buat sangat *overkill*. Untuk ruang input yang hanya 33 dimensi, memiliki jutaan parameter pada data yang nyaris murni *noise* adalah resep pasti untuk **overfitting ekstrem**. Model akan langsung menghafal *noise* pada set *training*.
- **Saran Perbaikan**:
  1. **Pertahankan Kerampingan**: Gunakan maksimal 1 atau 2 *hidden layer* (misal: `33 -> 64 -> 32 -> 1`).
  2. **Regularisasi Agresif**: Tambahkan *Dropout* (0.3 - 0.5) pada setiap *layer* dan gunakan *L2 Weight Decay* pada *optimizer* Adam.
  3. **Arsitektur Berbasis Urutan Waktu**: Ganti MLP statis dengan arsitektur yang mengerti urutan waktu secara historis (*memory*), seperti **LSTM/GRU** atau **TCN (Temporal Convolutional Network)**, dengan memberikan matriks masukan berbentuk 3D: `(batch, timesteps, features)`.

### B. KNN (K-Nearest Neighbors)
- **Kondisi Saat Ini**: Berjalan baik pada fase tuning karena penggunaan K yang besar, namun tetap saja memiliki performa yang kurang memuaskan.
- **Kelemahan**: KNN adalah korban utama dari **Curse of Dimensionality**. Mengukur jarak *Euclidean* lurus pada ruang dimensi 33 dari seri waktu keuangan tidak bermakna karena beberapa fitur mendominasi jarak (misal, MACD vs Lags).
- **Saran Perbaikan**: Jangan gunakan KNN biasa. Jika ingin metode berbasis ketetanggaan, gunakan metode kemiripan khusus *time series* seperti **Dynamic Time Warping (DTW)** atau turunkan dimensinya dulu dengan **UMAP**.

### C. XGBoost
- **Kondisi Saat Ini**: Ini adalah model andalan Anda. Satu-satunya yang mendapat R² marjinal positif dan DirAcc di atas 50%. Penggunaan *shallow trees* (max_depth=4) membuktikan bahwa regulasi ketat adalah kunci agar tak terjebak *noise*.
- **Saran Perbaikan**:
  1. Lakukan evaluasi **SHAP values** untuk mengidentifikasi fitur (dari 34 tersebut) mana yang *noise* murni dan mana yang informatif. Seringkali, membuang 20 fitur *noise* akan meningkatkan keakuratan XGBoost secara dramatis (*feature pruning*).
  2. Gunakan algoritma *boosting* dengan penanganan *overfitting* data ber-noise tinggi yang lebih kokoh seperti **CatBoost**.

---

## 4. Metrik Evaluasi & Validasi (*Backtesting*)

- **Kondisi Saat Ini**: Menggunakan blok split yang statis (Train < 2025, Val Q1-Q3 2025, Test Q4 2025+).
- **Kelemahan Utama**: Model dilatih di rezim pandemi (2020-2021) dan rentang datar (2022-2024), lalu diuji secara langsung di kondisi pecahnya tren (2025). Jarak waktu (*gap*) antara data uji dan awal data latih terlalu jauh, yang menyebabkan distribusi usang.
- **Saran Perbaikan**:
  1. **Walk-Forward Validation**: Latih model dalam jendela bergulir (misal latih pada 1 tahun terakhir, tes untuk 1 bulan ke depan; geser maju 1 bulan; latih ulang, dst).
  2. **Metrik Finansial**: $R^2$ dan RMSE sama sekali tidak mengukur seberapa baik strategi ini saat dipasangkan modal. Konversikan tebakan model menjadi strategi simulasi (Backtest). Hitung **Sharpe Ratio**, **Maximum Drawdown**, dan **Profit Factor**. Seringkali, model dengan akurasi 45% sangat profitabel jika RRR (Risk Reward Ratio)-nya optimal.

---

## 5. Rekomendasi "Next Steps" (Saran Pengembangan Strategis)

Jika tujuan utama proyek ini adalah bergeser dari sekadar tugas akademik menuju sistem analitik quant-trading:

1. **Gunakan Paradigma Meta-Labeling**:
   Mesin sangat kesulitan mendeteksi "Kemana harga akan pergi 15 menit ke depan".
   Alih-alih menyuruh ML menebak arah, gunakan strategi teknikal tradisional (misal Breakout Bollinger Bands) sebagai penghasil sinyal dasar. Lalu, latih XGBoost hanya pada momen sinyal teknikal tersebut menyala, dan tugas ML adalah memprediksi **Apakah sinyal teknikal tersebut adalah "False Breakout" atau "Valid Breakout" (1 atau 0)**. ML sangat fenomenal dalam perannya sebagai "Filter penolak risiko".

2. **Perbesar Rentang Prediksi (Horizon)**:
   Pergerakan 15 menit sangat didorong oleh algoritma arbitrase likuiditas Bank/Institusi (HFT). Pindahkan horizon prediksi ke `H1` (1 jam) atau `H4` (4 jam) dimana sentimen riil dan makro-ekonomi mulai membekas pada *price action*.

3. **Data Order Book / Sentimen**:
   Informasi OHLC saja merupakan *lagging data* (harga telah terjadi). Jika memungkinkan, integrasikan fitur derivatif tingkat kedua (seperti order-flow imbalance) atau indikator sentimen pasar (Yields Obligasi AS, DXY).

**Kesimpulan:** 
Dari perspektif rekayasa perangkat lunak dan arsitektur *data science*, apa yang telah Anda tulis dalam notebook sangat rapi, matang, dan bisa direproduksi dengan baik. Anda telah melakukan semua langkah validasi (*lookahead-free*, EDA clustering) dengan sangat profesional. Tantangan Anda ke depan adalah di ranah rekayasa keilmuan keuangan kuantitatif (*Quantitative Financial Engineering*), yaitu merumuskan fitur dan cara pandang baru yang lebih selaras dengan mekanisme aktual pertukaran uang di pasar valuta asing.
