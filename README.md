# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan yang telah berdiri
sejak tahun 2000 dan telah menghasilkan banyak lulusan. Namun, institusi
juga menghadapi permasalahan berupa mahasiswa yang tidak menyelesaikan
pendidikannya atau mengalami *dropout*.

Tingginya jumlah mahasiswa yang mengalami *dropout* menjadi perhatian
karena dapat berdampak pada keberhasilan studi mahasiswa dan proses
monitoring akademik institusi. Oleh karena itu, Jaya Jaya Institut
membutuhkan solusi yang dapat membantu memahami pola dan karakteristik
mahasiswa, memonitor performa akademik, serta melakukan screening awal
terhadap mahasiswa yang memiliki kemungkinan mengalami *dropout*.

Proyek ini menggunakan analisis data, dashboard, dan machine learning
sebagai alat bantu untuk mendukung proses monitoring dan screening.
Hasil model tidak dimaksudkan sebagai keputusan akhir mengenai kondisi
mahasiswa, tetapi sebagai informasi awal yang perlu diverifikasi oleh
pihak institusi.

### Permasalahan Bisnis

Permasalahan yang ingin diselesaikan dalam proyek ini adalah:

1.  Institusi membutuhkan pemahaman mengenai distribusi status
    mahasiswa.
2.  Institusi membutuhkan informasi mengenai perbedaan performa akademik
    antarstatus mahasiswa.
3.  Institusi perlu mengetahui karakteristik dan faktor akademik,
    administratif, serta ekonomi yang berkaitan dengan status mahasiswa.
4.  Institusi membutuhkan cara untuk melakukan screening awal terhadap
    mahasiswa yang berpotensi mengalami *dropout*.
5.  Hasil analisis perlu disajikan dalam bentuk visualisasi yang mudah
    dipahami untuk membantu proses monitoring dan tindak lanjut.

### Cakupan Proyek

Cakupan proyek meliputi:

1.  Memahami struktur dan karakteristik dataset mahasiswa.
2.  Melakukan pemeriksaan *missing value* dan *duplicate*.
3.  Melakukan Exploratory Data Analysis (EDA) terhadap variabel
    akademik, administratif, ekonomi, dan karakteristik mahasiswa.
4.  Menganalisis distribusi *Status* mahasiswa.
5.  Menganalisis *dropout rate* berdasarkan beberapa variabel
    kategorikal.
6.  Melakukan *feature engineering*.
7.  Menggunakan data **Graduate** dan **Dropout** untuk proses training
    model binary classification, sedangkan data **Enrolled** disimpan
    terpisah untuk kebutuhan screening/prediksi.
8.  Membagi data training dan testing menggunakan *stratified train-test
    split*.
9.  Membandingkan Random Forest dan XGBoost menggunakan
    RandomizedSearchCV dan StratifiedKFold.
10. Mengevaluasi model menggunakan accuracy, precision, recall,
    F1-score, confusion matrix, serta metrik khusus kelas Dropout.
11. Menyimpan model dalam format `.pkl`.
12. Membuat dashboard untuk monitoring kondisi mahasiswa.
13. Membuat prototype machine learning menggunakan Streamlit.
14. Menyediakan input Student Index untuk menerapkan model pada
    mahasiswa Enrolled.
15. Menyediakan risk level berdasarkan probabilitas dropout:
    -   **High Risk**: probabilitas dropout \>= 70%.
    -   **Medium Risk**: probabilitas dropout \>= 40% dan \< 70%.
    -   **Low Risk**: probabilitas dropout \< 40%.

### Persiapan

Sumber data: Dataset yang digunakan adalah dataset *Students'
Performance* yang disediakan untuk proyek akhir Dicoding. Dataset
terdiri dari 4.424 baris dan 37 kolom.

Setup environment:

``` bash
python -m venv .venv
```

Aktifkan environment pada Windows:

``` bash
.venv\Scripts\activate
```

Install seluruh dependency:

``` bash
pip install -r requirements.txt
```

## Business Dashboard

Dashboard dibuat untuk membantu Jaya Jaya Institut memahami kondisi
mahasiswa dan memonitor indikator yang berkaitan dengan *dropout*.

Dashboard menampilkan beberapa komponen utama:

1.  **Total Mahasiswa** --- menampilkan jumlah seluruh mahasiswa pada
    dataset.
2.  **Dropout Rate** --- menampilkan persentase mahasiswa yang berstatus
    Dropout.
3.  **Graduate Rate** --- menampilkan persentase mahasiswa yang
    berstatus Graduate.
4.  **Enrolled Rate** --- menampilkan persentase mahasiswa yang masih
    berstatus Enrolled.
5.  **Distribusi Status Mahasiswa** --- diagram donut memperlihatkan
    proporsi Graduate, Dropout, dan Enrolled.
6.  **Dropout Rate berdasarkan Course** --- diagram batang membandingkan
    dropout rate pada course dengan jumlah observasi yang memenuhi batas
    minimum data.
7.  **Status Pembayaran** --- diagram donut menampilkan distribusi
    status pembayaran tuition fees dengan label deskriptif.
8.  **Status Finansial Mahasiswa** --- diagram batang menampilkan
    distribusi berdasarkan status debtor dengan label **Berutang** dan
    **Tidak Berutang**.
9.  **Performa Semester 1 vs Semester 2** --- diagram batang
    membandingkan rata-rata nilai semester 1 dan semester 2 berdasarkan
    status mahasiswa.
10. **Approval Rate Semester 1 vs Semester 2** --- diagram batang
    membandingkan rasio mata kuliah yang disetujui terhadap mata kuliah
    yang diambil pada semester 1 dan semester 2.

Dashboard juga menyediakan penerapan model melalui sidebar. Pengguna
dapat memasukkan Student Index, kemudian menekan tombol **Terapkan
Model** untuk memperoleh hasil prediksi, probabilitas dropout, risk
level, dan probabilitas setiap status.

*Link Dashboard Looker Studio:*
https://datastudio.google.com/reporting/55c07af9-5410-4d96-b557-1f77ed44ba96

*Link Prototype Streamlit:*
https://institutjayajayadashboard-4olgxehshv9ftm3qexxg9p.streamlit.app/

*Link Video Penjelasan:*
https://youtu.be/D4zR2kGy-AI?si=AeB2Ub0CBp9zCJtf

## Menjalankan Sistem Machine Learning

Prototype machine learning dibuat menggunakan Streamlit. Model yang
telah dilatih disimpan dalam format `.pkl`, sehingga aplikasi tidak
perlu melakukan training ulang ketika dijalankan.

Model menggunakan **binary classification** dengan target:

-   `0` = Graduate
-   `1` = Dropout

Data dengan status **Enrolled** tidak digunakan dalam proses training.
Data tersebut disimpan terpisah untuk digunakan sebagai data
screening/prediksi pada prototype.

Model yang dibandingkan adalah Random Forest dan XGBoost dengan
hyperparameter tuning menggunakan RandomizedSearchCV dan validasi
StratifiedKFold. Model final yang digunakan adalah **XGBoost**.

### Evaluasi Model

Berdasarkan hasil evaluasi model final pada data testing:

-   **Accuracy:** 92,98%
-   **Precision kelas Dropout:** 90,59%
-   **Recall kelas Dropout:** 91,55%
-   **F1-score kelas Dropout:** 91,07%
-   **Macro F1-score:** 92,64%

Metrik tersebut menggambarkan performa model pada data pengujian dan
bukan jaminan bahwa setiap prediksi individu akan selalu benar.

### Menjalankan Prototype Secara Lokal

Pastikan file berikut tersedia:

``` text
app.py
Data/data.csv
student_dropout_model.pkl
model_metadata.pkl
requirements.txt
```

Jalankan:

``` bash
streamlit run app.py
```

Setelah itu buka alamat Streamlit yang muncul pada terminal, biasanya:

``` text
http://localhost:8501
```

### Cara Menggunakan Prototype

1.  Jalankan aplikasi Streamlit.
2.  Buka sidebar **Cari Student**.
3.  Masukkan **Student Index**.
4.  Klik **Terapkan Model**.
5.  Sistem mengambil data mahasiswa berdasarkan posisi baris pada data
    Enrolled.
6.  Sistem melakukan *feature engineering*.
7.  Model melakukan prediksi status mahasiswa.
8.  Sistem menampilkan Student Index, Predicted Status, Dropout
    Probability, Risk Level, probabilitas setiap status, dan ringkasan
    data mahasiswa yang dipilih.

## Conclusion

### 1. Kesimpulan Berdasarkan Hasil Analisis Data

Hasil EDA dan dashboard menunjukkan bahwa kondisi mahasiswa dapat
dianalisis melalui beberapa kelompok informasi, terutama performa
akademik, jumlah mata kuliah yang disetujui, perubahan performa
antarsemester, serta beberapa faktor administratif dan ekonomi.

Karakteristik yang perlu menjadi perhatian dalam monitoring adalah
mahasiswa yang menunjukkan performa akademik lebih rendah, tingkat
persetujuan mata kuliah yang lebih rendah, atau perubahan performa
akademik yang kurang baik antarsemester. Informasi seperti status
pembayaran tuition fees, status debtor, dan scholarship holder juga
dapat digunakan sebagai konteks tambahan dalam proses pendampingan.

Perbedaan dropout rate antar kategori pada dashboard merupakan temuan
deskriptif dari data. Temuan tersebut digunakan untuk membantu institusi
menentukan area yang perlu dipantau lebih lanjut dan tidak secara
langsung menunjukkan hubungan sebab-akibat.

### 2. Kesimpulan Berdasarkan Hasil Machine Learning

Model final yang digunakan adalah XGBoost dengan target binary
classification, yaitu Graduate dan Dropout. Pada data testing, model
memperoleh:

-   Accuracy sebesar **92,98%**.
-   Precision kelas Dropout sebesar **90,59%**.
-   Recall kelas Dropout sebesar **91,55%**.
-   F1-score kelas Dropout sebesar **91,07%**.
-   Macro F1-score sebesar **92,64%**.

Recall kelas Dropout sebesar 91,55% menunjukkan bahwa sebagian besar
mahasiswa yang termasuk kelas Dropout pada data pengujian berhasil
dikenali oleh model. Precision sebesar 90,59% menunjukkan bahwa sebagian
besar prediksi Dropout yang dihasilkan model sesuai dengan kelas Dropout
pada data pengujian.

Model menggunakan fitur akademik, administratif, ekonomi, dan fitur
hasil *feature engineering* yang tersedia pada dataset. Feature
engineering mencakup antara lain approval rate semester 1 dan 2,
evaluation rate, perubahan nilai, perubahan jumlah mata kuliah yang
disetujui, total mata kuliah yang disetujui, total mata kuliah yang
diambil, total evaluasi, rata-rata nilai semester, dan grade ratio.

Prototipe Streamlit digunakan untuk menerapkan model pada data mahasiswa
Enrolled sebagai alat bantu screening. Hasil prediksi, probabilitas
dropout, dan risk level dapat digunakan sebagai informasi awal bagi
pihak institusi dan tetap perlu diverifikasi berdasarkan kondisi
mahasiswa yang sebenarnya.

### Rekomendasi Action Items

-   Melakukan screening berkala terhadap mahasiswa menggunakan indikator
    akademik dan prototype machine learning.
-   Memprioritaskan verifikasi mahasiswa yang masuk kategori **High
    Risk** sebelum menentukan bentuk pendampingan.
-   Memantau nilai semester, jumlah mata kuliah yang disetujui, serta
    perubahan performa antarsemester.
-   Memantau faktor administratif dan ekonomi seperti status pembayaran,
    debtor, dan scholarship sebagai konteks tambahan.
-   Memberikan pendampingan yang sesuai berdasarkan hasil screening dan
    komunikasi langsung dengan mahasiswa.
-   Melakukan evaluasi model secara berkala menggunakan data mahasiswa
    terbaru agar performanya tetap relevan.
-   Memantau *false positive* dan *false negative* sebagai bagian dari
    evaluasi kualitas model.
-   Menjaga privasi data mahasiswa dan membatasi akses terhadap hasil
    prediksi sesuai kebutuhan institusi.
