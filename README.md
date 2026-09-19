# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan yang telah berdiri sejak tahun 2000 dan telah menghasilkan banyak lulusan. Namun, institusi juga menghadapi permasalahan berupa mahasiswa yang tidak menyelesaikan pendidikannya atau mengalami dropout.

Berdasarkan dataset yang digunakan, terdapat 4.424 data mahasiswa dengan tiga status akhir, yaitu Graduate, Dropout, dan Enrolled. Distribusi status menunjukkan bahwa 2.209 mahasiswa berstatus Graduate (49,93%), 1.421 Dropout (32,12%), dan 794 Enrolled (17,95%).

Permasalahan tersebut penting untuk ditangani karena institusi membutuhkan informasi yang dapat membantu memahami pola performa mahasiswa sekaligus melakukan deteksi awal terhadap mahasiswa yang memiliki kemungkinan dropout. Proyek ini menggabungkan analisis data, dashboard, dan machine learning sebagai alat bantu screening.



### Permasalahan Bisnis
Permasalahan yang ingin diselesaikan dalam proyek ini adalah:

1. Tingginya proporsi mahasiswa yang berstatus Dropout.
2. Institusi membutuhkan pemahaman mengenai distribusi status mahasiswa.
3. Institusi membutuhkan informasi mengenai perbedaan performa akademik antarstatus mahasiswa.
4. Institusi perlu memantau faktor akademik dan administratif yang berkaitan dengan status mahasiswa, seperti nilai semester, jumlah mata kuliah yang disetujui, status pembayaran tuition fees, status debtor, dan scholarship holder.
5. Institusi membutuhkan prototype machine learning yang dapat digunakan untuk melakukan screening status mahasiswa dan memperkirakan probabilitas dropout.
6. Hasil analisis perlu disajikan dalam bentuk visualisasi yang mudah dipahami sehingga dapat mendukung proses monitoring dan tindak lanjut.


### Cakupan Proyek
Cakupan proyek meliputi:
1. Memahami struktur dan karakteristik dataset mahasiswa.
2. Melakukan pemeriksaan missing value dan duplicate.
3. Melakukan Exploratory Data Analysis (EDA) terhadap variabel akademik, administratif, ekonomi, dan karakteristik mahasiswa.
4. Menganalisis distribusi Status mahasiswa.
5. Menganalisis dropout rate berdasarkan beberapa variabel kategorikal.
6. Melakukan feature engineering.
7. Membagi data menjadi data training dan testing menggunakan stratified train-test split.
8. Membandingkan Random Forest dan XGBoost menggunakan RandomizedSearchCV dengan StratifiedKFold.
9. Mengevaluasi model menggunakan accuracy, precision, recall, F1-score, confusion matrix, serta metrik khusus kelas Dropout.
10. Menyimpan model dalam format .pkl.
11. Membuat dashboard untuk monitoring performa mahasiswa.
12. Membuat prototype machine learning menggunakan Streamlit.
13. Menyediakan input Student Index untuk menerapkan model pada mahasiswa tertentu.
14. Menyediakan risk level berdasarkan probabilitas dropout:
    - High Risk: probabilitas dropout >= 70%
    - Medium Risk: probabilitas dropout >= 40% dan < 70%
    - Low Risk: probabilitas dropout < 40%


### Persiapan

Sumber data: Dataset yang digunakan adalah dataset Students' Performance yang disediakan untuk proyek akhir Dicoding. Dataset yang digunakan terdiri dari 4.424 baris dan 37 kolom.

Setup environment:
Pastikan Python sudah terpasang. Kemudian buat virtual environment:

python -m venv .venv

Aktifkan environment pada Windows:

.venv\Scripts\activate

Install seluruh dependency:

pip install -r requirements.txt


## Business Dashboard
Dashboard dibuat untuk membantu Jaya Jaya Institut memahami kondisi mahasiswa dan memonitor indikator yang berkaitan dengan dropout.

Dashboard menampilkan beberapa komponen utama:

1. Total Mahasiswa
Menampilkan jumlah seluruh mahasiswa pada dataset, yaitu 4.424 mahasiswa.

2. Dropout Rate
Menampilkan persentase mahasiswa yang berstatus Dropout, yaitu 32,12%.

3. Graduation Rate
Menampilkan persentase mahasiswa yang berstatus Graduate, yaitu 49,93%.

4. Enrolled Rate
Menampilkan persentase mahasiswa yang masih berstatus Enrolled, yaitu 17,95%.

5. Distribusi Status Mahasiswa
Diagram donut digunakan untuk memperlihatkan proporsi Graduate, Dropout, dan Enrolled.

6. Dropout Rate berdasarkan Course
Diagram batang digunakan untuk membandingkan dropout rate pada course dengan jumlah observasi yang memenuhi batas minimum data. Visualisasi ini membantu melihat kelompok course yang memiliki proporsi dropout yang berbeda.

7. Status Pembayaran
Diagram donut menampilkan distribusi status Tuition_fees_up_to_date.

8. Status Finansial Mahasiswa
Diagram batang menampilkan distribusi mahasiswa berdasarkan Debtor.

9. Performa Semester 1 vs Semester 2
Diagram batang membandingkan rata-rata nilai semester 1 dan semester 2 berdasarkan status mahasiswa.

10. Approval Rate Semester 1 vs Semester 2
Diagram batang membandingkan rasio mata kuliah yang disetujui terhadap mata kuliah yang diambil pada semester 1 dan semester 2.

Dashboard juga menyediakan bagian penerapan model melalui sidebar. Pengguna memasukkan Student Index, kemudian menekan tombol Terapkan Model untuk memperoleh hasil prediksi, probabilitas dropout, risk level, dan probabilitas setiap status.

*Link Dashboard Loocker Studio:* https://datastudio.google.com/reporting/55c07af9-5410-4d96-b557-1f77ed44ba96
*Link Video Penjelasa:* https://youtu.be/D4zR2kGy-AI?si=AeB2Ub0CBp9zCJtf


## Menjalankan Sistem Machine Learning
Prototype machine learning dibuat menggunakan Streamlit. Model yang telah dilatih disimpan dalam format .pkl, sehingga aplikasi tidak perlu melakukan training ulang ketika dijalankan.

Model final pada notebook dipilih berdasarkan hasil cross-validation. Notebook membandingkan Random Forest dan XGBoost menggunakan RandomizedSearchCV dan StratifiedKFold. Model yang terpilih pada hasil notebook adalah XGBoost dengan CV Accuracy sekitar 78,98%.

Untuk kelas Dropout, hasil evaluasi yang tercatat pada notebook adalah:

Precision: 78,20%

Recall: 73,24%

F1-score: 75,64%

Nilai tersebut digunakan sebagai informasi evaluasi model dan bukan sebagai jaminan bahwa setiap prediksi individu akan benar.
Menjalankan Prototype Secara Lokal

Pastikan file berikut tersedia:

app.py
Data/data.csv
student_dropout_model.pkl
requirements.txt

Jalankan:

streamlit run app.py

Setelah itu buka alamat Streamlit yang muncul pada terminal, biasanya:

http://localhost:8501
Cara Menggunakan Prototype

Jalankan aplikasi Streamlit.

Buka sidebar Cari Student.

Masukkan Student Index.

Klik Terapkan Model

Sistem mengambil data mahasiswa berdasarkan posisi baris dataset.

Sistem melakukan feature engineering.

Model melakukan prediksi status mahasiswa.

Sistem menampilkan:

Student Index

Predicted Status

Dropout Probability

Risk Level

Probabilitas setiap status

Ringkasan data mahasiswa yang dipilih

## Conclusion
Proyek ini menghasilkan solusi yang menggabungkan analisis data, dashboard, dan machine learning untuk membantu Jaya Jaya Institut memahami permasalahan dropout mahasiswa.

Dari 4.424 data mahasiswa, terdapat 1.421 mahasiswa Dropout atau sekitar 32,12%, 2.209 mahasiswa Graduate atau sekitar 49,93%, dan 794 mahasiswa Enrolled atau sekitar 17,95%. Distribusi tersebut menunjukkan bahwa status mahasiswa perlu dimonitor secara sistematis.

Analisis EDA berfokus pada performa akademik, jumlah mata kuliah yang disetujui, karakteristik mahasiswa, faktor administratif, serta beberapa faktor ekonomi. Perbedaan dropout rate antar kategori digunakan sebagai temuan deskriptif dan tidak diinterpretasikan sebagai hubungan sebab-akibat.

Pada tahap machine learning, Random Forest dan XGBoost dibandingkan menggunakan hyperparameter tuning dan cross-validation. Berdasarkan hasil notebook, XGBoost dipilih berdasarkan performa cross-validation dengan CV Accuracy sekitar 78,98%. Evaluasi khusus kelas Dropout menghasilkan precision 78,20%, recall 73,24%, dan F1-score 75,64%.

Prototype Streamlit memungkinkan pengguna memasukkan Student Index untuk menjalankan model pada satu mahasiswa dan melihat predicted status, dropout probability, serta risk level.

Dengan demikian, sistem yang dibuat dapat digunakan sebagai alat bantu monitoring dan screening awal. Keputusan pendampingan tetap perlu dilakukan oleh pihak institusi berdasarkan verifikasi dan konteks mahasiswa.


### Rekomendasi Action Items
Berikan beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
- Melakukan screening berkala: gunakan prototype untuk membantu mengidentifikasi mahasiswa yang perlu mendapatkan perhatian lebih awal.
- Memprioritaskan verifikasi mahasiswa berisiko: mahasiswa yang masuk kategori High Risk dapat ditinjau lebih lanjut oleh pihak akademik sebelum menentukan bentuk pendampingan.
- Memantau performa akademik: perhatikan nilai semester, jumlah mata kuliah yang disetujui, dan perubahan performa antarsemester.
- Memantau faktor administratif: perhatikan status pembayaran, debtor, dan scholarship sebagai bagian dari konteks pendampingan.
- Memberikan pendampingan yang sesuai: hasil model sebaiknya menjadi informasi awal yang dilanjutkan dengan komunikasi dan verifikasi oleh pihak akademik.
- Melakukan evaluasi model secara berkala: model perlu diuji kembali menggunakan data mahasiswa terbaru agar performanya tetap relevan.
- Memantau false positive dan false negative: dokumentasikan kesalahan prediksi untuk membantu mengevaluasi threshold dan kualitas model.
- Menjaga privasi data mahasiswa: data dan hasil prediksi perlu dikelola sesuai kebijakan privasi dan akses institusi.
