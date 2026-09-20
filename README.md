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
    RandomizedSearchCV dengan StratifiedKFold.
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

Dataset yang digunakan dalam proyek ini adalah **Students' Performance**.
Dataset disediakan untuk proyek akhir Dicoding dan terdiri dari 4.424 baris
dan 37 kolom.

**Sumber data:**

- Repository dataset Dicoding:
  https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance
- File dataset `data.csv`:


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

Berdasarkan dataset yang dianalisis, terdapat **4.424 mahasiswa** dengan tiga status akhir, yaitu **Graduate, Dropout, dan Enrolled**. Dashboard digunakan untuk menyajikan distribusi status tersebut secara kuantitatif serta membantu melihat pola dropout berdasarkan indikator akademik, administratif, dan ekonomi.

Untuk meningkatkan keterbacaan bagi pengguna non-teknis, nilai biner pada
dashboard ditampilkan menggunakan label deskriptif. Contohnya, Gender
ditampilkan sebagai **Perempuan/Laki-laki**, status finansial sebagai
**Tidak Berutang/Berutang**, dan status pembayaran sebagai **Up to Date/Not
Up to Date**.

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

11. **Filter Gender** --- filter interaktif untuk melihat dashboard
    berdasarkan kategori **Perempuan** dan **Laki-laki**.
12. **Filter Status Financial** --- filter interaktif untuk melihat
    dashboard berdasarkan kategori **Tidak Berutang** dan **Berutang**.

Dashboard juga menyediakan penerapan model pada halaman utama. Pengguna
dapat memasukkan Student Index, kemudian menekan tombol **Terapkan
Model** untuk memperoleh hasil prediksi, probabilitas dropout, risk
level, dan probabilitas setiap status.

*Link Dashboard Looker Studio:*
https://datastudio.google.com/reporting/55c07af9-5410-4d96-b557-1f77ed44ba96

*Link Prototype Streamlit Cloud:*
https://institutjayajayadashboard-4olgxehshv9ftm3qexxg9p.streamlit.app/

*Link Video Penjelasan:*
https://youtu.be/D4zR2kGy-AI?si=AeB2Ub0CBp9zCJtf

### Deployment ke Streamlit Community Cloud

Prototype machine learning dideploy menggunakan Streamlit Community Cloud
dan terhubung dengan repository GitHub. Agar fitur screening dapat berjalan,
repository harus memuat `app.py`, dataset, dependency, serta model binary
hasil training.

Alur deployment:

1. Upload/push seluruh project ke repository GitHub.
2. Login ke Streamlit Community Cloud.
3. Hubungkan repository GitHub dengan Streamlit Community Cloud.
4. Pilih branch dan file utama `app.py`.
5. Pastikan `requirements.txt`, folder `Data/`, `student_dropout_model.pkl`,
   dan `model_metadata.pkl` tersedia pada repository.
6. Deploy aplikasi.
7. Uji URL Streamlit dengan memilih Student Index dan menekan **Terapkan Model**.
8. Pastikan hasil prediksi muncul sebelum menggunakan URL tersebut sebagai
   prototype final submission.

Contoh perintah untuk memperbarui repository:

```bash
git add .
git commit -m "Update final project"
git push origin main
```

**Catatan:** `student_dropout_model.pkl` harus merupakan model binary dengan
kelas `0 = Graduate` dan `1 = Dropout`. File `model_metadata.pkl` digunakan
untuk menyimpan informasi model dan metrik evaluasi, bukan sebagai objek
model yang dipanggil untuk prediksi.


### Pembagian Fungsi Dashboard dan Prototype

- **Looker Studio** digunakan sebagai dashboard analisis dan monitoring.
  Pengguna dapat mengeksplorasi distribusi status, dropout rate,
  performa akademik, kondisi pembayaran, status finansial, serta
  menggunakan filter Gender dan Status Financial.
- **Streamlit** digunakan sebagai prototype Machine Learning untuk
  screening mahasiswa Enrolled menggunakan model binary Graduate vs
  Dropout.

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

Setelah feature engineering, data pemodelan memiliki **48 fitur**, yang
terdiri dari fitur asli yang digunakan dalam model dan **12 fitur
tambahan** hasil feature engineering.

Model yang dibandingkan adalah Random Forest dan XGBoost dengan
hyperparameter tuning menggunakan RandomizedSearchCV dan validasi
StratifiedKFold. Model final yang digunakan adalah **XGBoost**.

### Evaluasi Model

Pada tahap pemodelan, sebanyak **3.630 mahasiswa** digunakan sebagai data training dan testing setelah status `Enrolled` dipisahkan. Distribusi target pada data pemodelan adalah:

- **Graduate:** 2.209 mahasiswa (60,85%)
- **Dropout:** 1.421 mahasiswa (39,15%)

Data kemudian dibagi secara *stratified* menjadi **2.904 data training** dan **726 data testing**. Distribusi target pada data training dan testing tetap serupa.

Dua model dibandingkan menggunakan *RandomizedSearchCV* dan 5-fold `StratifiedKFold`:

| Model | Best CV Accuracy |
|---|---:|
| Random Forest | **90,46%** |
| XGBoost | **90,84%** |

Berdasarkan hasil cross-validation, **XGBoost dipilih sebagai model final**.

Hasil cross-validation tambahan untuk model XGBoost:

| Metrik | Hasil |
|---|---:|
| Accuracy | **90,84%** |
| Precision | **92,51%** |
| Recall | **83,38%** |
| F1-Score | **87,70%** |

Setelah pemilihan model, XGBoost dievaluasi satu kali pada data testing:

| Metrik | Hasil |
|---|---:|
| Accuracy | **92,98%** |
| Macro Precision | **92,56%** |
| Macro Recall | **92,72%** |
| Macro F1 | **92,64%** |
| Weighted F1 | **92,98%** |
| Precision kelas Dropout | **90,59%** |
| Recall kelas Dropout | **91,55%** |
| F1-score kelas Dropout | **91,07%** |

Recall kelas `Dropout` sebesar **91,55%** menunjukkan proporsi data Dropout pada data testing yang berhasil dikenali oleh model. Precision kelas `Dropout` sebesar **90,59%** menunjukkan proporsi prediksi Dropout yang benar-benar termasuk kelas Dropout pada data testing.

Metrik tersebut menggambarkan performa model pada data pengujian dan bukan jaminan bahwa setiap prediksi individu akan selalu benar.

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
2.  Buka bagian **Cari Student untuk Screening** pada halaman utama.
3.  Masukkan **Student Index**.
4.  Klik **Terapkan Model**.
5.  Sistem mengambil data mahasiswa berdasarkan posisi baris pada data
    Enrolled.
6.  Sistem melakukan *feature engineering*.
7.  Model melakukan prediksi status mahasiswa.
8.  Sistem menampilkan Student Index, Predicted Status, Dropout
    Probability, Risk Level, probabilitas setiap status, dan ringkasan
    data mahasiswa yang dipilih.

### File yang Dibutuhkan untuk Prototype

Struktur minimum repository untuk menjalankan prototype:

```text
project/
├── app.py
├── requirements.txt
├── student_dropout_model.pkl
├── model_metadata.pkl
└── Data/
    └── data.csv
```

`student_dropout_model.pkl` adalah objek model XGBoost binary yang digunakan
untuk `predict()` dan `predict_proba()`. Aplikasi memvalidasi bahwa model
memiliki dua kelas, yaitu `0` dan `1`, sebelum fitur screening diaktifkan.

## Conclusion

### 1. Kesimpulan Berdasarkan Hasil Analisis Data

Hasil EDA menunjukkan adanya perbedaan karakteristik akademik antarstatus mahasiswa. Berdasarkan ringkasan statistik per status, mahasiswa **Dropout** memiliki rata-rata `Curricular_units_1st_sem_approved` sebesar **2,55**, sedangkan `Graduate` sebesar **6,23**. Pada semester kedua, rata-rata `Curricular_units_2nd_sem_approved` pada Dropout sebesar **1,94**, sedangkan Graduate sebesar **6,18**.

Perbedaan juga terlihat pada rata-rata nilai akademik. Rata-rata `Curricular_units_1st_sem_grade` pada Dropout sebesar **7,26**, sedangkan Graduate sebesar **12,64**. Pada semester kedua, rata-rata `Curricular_units_2nd_sem_grade` pada Dropout sebesar **5,90**, sedangkan Graduate sebesar **12,70**.

Hasil tersebut menunjukkan bahwa karakteristik yang perlu menjadi perhatian dalam monitoring dropout terutama berkaitan dengan **performa akademik, jumlah mata kuliah yang disetujui, dan progres akademik antarsemester**. EDA juga menganalisis faktor kategorikal seperti status pembayaran, status debtor, beasiswa, pendaftaran, course, dan karakteristik mahasiswa untuk melihat perbedaan dropout rate antar kelompok.

Selain faktor akademik, informasi administratif dan ekonomi seperti **status pembayaran tuition fees, status debtor, dan scholarship holder** dapat digunakan sebagai konteks tambahan dalam proses monitoring dan pendampingan.

Temuan EDA dan dashboard bersifat **deskriptif**. Oleh karena itu, perbedaan antar kelompok tidak dapat langsung diartikan sebagai hubungan sebab-akibat. Hasil analisis lebih tepat digunakan untuk mengidentifikasi kelompok atau indikator yang perlu dipantau lebih lanjut.

Secara keseluruhan, dashboard memberikan gambaran kuantitatif mengenai
distribusi status mahasiswa dan indikator yang berkaitan dengan dropout,
sedangkan prototype Streamlit melanjutkan hasil analisis tersebut ke tahap
screening berbasis model binary.

### 2. Kesimpulan Berdasarkan Hasil Machine Learning

Model final yang digunakan adalah **XGBoost** dengan target binary classification:

- `0` = **Graduate**
- `1` = **Dropout**

Data `Enrolled` tidak digunakan sebagai kelas training. Sebanyak **3.630** data Graduate dan Dropout digunakan untuk pemodelan, sedangkan **794** data Enrolled disimpan terpisah untuk kebutuhan screening/prediksi pada prototype.

XGBoost dipilih setelah dibandingkan dengan Random Forest menggunakan 5-fold `StratifiedKFold`. XGBoost memperoleh **CV Accuracy 90,84%**, sedikit lebih tinggi dibandingkan Random Forest sebesar **90,46%**.

Pada data testing, XGBoost memperoleh **Accuracy 92,98%**, **Precision kelas Dropout 90,59%**, **Recall kelas Dropout 91,55%**, dan **F1-score kelas Dropout 91,07%**. Nilai **Macro F1 sebesar 92,64%** menunjukkan performa rata-rata model yang baik pada kedua kelas.

### Interpretasi Feature Importance

Notebook juga menampilkan *feature importance* dari model XGBoost. Fitur dengan nilai importance tertinggi adalah:

| Peringkat | Fitur | Importance |
|---:|---|---:|
| 1 | `Approval_Rate_2nd_Sem` | **0,176687** |
| 2 | `Curricular_units_2nd_sem_approved` | **0,042873** |
| 3 | `Tuition_fees_up_to_date_1` | **0,035203** |
| 4 | `Approval_Rate_1st_Sem` | **0,034774** |
| 5 | `Tuition_fees_up_to_date_0` | **0,029168** |
| 6 | `Curricular_units_2nd_sem_enrolled` | **0,021461** |
| 7 | `Curricular_units_1st_sem_enrolled` | **0,017717** |
| 8 | `Course_171` | **0,016944** |
| 9 | `Course_9556` | **0,015224** |
| 10 | `Total_Approved` | **0,015141** |

Berdasarkan *feature importance*, `Approval_Rate_2nd_Sem` merupakan fitur dengan nilai importance tertinggi pada model final. Beberapa fitur akademik semester kedua, jumlah mata kuliah yang disetujui, approval rate semester pertama, dan status pembayaran juga termasuk dalam fitur dengan importance tinggi.

Nilai *feature importance* merupakan **interpretasi model**, bukan bukti bahwa fitur tersebut secara langsung menyebabkan mahasiswa mengalami dropout. Hubungan sebab-akibat tidak dapat disimpulkan hanya dari nilai importance.

Prototype Streamlit menggunakan model yang telah disimpan dalam `student_dropout_model.pkl` untuk melakukan screening terhadap mahasiswa `Enrolled`. Sistem menampilkan hasil prediksi, probabilitas `Dropout`, dan `Risk Level` sebagai informasi awal yang tetap perlu diverifikasi oleh pihak institusi.

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
