# Aktivitas Transaksi Digital yang Tidak Wajar

Deteksi transaksi fraud pada layanan *mobile money* dengan *machine learning* klasik.
Dataset: PaySim. Repositori ini memuat luaran **UTS Milestone 1**.

## Identitas

| | |
|---|---|
| **Project ID** | B12 |
| **Kelompok** | Kelompok 3 |
| **Anggota** | 1. Muhammad Aswan Taqiy (231112002)<br>2. Steven Raffael Antonius Sidauruk (231112060)<br>3. William Patar Wijaya Marpaung (231112393) |

## Deskripsi masalah

Layanan *mobile money* memproses jutaan transaksi, dan hanya sebagian sangat kecil yang merupakan penipuan. Pada dataset PaySim, fraud hanya 0.13% dari 6,362,620 transaksi. Skema yang disimulasikan: pelaku mengambil alih akun, memindahkan saldo ke akun lain (`TRANSFER`), lalu mencairkannya (`CASH_OUT`). Dana yang sudah keluar dari sistem sulit dipulihkan, sehingga deteksi harus terjadi **sebelum dana berpindah**.

| | |
|---|---|
| **Tujuan** | Memberi skor risiko fraud pada setiap transaksi saat transaksi diajukan |
| **Target** | `isFraud` (1 = fraud, 0 = normal) |
| **Pendekatan** | Klasifikasi biner, *supervised learning* klasik |
| **Lingkup** | Tipe `TRANSFER` dan `CASH_OUT`, satu-satunya tipe yang memuat fraud |
| **Metrik utama** | PR-AUC, karena kelas positif sangat langka; akurasi tidak dipakai |

Dua keputusan desain menentukan hasil proyek ini:

1. **Empat kolom saldo tidak dipakai pada skenario utama.** Dokumentasi dataset menyatakan kolom `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, dan `newbalanceDest` tidak boleh dipakai untuk deteksi fraud. EDA juga menunjukkan kolom ini membawa pola simulator: aturan satu baris `amount == oldbalanceOrg` sudah mencapai PR-AUC 0.985.
2. **Evaluasi mengikuti urutan waktu.** Model dilatih dari masa lalu dan diuji pada masa depan. Data uji (*hold-out*) hanya dipakai satu kali, setelah pilihan model dibekukan.

## Ringkasan hasil awal

Skenario utama tanpa kolom saldo. CV = validasi silang berbasis waktu, 5 fold, pada periode latih (step 1–354). Hold-out = step 355–743.

| Model | Peran | PR-AUC CV (rata-rata ± sd) | PR-AUC hold-out | ROC-AUC hold-out |
|---|---|---|---|---|
| Dummy (prior) | Referensi | 0.0019 ± 0.0010 | 0.0077 | 0.5000 |
| Aturan `isFlaggedFraud` | Referensi | 0.0029 ± 0.0016 | 0.0107 | 0.5015 |
| Logistic Regression | Baseline | 0.1986 ± 0.0177 | 0.3305 | 0.8714 |
| Decision Tree | Kandidat | 0.2684 ± 0.1167 | 0.5240 | 0.8767 |
| Random Forest | Kandidat | 0.3017 ± 0.1413 | 0.5616 | 0.9407 |
| **Hist. Gradient Boosting** | **Terpilih** | **0.3727 ± 0.0398** | **0.5661** | **0.9424** |

Pada hold-out, model terpilih menangkap 1,407 dari 4,258 fraud (recall 0.33) dengan precision 0.88. Hasil lengkap ada di `results/tabel_hasil_eksperimen.csv`, dan pembahasannya di laporan.

> Skor nyaris sempurna pada notebook Pertemuan 4 (F1 Random Forest 0.985) **ditarik**. Skor itu berasal dari kolom saldo dan pembagian data acak, bukan dari kemampuan model. Penjelasannya ada di laporan bagian H.6–H.7.

## Struktur repositori

```
├── README.md                         ← berkas ini
├── requirements.txt                  ← versi paket yang dipakai
├── UTS_Milestone1_PaySim.ipynb       ← notebook utama Milestone 1 (sumber semua angka)
├── Project_Task_02_PaySim.ipynb      ← riwayat kerja Pertemuan 2–4 (bukan sumber angka laporan)
├── results/                          ← dihasilkan notebook
│   ├── tabel_hasil_eksperimen.csv    ← tabel hasil eksperimen awal (semua eksperimen E1–E6)
│   ├── cv_per_fold.csv               ← skor setiap fold
│   ├── holdout.csv                   ← hasil evaluasi akhir
│   ├── metrics.json                  ← angka yang dikutip laporan
│   └── ...                           ← tabel EDA, kualitas data, diagnosis

```

## Dataset

| | |
|---|---|
| Nama | *Synthetic Financial Datasets For Fraud Detection* (PaySim), versi 2 |
| Pembuat | Edgar Alonso Lopez-Rojas, Ahmad Elmir, Stefan Axelsson |
| Sumber | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Lisensi | CC BY-SA 4.0: boleh dipakai dan dibagikan dengan mencantumkan atribusi ke pembuat; karya turunan memakai lisensi yang sama |
| Sitasi | Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. *28th European Modeling and Simulation Symposium (EMSS)*, Larnaca, Cyprus. |
| Berkas | `PS_20174392719_1491204439457_log.csv` (±494 MB, 6,362,620 baris, 11 kolom) |
| SHA-256 | `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b` |

CSV tidak disimpan di repositori karena ukurannya (`*.csv` ada di `.gitignore`, kecuali `results/*.csv`). Notebook mencari berkas dengan urutan berikut, jadi pilih salah satu cara:

1. **Variabel lingkungan:** isi `PAYSIM_CSV` dengan path lengkap ke CSV.
2. **Folder proyek:** unduh dari Kaggle, ekstrak, lalu letakkan CSV di folder ini atau di `data/`.
3. **Otomatis:** jika tidak ditemukan, notebook mengunduh versi 2 lewat `kagglehub`. Mungkin perlu akun atau token API Kaggle.

Notebook mencetak SHA-256 berkas yang dibaca dan membandingkannya dengan nilai di atas.

## Cara menjalankan

**1. Siapkan lingkungan (Python 3.13)**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

**2. Siapkan dataset** (lihat bagian Dataset).

**3. Jalankan notebook** dari awal sampai akhir dengan *Restart & Run All* di Jupyter atau VS Code, atau lewat terminal:

```bash
pip install nbconvert
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UTS_Milestone1_PaySim.ipynb
```

Satu kali jalan memakan ±20–25 menit pada laptop Intel i7-13650HX (20 thread) dengan RAM 16 GB. Notebook menulis ulang seluruh isi `results/` dan `figures/`.


## Catatan penggunaan

- **Urutan notebook:** bagian A–I pada notebook sama dengan bagian A–I pada laporan.
- **Reproduksibilitas:** semua model memakai `random_state=42`. Random Forest dijalankan paralel, sehingga skornya bisa berbeda di sekitar desimal keenam antar-jalan (contoh: PR-AUC CV 0.3016738 vs 0.3016728). Ini tidak mengubah angka yang dilaporkan dengan empat desimal.
- **Data uji sudah dipakai satu kali.** Pekerjaan setelah UTS harus mengambil semua keputusan (fitur, tuning, ambang) dari CV pada periode latih, dan baru mengevaluasi hold-out sekali lagi di akhir.
- **`Project_Task_02_PaySim.ipynb`** disimpan sebagai riwayat. Hasil pemodelannya memakai kolom saldo dan pembagian acak, sehingga tidak dipakai lagi.

## Penggunaan alat bantu AI

Claude Code (Anthropic) dipakai sebagai alat bantu untuk memperbaiki kode, serta melakukan push ke GitHub. Seluruh angka berasal dari eksekusi notebook. Kelompok memeriksa hasilnya dan bertanggung jawab atas ketepatan isi, analisis, kode, dan hasil yang dikumpulkan.



