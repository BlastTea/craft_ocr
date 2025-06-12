# Craft OCR Testing Dataset

Dataset ini dibuat untuk menguji performa model Craft OCR dalam berbagai kondisi nyata. Terdiri dari 10 kategori dengan satu atau lebih gambar uji pada masing-masing folder.

## Struktur Folder

- `01_clear_text/`: Teks lurus dan bersih, ideal.
- `02_complex_background/`: Teks berada di background ramai atau penuh warna.
- `03_curved_or_slanted_text/`: Teks melengkung, miring, atau tidak lurus.
- `04_rotated_text/`: Teks dalam gambar yang diputar.
- `05_small_text/`: Teks ukuran kecil, sulit dibaca.
- `06_low_light_or_high_contrast/`: Gambar dengan pencahayaan buruk atau kontras ekstrem.
- `07_multilanguage_or_symbols/`: Teks dalam berbagai bahasa atau simbol.
- `08_handwritten_or_stylized/`: Tulisan tangan atau teks dengan font artistik.
- `09_textured_background/`: Teks berada di atas permukaan seperti kayu, kain, batu.
- `10_realworld_camera_noise/`: Gambar dengan gangguan seperti blur, noise, low resolution.
- `11_no_text/`: Gambar yang tidak mengandung teks sama sekali. Tujuannya adalah untuk menguji apakah Craft OCR secara keliru mendeteksi "teks palsu" (false positives).


## Cara Menggunakan

1. Jalankan Craft OCR pada masing-masing gambar.
2. Simpan hasil bounding box/text detection ke format JSON/TXT.
3. Bandingkan hasil dengan ground truth (jika tersedia).
4. Analisis keberhasilan deteksi per kategori.

## Lisensi
Gunakan hanya untuk pengujian dan penelitian pribadi. Pastikan hak cipta gambar digunakan dengan benar.
