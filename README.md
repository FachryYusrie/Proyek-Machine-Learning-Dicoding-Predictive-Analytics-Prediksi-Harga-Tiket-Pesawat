# Laporan Proyek Machine Learning - Muhammad Lailfachry Yusrie
## 1. Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Ekonomi dan Bisnis**, dengan judul **Predictive Analytics: Harga Tiket Pesawat**  

### 1.1 Latar Belakang

![Ilustrasil](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/99946880c23ccc2413dad77504644cefc33d7754/image/Ilustrasi.png)

Industri penerbangan global telah mengalami pertumbuhan yang signifikan, menjadikan perjalanan udara sebagai moda transportasi yang semakin mudah diakses dan penting bagi individu maupun bisnis. Di tengah dinamika pasar yang kompleks, harga tiket pesawat seringkali berfluktuasi secara drastis, dipengaruhi oleh berbagai faktor seperti waktu keberangkatan, rute, maskapai penerbangan, kelas penerbangan, musim, hingga dinamika penawaran dan permintaan. Volatilitas harga ini menjadi tantangan tersendiri bagi konsumen yang ingin merencanakan perjalanan dengan efisien dan hemat biaya. Tanpa informasi yang akurat mengenai estimasi harga yang wajar, penumpang berisiko membayar lebih mahal dari yang seharusnya atau kesulitan dalam mengalokasikan anggaran perjalanan secara optimal [[1](https://kumparan.com/jendela-dunia/kenapa-harga-tiket-pesawat-naik-turun-cari-tahu-jawabannya-di-sini-212Dwje2OR3)].
Pengembangkan model prediktif harga tiket pesawat ini diharapkan dapat memberikan estimasi harga yang akurat dan relevan bagi travel agent. Dengan memanfaatkan teknik machine learning berbasis regresi pada kumpulan data pemesanan tiket pesawat, model ini diharapkan mampu mengidentifikasi pola dan hubungan antara berbagai fitur perjalanan (maskapai, waktu keberangkatan, jumlah transit, durasi penerbangan, kota asal/tujuan, kelas, dan sisa hari sebelum keberangkatan) dengan harga tiket. Informasi prediksi harga ini akan memberdayakan konsumen untuk mengambil keputusan yang lebih cerdas. Penumpang dapat membandingkan harga yang ditawarkan oleh maskapai atau agen perjalanan dengan harga prediksi model, sehingga memungkinkan mereka untuk mengidentifikasi apakah harga yang ditawarkan lebih mahal atau lebih murah dari ekspektasi.

### 2. Business Understanding
Pengembangan model prediksi harga tiket pesawat memiliki potensi untuk memberikan manfaat bagi berbagai pihak, termasuk calon penumpang, travel agent, dan maskapai penerbangan. Model ini dapat membantu mengestimasi harga tiket dari data yang ada, menjual tiket pesawat dengan harga yang optimal, dan meningkatkan kepercayaan penumpang. Contoh potensi manfaat hasil prediksi harga tiket pesawat yang akurat dapat membantu travel agent dan penumpang dalam menetapkan harga tiket yang tepat dan penumpang dapat merencanakan keuangan untuk perjalanannya dengan lebih tepat sasaran.

### 2.1 Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
-  Bagaimana membuat model machine learning yang dapat memprediksi harga tiket pesawat berdasarkan data harga tiket sebelumnya dan faktor yang mempengaruhi?
-  Model yang seperti apa yang memiliki akurasi paling baik?
-  Bagaimana model ini dapat membantu travel agent dan penumpang dalam mengestimasi harga tiket pesawat dengan tepat?
  
### 2.2 Goals
Tujuan dari proyek ini adalah:
- Membuat model machine learning yang dapat memprediksi harga tiket pesawat berdasarkan data harga tiket sebelumnya.
- Membandingkan beberapa algoritma model untuk menemukan akurasi terbaik dalam memprediksi harga tiket.
- Mengembangkan aplikasi yang mudah digunakan untuk membantu penumpang mengestimasi harga tiket pesawat dan mengimplmentasikan model prediktif kedalam sistem yang digunakan travel agent.

### 2.3 Solution Statements
-  Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi matrix antar fitur dan mendeteksi outlier.
- Melakukan proses data cleaning dan normalisai data agar mendapat prediksi yang baik.
- Membuat beberapa variasi model untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk prediksi harga tiket pesawat. Diantaranya adalah menggunakan:
    * Linear Regression adalah algoritma sederhana yang memodelkan hubungan antara satu atau lebih feature dan target variable dengan cara menemukan best-fit line yang meminimalkan selisih antara nilai yang diprediksi dan nilai aktual.[[2](https://www.nucleusbox.com/assumptions-of-linear-regression/)]
    * Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. Random forest menggabungkan prediksi estimator untuk menghasilkan prediksi yang lebih akurat .[[3](https://deepai.org/machine-learning-glossary-and-terms/random-forest)]
    * XGBoost (Extreme Gradient Boosting) adalah algoritma yang menggunakan pendekatan boosting, di mana model dibangun secara bertahap, dengan setiap model baru dilatih untuk memperbaiki kesalahan prediksi dari model sebelumnya.[[4](https://xgboost.readthedocs.io/en/stable/)]

## 3. Data Understanding
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | _flight price prediction_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?select=Clean_Dataset.csv) |
| Maintainer | [Shubham Bathwal ⚡](https://www.kaggle.com/shubhambathwal) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Computer Science, Business, Economics, Data Visualization, Regression, Exploratory Data Analysis_ |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah perusahaan pertanian Amerika, yang disediakan secara publik di kaggle dengan nama datasets yaitu: _Apple Quality_

| Unnamed | airline | flight | source_city | departure_time | stops | arrival_time | destination_city | class | duration | days_left | price |
| ------ | ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ | ------ | ------ |
| 0 | SpiceJet | SG-8709 | Delhi | Evening	| zero | Night | Mumbai | Economy | 2.17 | 1 | 5953 |
| 1 | SpiceJet | SG-8157 | Delhi | Early_Morning | zero | Morning | Mumbai | Economy | 2.33 | 1 | 5953 |
| 2 | AirAsia | I5-764 | Delhi | Early_Morning | zero | Early_Morning | Mumbai | Economy | 2.17 | 1 | 5956 |
| 3 | Vistara | UK-995 | Delhi | Morning | zero | Afternoon | Mumbai | Economy | 2.25 | 1 | 5955 |
| 4 | Vistara | UK-963 | Delhi | Morning | zero | Morning | Mumbai | Economy | 2.33 | 1 | 5955 |


Tabel 3.1 5 Data Pertama dari dataset

Karena kolom 'Unnamed' hanya merepresentasikan nomor baris pada dataset, maka kolom ini dihapus dari dataset agar model prediksi yang dihasilkan tidak bias. Setelah kolom 'Unnamed' dihapus dari dataset, terdapat beberapa informasi yang diperoleh mengenai dataset diantaranya: 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 300153 sample dengan 11 fitur.
- Dataset memiliki 3 fitur bertipe muneric (2 int64 dan 1 float64) dan 8 fitur bertipe object.

### 3.1 EDA - Deskripsi Variabel 
Berdasarkan _Tabel 3.1_, fitur/variabel yang terdapat pada dataset sebagai berikut:
1) 'airline': Nama perusahaan penerbangan disimpan di kolom maskapai penerbangan. Ini adalah fitur kategoris yang memiliki 6 maskapai penerbangan yang berbeda.
2) 'flight': Penerbangan menyimpan informasi mengenai kode penerbangan pesawat. Ini adalah fitur kategoris.
3) 'source_city': Kota tempat penerbangan lepas landas. Ini adalah fitur kategoris yang memiliki 6 kota unik.
4) 'departure_time': Ini adalah fitur kategoris turunan yang diperoleh dengan mengelompokkan periode waktu ke dalam kelompok. Ini menyimpan informasi tentang waktu keberangkatan dan memiliki 6 label waktu yang unik.
5) 'stops': Fitur kategoris dengan 3 nilai berbeda yang menyimpan jumlah perhentian antara kota asal dan tujuan.
6) 'arrival_time': Ini adalah fitur kategoris turunan yang dibuat dengan mengelompokkan interval waktu ke dalam kotak. Fitur ini memiliki enam label waktu yang berbeda dan menyimpan informasi tentang waktu kedatangan.
7) 'destinationc_city': Kota tempat pesawat akan mendarat. Ini adalah fitur kategoris yang memiliki 6 kota unik.
8) 'class': Fitur kategoris yang berisi informasi tentang kelas kursi; fitur ini memiliki dua nilai yang berbeda: Bisnis dan Ekonomi.
9) 'duration': Fitur berkelanjutan yang menampilkan jumlah waktu keseluruhan yang diperlukan untuk bepergian antar kota dalam hitungan jam.
10) 'days_left': Ini adalah karakteristik turunan yang dihitung dengan mengurangi tanggal perjalanan dengan tanggal pemesanan.
11) 'price': Variabel target menyimpan informasi tentang harga tiket (dalam kurs India/Rupe).

### 3.2 EDA - Menangai Missing Value dan Outliers
Setelah dilakukan pengujian, tidak terdapat missing value pada dataset, tetapi terdapat outliers yang harus ditangani. Bagian ini akan dijelaskan lebih detail pada bagian **Data Preparation**.

### 3.3 EDA - Univariate Analysis

![univariate analysis](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/6f0dc626109148ef907d73e9b200c4644ef77890/image/Analisis_Univariate_Fitur_Kategorik_ver2.png)

Gambar 3.1 Analisis Univariat (Data Kategori) 

![Univariate Analysis](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/ab189082713b5ab9395e128dac2c808692935472/image/Analisis_Univariate_Fitur_Numerik.png)

Gambar 3.2 Analisis Univariat (Data Numerik) 

Berdasarkan _Gambar 3.1_, diketahui bahwa Vistara merupakan maskapai yang paling populer di kalangan penumpang, sementara SpiceJet memiliki jumlah penumpang terendah. Hal ini menunjukkan bahwa Vistara mungkin memiliki pangsa pasar yang lebih besar dibandingkan dengan maskapai lainnya. Terlihat juga bahwa begitu banyak nomor kode penerbangan yang terdapat pada data penerbangan. Selain itu, banyak penerbangan yang hanya membutuhkan satu kali penerbangan tanpa perlu melakukan transit. Lalu kelas ekonomi merupakan pilihan yang paling disukai oleh para pelancong, secara signifikan melampaui popularitas kelas bisnis dan kota Delhi menjadi kota asal dengan jumlah penerbangan tertinggi, diikuti oleh Mumbai dan Bangalore. Tren ini dapat dikaitkan dengan kota metropolitan besar seperti Delhi yang memiliki bandara yang lebih besar dan volume penerbangan yang lebih besar. Kemudian Kota Mumbai menjadi kota tujuan paling populer untuk penerbangan, diikuti oleh Delhi dan Bangalore di belakangnya. Didapatkan bahwa pagi dan dini hari adalah waktu keberangkatan puncak, menunjukkan preferensi di antara penumpang untuk memulai perjalanan mereka di pagi hari. Sebaliknya, malam dan sore hari adalah waktu kedatangan yang paling umum, menunjukkan bahwa penumpang lebih menyukai perjalanan di malam hari untuk mencapai tujuan mereka.

Berdasarkan _Gambar 3.2_, untuk data numerik memiliki karakteristik, yaitu distribusi kolom durasi menunjukkan kecenderungan yang signifikan. Visualisasi menunjukkan bahwa penerbangan dengan durasi antara 0-5 jam memiliki frekuensi tertinggi, menunjukkan bahwa rentang ini adalah yang paling umum dalam dataset.
Distribusi variabel harga juga menunjukkan tanda-tanda skewness.

### 3.4 EDA - Bivariate Analysis

![Bivariate Analysis](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/6f0dc626109148ef907d73e9b200c4644ef77890/image/Analisis_Bivariate_Fitur_Kategorik_ver2.png)


Gambar 3.3 Analisis Biivariat (Kategorikal)

![Bivariate Analysis](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/ab189082713b5ab9395e128dac2c808692935472/image/Analisis_Bivariate_Fitur_Numerik.png)


Gambar 3.4 Analisis Bivariate (Numerikal)

![Correlation Matrix](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/ab189082713b5ab9395e128dac2c808692935472/image/Correlation_Metrix.png)


Gambar 3.5 Analisis Matriks Korelasi

Berdasarkan _Gambar 3.3_ , dapat dilihat bahwa dari analisis bivariate untuk fitur kategori terhadap harga tiket didapatkan:

- Price by Airline: Vistara memiliki tarif penerbangan rata-rata tertinggi di antara semua maskapai penerbangan, yang mengindikasikan bahwa tiket Vistara secara umum lebih mahal dibandingkan dengan para pesaingnya. Di sisi lain, SpiceJet, GO_FIRST, Indigo, dan AirAsia menawarkan tarif penerbangan yang relatif lebih rendah, sehingga menjadi pilihan yang lebih terjangkau bagi para penumpang.
- Price by Flight: Kode penerbangan yang termahal secara mahal mungkin mengindikasi penerbangan yang menggunakan airline dari Vistara.
- Price by Class: Harga tiket kelas bisnis secara signifikan lebih tinggi dibandingkan kelas ekonomi. Harga tiket kelas ekonomi lebih terjangkau, sehingga menjadi pilihan yang populer bagi sebagian besar penumpang.
- Price by Total Stops: Penerbangan dengan jumlah stop >= 2 cenderung lebih mahal dibandingkan penerbangan non-stop. Temuan ini cukup menarik karena biasanya penerbangan nonstop akan cenderung lebih pricey. Hal ini mungkin dapat disebabkan karena mayoritas penerbangan nonstop adalah penerbangan jarak dekat yang memiliki harga tiket lebih rendah.
- Price by Source City: Kalkota, Chennai dan Bangalore menjadi kota keberangkatan dengan harga tiket pesawat paling tinggi.
- Price by Destination City: Chennai, Kolkata, dan Mumbai menjadi kota tujuan penerbangan dengan harga tiket pesawat paling tinggi.
- Price by Departure dan Arrival Time: Harga tiket pesawat untuk penerbangan atau kedatangan tengah malam cenderung lebih murah dibandingkan waktu lainnya.

Berdasarkan _Gambar 3.4_, dapat dilihat bahwa Dari analisis bivariate untuk fitur numerikal terhadap harga tiket untuk kelas ekonomi dan bisnis didapatkan:
- Price by Duration: Kelas Bisnis memiliki kisaran harga yang jauh lebih tinggi dibandingkan dengan Kelas Ekonomi untuk hampir semua durasi penerbangan. Umumnya, harga untuk Kelas Ekonomi di bawah 400 USD, sedangkan Kelas Bisnis bisa mencapai harga di atas 1200 USD. Durasi penerbangan yang lebih lama tidak selalu berkorelasi langsung dengan harga yang lebih tinggi. Meskipun harga cenderung lebih tinggi untuk penerbangan yang lebih lama, terbukti bahwa harga tertinggi sedikit menurun untuk beberapa durasi yang sangat lama, terutama di Kelas Bisnis. Hal ini menunjukkan bahwa faktor selain durasi, seperti permintaan atau rute tertentu, juga dapat memengaruhi dinamika harga untuk penerbangan yang lebih lama.
- Price by Days Left: Tidak ada korelasi yang kuat antara harga tiket dan jumlah hari yang tersisa sebelum penerbangan, baik untuk Kelas Bisnis maupun Kelas Ekonomi. Oleh karena itu, tidak dapat diasumsikan bahwa membeli tiket lebih awal akan menghasilkan harga yang lebih murah. Namun, ada tren yang menunjukkan bahwa harga tiket cenderung naik sedikit ketika mendekati tanggal keberangkatan, meskipun harga tetap cukup bervariasi secara keseluruhan. Hal ini menunjukkan bahwa meskipun membeli tiket di menit-menit terakhir tidak selalu menghasilkan harga yang lebih tinggi secara signifikan, harga tiket dapat berfluktuasi berdasarkan permintaan dan ketersediaan.

Sehingga dapat dlihat pada _Gambar 3.5_, yang merupakan Correlation Matrix, menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur duration memiliki skor korelasi yang cukup besar 0.22 dengan fitur target price. Sedangkan fitur days left berkolerasi negatif terhadap price.
  
## 4. Data Preparation
Pada proses _Data Preparation_ dilakukan kegiatan seperti _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. Pada proses Data Gathering, data diimpor dari _kaggle_ sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Setelah melalui proses _Data Gathering_, ditemukan adanya kolom 'Unnamed' yang merepresentasikan nomor baris pada dataset, yang mana hal ini tidak diperlukan dalam model prediksi sehingga dihapus dari dataset. Selain itu, juga dilakukan rekayasa fitur dengan menambahkan fitur baru 'price_usd' hasil konversi harga tiket pada fitur 'price' dari Rupe (India) ke USD (Amerika Serikat). Kemudian fitur 'price' asli dihapus dari dataset sehingga fitur 'price_usd' menjadi target dalam supervised learning kasus regresi.

Selanjutnya, untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
- Melakukan pengecekan terhadap _Missing value_ (data atau informasi yang "hilang" atau tidak tersedia).
- Melakukan pengecekan terhadap keberadaan Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Pada proyek kasus ini tidak ditemukannya _missing value_. Namun, setelah pengecekan, ditemukan adanya _outlier_ metode _dropping_ menggunakan metode IQR.  IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

$$IQR = Q_3 - Q_1$$

- Q1 adalah kuartil pertama 
- Q3 adalah kuartil ketiga.

Setelah menggunakan metode IQR untuk menghilangkan _outlier_, jumlah dataset berkurang menjadi `297920` yang awalnya adalah `300153` data.

Kemudian, dilanjutkan dengan proses _Data Cleaning_ sebagai berikut:
- Feature Selection dan Feature Engineering (Memilih fitur yang relevan untuk prediksk model dan membuat fitur baru dari rekayasa fitur sebelumnya). 
- Encoding dengan One Hot Encodier (Mengubah tipe katergorik menjadi numerik).
- Train Test Split (membagi data menjadi data latih dan data uji).
- Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding).

Dari 11 fitur terdapat pada dataset, melalui proses _feature selection_, yang akan dipakai hanya fitur 'departure_time', 'stops', arrival_time', 'class', 'duration', 'days_left', dan 'price' sebagai target. Fitur yang lain tidak dipakai karena datanya spesifik hanya berada di India. Oleh karena itu agar predictive model lebih relevan dan dapat digunakan oleh berbagai negara, maka data selain fitur diatas dihapus. Selain itu, juga dilakukan _feature engineering_ terhadap Fitur 'price' berisi harga tiket dalam mata uang India (Rupe), di mana agar prediksi lebih general dan bisa digunakan berbagai negara, maka dibuat kolom baru dengan 'price_usd' yang merupakan hasil konversi kolom 'price' yang menggunakan Rupe menjadi harga tiket dengan nilai dalam uang Amerika Serikat (USD). Kemudian tipe datanya adalah menjadi 'float64' karena hasil konversi dari Rupe ke US Dolar. Setelah itu, kolom 'price' dihapus dari dataset.

Berikut 5 data pertama setelah menghapus beberapa fitur yang tidak akan dipakai dan kolom baru 'price_usd' yang sudah dibuat sebagai berikut:

| departure_time | stops | arrival_time | class | duration | days_left | price_usd |
| ------ | ------ | ------ |------ | ------ |------ | ------ |
| Evening	| zero | Night | Economy | 2.17 | 1 | 71.436 |
| Early_Morning | zero | Morning | Economy | 2.33 | 1 | 71.436 |
| Early_Morning | zero | Early_Morning | Economy | 2.17 | 1 | 71.472 |
| Morning | zero | Afternoon | Economy | 2.25 | 1 | 71.460 |
| Morning | zero | Morning | Economy | 2.33 | 1 | 71.460 |


Tabel 4.1 5 Data Pertama dari dataset setelah seleksi fitur

Pada proses encoding, metode yang digunakan yaitu menggunakan _One Hot Encoder_, karena fitur kategorikal pada dataset tidak bersifat "Ordinal" atau tidak memiliki tingkatan sehingga metode ini akan lebih baik dalam mencegah bias dalam model prediksi. Kemudian, pada proyek ini digunakan _Train Test Split_ pada library *sklearn.model_selection* untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 10:90 dan random state sebesar 123. Pada proyek kasus ini menggunakan _Normalization_ pada library _sklearn.preprocessing.MinMaxScaler_ untuk menormalisasi dataset agar setiap fitur yang tetah berbentuk numerik memiliki rentang nilai yang sama. Semua proses ini diperlukan dalam rangka membuat model yang baik.

## 5. Modeling
Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu:
- Linear Regression dipilih karena simple dan interpretable dalam menjelaskan hubungan antara fitur dan target.
- Random Forest, sebagai model ensemble, membantu dalam menangani non-linearitas dan interaksi antar fitur dengan memberikan prediksi yang lebih stabil.
- XGBoost dikenal sebagai algoritma boosting yang efisien dan sangat efektif dalam menangani data besar dan kompleks, serta dapat meningkatkan akurasi model.

Sebelum masuk pengembangan model, karena target variabel yang dianalisis adalah numerik kontinu, rata-rata digunakan sebagai baseline model. Dengan membandingkan hasil prediksi model dengan nilai rata-rata dari data, dapat diukur seberapa baik model yang dibangun mampu menjelaskan variasi dalam data dan memberikan prediksi yang lebih akurat daripada sekadar menggunakan rata-rata. Didapatkan bahwa RMSE Baseline dari training set yang ada tanpa pelatihan model sebelumnya yaitu 271.93. Sehingga RSME yang diperoleh dari model yang digunakan akan dibandingkan dengan baseline untuk melihat performa model algoritma.

Untuk tahap pertama, yaitu pemilihan moodel terbaik dari 3 model yang ada dari hasil training yang didapatkan. Berikut penjelasan masing-masing model:
- _Linear Regression_ adalah memodelkan hubungan antara satu atau lebih feature dan target variable dengan cara menemukan best-fit line yang meminimalkan selisih antara nilai yang diprediksi dan nilai aktual. Model Linear Regression terdapat pada library from _sklearn.linear_model import LinearRegression_.

  Keunggulan _Linear Regression_ :
  - Cepat dan ringan — cocok untuk data kecil/menengah.
  - Mudah diinterpretasi — koefisien menunjukkan pengaruh tiap fitur.
  - Cocok untuk masalah linier sederhana.
  - Tidak mudah overfitting jika asumsi terpenuhi.
  
  Kerugian _Linear Regression_ :
  - Asumsi linearitas antara fitur dan target harus terpenuhi. 
  - Tidak bisa menangkap interaksi non-linear antar fitur. 
  - Sensitif terhadap multikolinearitas (fitur saling berkorelasi).
  - Performa rendah pada data kompleks.

 _Random Forest_ adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang sementara diatur pada proyek ini adalah n_estimators dan random_state, sedangkan parameter lainnya yang terdapat pada Random Forest akan mengikuti default dari libray scikit-learn:
- n_estimators = 50 : Jumlah pohon (trees) dalam hutan. Semakin banyak, semakin stabil, tapi lebih lambat. Default: 100
- random_state = 123 : Seed untuk pengacakan, agar hasil dapat direproduksi.

  Keunggulan _Random Forest_ :
  - Memiliki akurasi prediksi yang tinggi.
  - Mampu menangani dataset dengan dimensi tinggi.
  - Tidak sensitif terhadap outlier.
  
  Kerugian _Random Forest_ :
  - Cenderung overfit pada dataset kecil. 
  - Membutuhkan banyak waktu komputasi untuk pelatihan. 
  - Sulit untuk diinterpretasikan.

 _XGBoost (Extreme Gradient Boosting)_ adalah algoritma pembelajaran mesin yang dirancang untuk meningkatkan akurasi dan efisiensi prediksi, terutama dalam konteks regresi dan klasifikasi. Algoritma ini menggunakan pendekatan boosting, di mana model dibangun secara bertahap, dengan setiap model baru dilatih untuk memperbaiki kesalahan prediksi dari model sebelumnya. XGBoost mengoptimalkan fungsi kehilangan dengan menggunakan teknik regularisasi dan pembelajaran paralel untuk meningkatkan performa dan kecepatan. Beberapa parameter yang diatur secara manual diantaranya sebagai berikut:
- objective='reg:squarederror' : Fungsi objektif. Untuk regresi biasa menggunakan : "reg:squarederror".
- random_state = 123 : Untuk reproduksibilitas hasil.  

  Keuntungan  _XGBoost (Extreme Gradient Boosting)_ :
  - Sangat presisi tinggi — sering menang di kompetisi seperti Kaggle.
  - Mendukung regularisasi → mengurangi overfitting
  - Cepat dan efisien (dibanding model boosting lainnya).
  - Sangat baik untuk data besar & kompleks.
  
  Kerugian  _XGBoost (Extreme Gradient Boosting)_ :
  - Lebih kompleks — banyak hyperparameter perlu disetel.
  - Training time lebih lama dari Random Forest (terutama untuk tuning).
  - Bisa overfitting jika tidak dituning dengan baik.

Berikut hasil RMSE Training Set, RMSE Cross Validation, dan $$R^2$$ dari 3 buah model yang latih:

| Model | RMSE Train | RMSE CV | R2 Score |
| ------ | ------ | ------ | ------ |
| Linear Regression | 84.230744 |	84.236595 | 0.904056 |
| RandomForest  | 45.982051 |	67.913056 | 0.971407 |
| XGBoost | 58.431482 |	59.637176 | 0.953829 |

Tabel 5.1 Hasil RMSE Train, RMSE CV, dan $$R^2$$ Score

Dilihat dari _Tabel 5.1 Hasil Hasil RMSE Train, RMSE CV, dan $$R^2$$ Score_, dapat diketahui bahwa model dengan algoritma _Random Forest_ memiliki $$R^2$$ Score yang lebih tinggi sebesar `97.14%` . Untuk itu model tersebut yang akan dipilih untuk digunakan. Namun perlu dilakukan hyperparameter tunning terhadap model Random Forest agar  model tidak overfitting. Hal ini telihat dari RMSE yang didapatkan bahwa RMSE Train memiliki performa yang baik dalam train data latih, tapi RMSE SV lebih besar dari RMSE Train yang menunjukkan adanya potensi overfitting. Untuk melakukan hyperparameter tunning, yaitu mengatur parameter terbaik dapat menggunakan RandomizedSearchCV dari sckit-learn. RandomizedSearchCV adalah teknik untuk mencari kombinasi hyperparameter terbaik dari sebuah model dengan mengambil sejumlah kombinasi acak dari ruang parameter yang kamu tentukan.

Sebelum menggunakan algoritma RandomnizedSearchCV ini, perlu mendefinisikan parameter distribution (_param_dist_) yang akan dicari nilai terbaiknya seperti n_estimators, max_depth, min_samples_split, min_sample_leaf, dan max_features. Kemudian untuk menjalankan algoritma RandomnizedSearchCV, perlu menginisiasi parameter seperti:
- estimator = RandomForestRegressor(random_state=123, n_jobs = -1) : Model machine learning yang ingin dituning.
- param_distributions = param_dist : Dictionary berisi daftar parameter dan nilai-nilai yang mungkin dicoba. Nilainya bisa berupa list atau distribusi probabilitas dari scipy.stats.
- n_iter = 2 : Jumlah kombinasi acak yang akan dicoba dari param_distributions.
- cv = 3 : Jumlah lipatan (fold) untuk validasi silang (cross-validation).
- scoring = 'neg_mean_squared_error' : Metrik evaluasi yang digunakan untuk memilih model terbaik.
- verbose = 2 Level tampilan log proses selama fitting.
- random_state = 123 :  Seed acak untuk reprodusibilitas hasil pencarian.
- n_jobs = -1 : Jumlah core CPU yang digunakan.
Setelah melakukan hyperparameter tunning dengan RandomnizedSearchCV, didapatkan nilai terbaik dari parameter distribution yang dicari dari Random Forest sebagai berikut:
- max_depth=30 : Kedalaman maksimum pohon. Jika None, pohon akan tumbuh sampai semua daun bersifat murni.
- max_features='sqrt' : Jumlah fitur maksimum yang dipertimbangkan saat split. Ex: "auto", "sqrt", "log2"
- min_samples_leaf=4 : Jumlah minimum sampel di daun. Membantu mencegah overfitting.
- min_samples_split=8 : Jumlah minimum sampel untuk membagi node. Default: 2.
- n_estimators=196 : Jumlah pohon (trees) dalam hutan. Semakin banyak, semakin stabil, tapi lebih lambat. Default: 100
- n_jobs=-1 : Jumlah core CPU yang digunakan. -1 untuk semua core.
- random_state=123 : Seed untuk pengacakan, agar hasil dapat direproduksi.

## 6. Evaluation
Dalam tahap evaluasi, metrik yang digunakan untuk melihat performa model seperti pada bagian **Modelling** adalah RMSE untuk Training Set, RMSE untuk Cross Validation, dan $$R^2$$ Score yang menjadi indikator performa model. R² Score adalah sebuah metrik yang digunakan untuk mengevaluasi seberapa baik model regresi dapat menjelaskan variasi dari data target (nilai aktual). Root Mean Square Error (RMSE) adalah metrik evaluasi untuk mengukur seberapa besar rata-rata kesalahan prediksi sebuah model regresi terhadap nilai aktual, dalam satuan yang sama dengan target, sedangkan Nilai R² menunjukkan persentase variasi data yang berhasil dijelaskan oleh model. Rumus:

![formula](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/6f0dc626109148ef907d73e9b200c4644ef77890/image/rmse_formula.png)


Gambar 6.1 Rumus RMSE

Keterangan:
- $$y_i$$: Nilai aktual
- $$\hat{y}_i$$: Nilai prediksi
- $$n$$: Jumlah data

![formula](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/ab189082713b5ab9395e128dac2c808692935472/image/r2_score_formula.png)


Gambar 6.2 Rumus $$R^2$$ Score

Keterangan:
- $$y_i$$: Nilai aktual
- $$\hat{y}_i$$: Nilai prediksi
- $$\bar{y}$$: Rata-rata dari nilai aktual
- $$n$$: Jumlah data.

Setelah mendapatkan model terbaik dari hasil hyperparameter tunning dari model Random Forest, model tersebut kemudian diberi nama 'best_regressor'. Setelah menjalankan model 'best_regressor' didapatkan evaluasi model seperti berikut:

| Metrics | Score |
| ------ | ------ |
| RMSE Train | 52.714566 |
| RMSE CV	| 61.470839 |
| R2 Score | 0.962422 |


Tabel 6.1 Hasil evaluasi model 'best_regressor'

Dari hasil evaluasi model 'best_regressor' yang merupakan hasil hyperparameter tunning dari Random Forest sebelumnya, diketahui bahwa $$R^2$$ Score mengalami penurunan menjadi 96.24% dibandingkan sebelumnya 97.14%. Namun, potensi terjadinya overfitting dalam diminimalisir karena selisih nilai RMSE Train dan RMSE CV semakin dekat yang mengindikasikan bahwa prediksi model dengan data train nilainya mirip dengan hasil prediksi model dengan data cross validation.

Selanjutnya, akan diuji hasil prediksi model 'best_regressor' dengan data uji untuk melihat apakah model dapat memprediksi harga tiket dengan tepat. Setelah dilakukan pengujian didapatkan bahwa:
- RMSE Test set = 61.27
- R2 Score Test set: 95.0%

Hasil pengujian model dengan data uji ini membuktikan bahwa model dapat memprediksi harga tiket dengan akurasi yang bagus seperti pengujian model dengan data latih. Untuk visualisasi bagaimana harga tiket hasil prediksi model dibandingkan dengan harga tiket yang sebenarnya dapat dilihat pada Gambar 6.2 berikut

![testing](https://raw.githubusercontent.com/FachryYusrie/Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Prediksi-Harga-Tiket-Pesawat/6f0dc626109148ef907d73e9b200c4644ef77890/image/Prediction_vs_Actual.png)


Gambar 6.2 Prediction vs Actual Harga Tiket Pesawat







## Referensi
[1]. [https://kumparan.com/jendela-dunia/kenapa-harga-tiket-pesawat-naik-turun-cari-tahu-jawabannya-di-sini-212Dwje2OR3](https://kumparan.com/jendela-dunia/kenapa-harga-tiket-pesawat-naik-turun-cari-tahu-jawabannya-di-sini-212Dwje2OR3).

[2]. [https://www.nucleusbox.com/assumptions-of-linear-regression/](https://www.nucleusbox.com/assumptions-of-linear-regression/)

[3]. [https://deepai.org/machine-learning-glossary-and-terms/random-forest](https://deepai.org/machine-learning-glossary-and-terms/random-forest).

[4]. [https://xgboost.readthedocs.io/en/stable/](https://xgboost.readthedocs.io/en/stable/)
