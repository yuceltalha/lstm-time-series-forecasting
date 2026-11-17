# LSTM ile Sıcaklık Regresyonu — Jena Climate

**Yüksek lisans vize projesi | Regresyon | Python 3.11 | TensorFlow / Keras**

## Bu proje neden regresyon ödevidir?

**Regresyon**, girdilerden **sürekli sayısal bir hedefi** tahmin etme problemidir. Buradaki hedef, Jena Climate ölçümlerinden **bir saat sonraki hava sıcaklığını °C cinsinden** tahmin etmektir. Örneğin modelin çıktısı `12.7 °C` olabilir; bir sınıf etiketi veya kategori değildir.

**LSTM (Long Short-Term Memory)** tek başına “regresyon yöntemi” adından ziyade bir tekrarlayan sinir ağı mimarisidir. Bu projede **LSTM tabanlı sinir ağı regresyonu** uygulanır: son katman `Dense(1)` ve doğrusal (aktivasyonsuz) çıkıştır, MSE ile eğitilir. **Yalnız bir regresyon yaklaşımı** kullanılır; Linear Regression, Random Forest, SVR gibi alternatifler eğitilmez. Proje bir regresyon ödevi olarak sunulurken bu ayrım açıkça belirtilmelidir.

### Problem tanımı

- **Bağımsız değişkenler (X):** Son 24 saatin basınç, sıcaklık, bağıl nem, rüzgâr hızı ve rüzgâr yönü ölçümleri; her örnek `(24, 5)` boyutunda.
- **Bağımlı değişken (y):** Bir saat sonraki sıcaklık, sürekli sayısal değer (°C).
- **Öğrenilen fonksiyon:** `f_θ: R^(24×5) → R`. Model, geçmiş meteorolojik gözlemlerden sıcaklığa eşleme öğrenir.
- **Regresyon kaybı:** MSE (ortalama karesel hata), eğitimde standartlaştırılmış hedef üzerinde hesaplanır.
- **Regresyon değerlendirmesi:** MAE (°C), RMSE (°C), R²; gerçek birim üzerinden raporlanır.
- **Tek yöntem:** `Input(24,5) → LSTM(32) → Dropout(0.2) → Dense(1)`.

## Proje dosyaları

- `src/`: indirme, ön işleme, LSTM regresyon eğitimi, değerlendirme ve görselleştirme; Türkçe açıklamalar
- `config.yaml`: veri ve eğitim ayarları
- `notebooks/academic_presentation.ipynb`: hocaya yönelik **regresyon odaklı** Jupyter sunumu
- `learning/lstm_interactive_guide.html`: kişisel öğrenme için etkileşimli LSTM rehberi; regresyon bağlantısı eklendi
- `learning/regression_foundations.md`: regresyonun matematiksel temeli ve olası sözlü sorular
- `tests/`: veri sızıntısı ve kaydırmalı pencere testleri

## Hızlı başlangıç (Windows 11 / PowerShell)

```powershell
git clone https://github.com/yuceltalha/lstm-time-series-forecasting.git
cd lstm-time-series-forecasting
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train --config config.yaml
python -m jupyter lab notebooks/academic_presentation.ipynb
```

PowerShell aktivasyonu engellenirse `cmd` ile `.venv\Scripts\activate.bat` kullanılabilir.

## Veri ve yöntem

1. [Jena Climate veri seti](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) çalıştırıldığında otomatik indirilir.
2. 10 dakikalık örneklerden her 6. satır seçilerek saatlik örnekler elde edilir; bu işlem saatlik ortalama alma değildir.
3. Eksik değer işaretleri temizlenir; zaman sırasına göre ileri doldurma yapılır.
4. Eğitim/doğrulama/test bölümleri kronolojik **%70/%20/%10** oluşturulur; StandardScaler **yalnız eğitim verisi** ile öğrenilir.
5. 24 saatlik giriş pencerelerinden bir sonraki saatin sıcaklığı hedeflenir.
6. LSTM, Adam optimizer ve MSE ile eğitilir; EarlyStopping / ModelCheckpoint kullanılır.
7. Test sonunda gerçek °C değerleri üzerinden MAE, RMSE, R² hesaplanır ve grafikler kaydedilir.

## Formüller

```text
MSE  = (1/n) Σ(y_i - ŷ_i)²
MAE  = (1/n) Σ|y_i - ŷ_i|
RMSE = √[(1/n) Σ(y_i - ŷ_i)²]
R²   = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²
```

MAE ve RMSE °C, R² boyutsuzdur. R² negatif olabilir. Çıkış aktivasyonu kullanılmaz: sıcaklık negatif olabilir.

## Eğitim çıktıları

- `outputs/models/best_lstm.keras`: en iyi doğrulama kaybındaki model
- `outputs/metrics.json`: **yalnız `lstm`** anahtarında gerçek test sonuçları
- `outputs/figures/predictions.png`: gerçek sıcaklık / regresyon tahmini
- `outputs/figures/training_history.png`: eğitim / doğrulama MSE

Çıktılar git içinde takip edilmez; modeli çalıştırmadan **başarı skoru iddia edilmez**. Kendi bilgisayarınızda eğitip notebook'u açın.

## Sunumda kullanılabilecek doğru tanım

> “Bu çalışmada sürekli bir meteorolojik değişken olan hava sıcaklığını tahmin etmek için LSTM tabanlı derin öğrenme regresyonu kullandım. Tek bir regresyon yaklaşımı uyguladım. Modelin son katmanı tek bir sayısal çıktı üretir ve MSE kaybıyla eğitilir; test başarısını MAE, RMSE ve R² ile değerlendirdim.”

**Akademik nüans:** Eğer öğretim üyesi “özellikle klasik regresyon algoritması (ör. doğrusal regresyon) kullanın” dediyse LSTM mimarisi bu daha dar gerekliliği karşılamayabilir. Ancak ödev “bir regresyon problemi çözün” ise bu proje uygundur.

## Sınırlamalar ve kaynaklar

Tek meteoroloji istasyonu; tek adımlı tahmin; rüzgâr yönü ham açı olarak kullanılıyor; tek zaman bölmesi; genel geçer başarı garantisi yok.

- [TensorFlow Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Hochreiter & Schmidhuber (1997), LSTM](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Veri seti](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)
