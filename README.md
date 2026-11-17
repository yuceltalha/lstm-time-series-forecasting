# LSTM ile Jena Climate Sıcaklık Tahmini

**Yüksek lisans vize projesi** · Python 3.11 · TensorFlow/Keras · Türkçe açıklamalı

Bu proje, Jena Climate meteorolojik zaman serisindeki **son 24 saatlik 5 ölçümü** kullanarak **bir saat sonraki sıcaklığı (°C)** tahmin eder. Tek katmanlı LSTM, bir naif kalıcılık (persistence) baseline ile karşılaştırılır. Gerçek ölçütler yalnızca model eğitildikten sonra üretilir; depoda uydurma skor bulunmaz.

## Proje dosyaları

- `src/`: çalışan, Türkçe yorumlu indir/ön işle/eğit/değerlendir kodu
- `config.yaml`: ayarlanabilir hiperparametre ve veri ayarları
- `notebooks/academic_presentation.ipynb`: **hocaya sunum** için ayrı notebook
- `learning/lstm_interactive_guide.html`: **kişisel öğrenme** için çevrimdışı HTML rehberi
- `tests/`: veri sızıntısı ve pencereleme gibi hatalara karşı testler

## Hızlı başlangıç (Windows 11, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train --config config.yaml
python -m jupyter lab notebooks/academic_presentation.ipynb
```

> PowerShell `Activate.ps1` yürütme politikası engellerse `cmd` üzerinde `.venv\Scripts\activate.bat` çalıştırabilir veya tüm komutları `.venv\Scripts\python.exe` ile verebilirsin.
>
> İlk eğitimde internet gerekir. CSV otomatik `data/raw` altına indirilir. Notebook daha önce eğitim çalıştırılmamışsa kendi başına modeli eğitmez; hazır çıktıları okuyarak sunum yapar. Önce `python -m src.train` çalıştır.

## Deney tasarımı

1. Orijinal 10 dakikalık veriden her 6. örnek alınır (~1 saat). Tarih sırası korunur.
2. `-9999` eksik rüzgâr işaretleri NaN'a çevrilir; yalnız geçmişe dayalı ileri doldurma yapılır.
3. Zaman sıralı **%70 eğitim / %20 doğrulama / %10 test** ayrımı uygulanır.
4. StandardScaler yalnızca eğitim dönemine `fit` edilir; gelecek istatistikleri eğitime sızmaz.
5. 24 gözlemli pencere: `(24 saat, 5 özellik)` → 1 saat sonrası sıcaklık.
6. `LSTM(32)` → `Dropout(0.2)` → `Dense(1)`; Adam + MSE ile eğitim.
7. EarlyStopping / en iyi model checkpoint; testte MAE, RMSE, R² ve gerçek-tahmin grafiği.
8. Son bilinen sıcaklığın aynısını tahmin eden **persistence baseline** ile karşılaştırma.

## Çıktılar

- `outputs/models/best_lstm.keras`: doğrulama kaybına göre en iyi model
- `outputs/metrics.json`: model ve baseline test metrikleri
- `outputs/figures/predictions.png`: test tahminleri
- `outputs/figures/training_history.png`: eğitim/doğrulama kaybı

Bu dosyalar varsayılan olarak Git'e eklenmez. Gerçek sonuçlar deney çalıştırıldıktan sonra oluşur.

## Yorumlama ve sınırlamalar

- Saatlik yeniden örnekleme, özgün ölçümlerin yalnız her 6. satırını alır; ortalama değildir.
- Rüzgâr yönü açısal bir değişkendir ve burada basit derece değeriyle kullanılır; sin/cos kodlaması ileri çalışmaya bırakılmıştır.
- Veriler tek istasyondandır, farklı coğrafyalara genellenebilirlik test edilmemiştir.
- Sabit bir split ile sonuçlar zaman kaymasına duyarlı olabilir; farklı dönemlerle tekrar test ileride yapılabilir.
- GPU isteğe bağlıdır; CPU ile eğitim mümkündür, süre makineye göre değişir.

## Veri kaynağı

Jena Climate (Max Planck Institute for Biogeochemistry), TensorFlow'un eğitim materyallerinde dağıtılan veri seti. İndirilen özgün veriler repository içinde tutulmaz.

- [Veri seti ZIP bağlantısı](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip)
- [TensorFlow zaman serisi öğreticisi](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [LSTM makalesi: Hochreiter & Schmidhuber (1997)](https://doi.org/10.1162/neco.1997.9.8.1735)

## Akademik dürüstlük

Notebook, eğitimden elde edilen sonuçları otomatik görüntüler. Eğitim henüz yapılmadıysa **sonuç üretildiği iddia edilmez**. Yapılan ön işleme ve deney kararlarını sunumda açıklayın.

## Public GitHub deposuna yükleme

GitHub üzerinde **yuceltalha/lstm-time-series-forecasting** adlı **Public**, başlangıç README/.gitignore/lisans eklenmemiş **boş** depo oluşturun. Ardından ZIP içeriğindeki proje klasöründe PowerShell açıp:

```powershell
git init -b main
git add .
git commit -m "Initial LSTM Jena Climate coursework"
git remote add origin https://github.com/yuceltalha/lstm-time-series-forecasting.git
git push -u origin main
```

GitHub kullanıcı doğrulaması istenebilir. Bu adımlar **depo GitHub üzerinde oluşturulduktan sonra** çalışır.
