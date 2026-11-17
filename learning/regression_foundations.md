# LSTM Tabanlı Regresyon — Kişisel Öğrenme Notları

## 1. Problem türü: regresyon

Denetimli öğrenmede sınıflandırmanın hedefi kategoriktir; regresyonda sürekli sayısaldır. Bu projede:

- X: geçmiş 24 saat × 5 değişken
- y: bir saat sonraki sıcaklık, °C
- ŷ: LSTM ağının tek sayısal çıktısı

Örneğin gerçek 11.5 °C ve tahmin 12.8 °C ise mutlak hata 1.3 °C, karesel hata 1.69 °C²'dir.

## 2. LSTM ile regresyonun farkı

LSTM bir *ağ katmanı/mimarisidir*. Regresyon, modele verilen görevdir. Ardışık girişler LSTM'de işlenir, son gizli durum `Dense(1)` katmanı üzerinden sınırsız bir reel sayıya dönüşür. Bu nedenle “LSTM ile regresyon”, doğru bir ifadedir; “LSTM = doğrusal regresyon”, yanlıştır.

## 3. Neden Dense(1), neden doğrusal çıkış?

Bir saat sonra tek sıcaklık değeri tahmin ediliyor; çıkış boyutu 1. ReLU negatif sıcaklıkları kapatır; sigmoid 0–1'e sıkıştırır; softmax sınıflandırma olasılığı üretir. Aktivasyonsuz Dense, hedef ölçeğinde serbest tahmine izin verir.

## 4. Kayıp ve metrikler

- MSE = (1/n) Σ (yᵢ−ŷᵢ)². Eğitim sırasında, standartlaştırılmış y için optimize edilir.
- MAE = (1/n) Σ |yᵢ−ŷᵢ|. Testte °C cinsindendir.
- RMSE = √MSE. Testte °C cinsindendir ve büyük hataları daha sert yansıtır.
- R² = 1 − Σ(yᵢ−ŷᵢ)² / Σ(yᵢ−ȳ)². Negatif olabilir; tek başına yeterli değildir.

**Önemli:** Eğitim MSE'si standartlaştırılmış sıcaklıkta, raporlanan MAE/RMSE ters ölçeklenmiş °C üzerinden hesaplanır. Sayıların ölçekleri karıştırılmamalı.

## 5. Regresyon açısından veri hazırlığı

Zaman serisi, normal tablo regresyonundan farklı olarak sıranın korunduğu bir probleme dönüşür. Örnek X[n] matrisi 24×5, y[n] tek sıcaklıktır. Zamansal %70/%20/%10 bölme sonrası scaler yalnız eğitim üzerinde fit edilir. Test sonucu doğrulama için kullanılmaz; aşırı iyimser sonuç önlenir.

## 6. Savunma soruları

**“LSTM bir regresyon algoritması mı?”** LSTM, RNN ailesinden bir sinir ağı mimarisidir. Bu mimariyi tek sayısal çıkış ve regresyon kaybıyla regresyon modeli haline getirdim.

**“Neden linear regression kullanmadın?”** Ödevde bir regresyon yaklaşımı yeterli olduğu için zaman bağımlılıklarını ele alan LSTM tabanlı regresyonu seçtim. Klasik doğrusal regresyon uygulanmadı. Hocanın özellikle *klasik* bir yöntem istemesi halinde yaklaşım yeniden değerlendirilmelidir.

**“Model ne tahmin ediyor?”** Bir sonraki saatin sıcaklığını °C olarak.

**“Sınıflandırma neden değil?”** Çünkü etiketler (“soğuk/sıcak”) değil, sürekli gerçek sıcaklık değerleri hedefleniyor.

**“Hata neden 0 olmayabilir?”** Ölçüm gürültüsü, bilgi eksikliği, model kapasitesi ve gelecekteki değişkenler belirsizlik yaratır.

## 7. Akademik anlatım sırası

Problem tanımı → özellik/etiket ayrımı → veri seti → zamansal split ve scaling → pencereleme → LSTM+Dense mimarisi → MSE ile eğitim → MAE/RMSE/R² → gerçek test sonuçları ve sınırlamalar.

**Ölçüm sonucu üretmeden metrik iddia etme.** Önce `python -m src.train` çalıştır ve notebook'taki gerçek tabloyu kullan.
