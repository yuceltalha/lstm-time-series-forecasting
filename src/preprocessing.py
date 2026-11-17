"""Zaman serisi bölme, ölçekleme ve hedefe uygun pencere üretimi.

En önemli akademik ilke: test/validation verisinden eğitim aşamasında
istatistik öğrenilmemelidir (data leakage).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_frame(csv_path, features, target, sample_every=6):
    """CSV'yi oku, kronolojik sırada tut, saatte bir örneğe indir.

    Jena veri setinde rüzgar hızı için -9999.0 eksik değer işareti olarak
    kullanılır. Bu değerler NaN'a dönüştürülür ve geçmiş/gelecek sızmasını
    engellemek için yalnız ileri doldurulur; başlangıçtaki NaN'lar çıkarılır.
    """
    df = pd.read_csv(csv_path)
    required = list(dict.fromkeys([*features, target]))
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Veri setinde eksik sütunlar: {sorted(missing)}')
    if 'Date Time' not in df.columns:
        raise ValueError('Date Time sütunu bulunamadı')
    # Jena formatı: 01.01.2009 00:10:00; dayfirst=True zorunlu.
    df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
    df = df.sort_values('Date Time').drop_duplicates('Date Time')
    if sample_every < 1:
        raise ValueError('sample_every pozitif tam sayı olmalıdır')
    df = df.iloc[::sample_every].copy()
    df[required] = df[required].replace(-9999.0, np.nan)
    df[required] = df[required].ffill()
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 100:
        raise ValueError('Çok az gözlem bulundu')
    return df


def split_and_scale(df, features, target, train_fraction, val_fraction):
    """Ölçekleyicileri SADECE eğitim dönemine fit eder.

    Hem X hem y için ayrı StandardScaler kullanıyoruz; bu sayede modelin
    normalize sıcaklık tahmini tekrar °C birimine dönüştürülebilir.
    Train/val/test kesimi zaman eksenindedir, karıştırma YOKTUR.
    """
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
            train_fraction + val_fraction < 1):
        raise ValueError('Bölme oranları geçersiz')
    n = len(df)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    x_scaler = StandardScaler().fit(df.iloc[:train_end][features])
    y_scaler = StandardScaler().fit(df.iloc[:train_end][[target]])
    x = x_scaler.transform(df[features]).astype('float32')
    y = y_scaler.transform(df[[target]]).ravel().astype('float32')
    return x, y, train_end, val_end, x_scaler, y_scaler


def make_windows(x, y, start_target, end_target, lookback=24, horizon=1):
    """Hedef indeksleri [start_target, end_target) aralığında olan pencereler.

    Bir hedef t için giriş [t-horizon-lookback+1, ..., t-horizon] indekslerini
    kapsar. horizon=1 ile girdide en son t-1, hedefte t bulunur.
    Bu yöntem validasyon/test başlangıcında geçmiş dönem gözlemlerini giriş
    olarak kullanabilir; fakat hedefleri asla başka split'e taşımaz.
    """
    if lookback < 1 or horizon < 1:
        raise ValueError('lookback ve horizon pozitif olmalı')
    windows, targets, indices = [], [], []
    for t in range(start_target, end_target):
        end = t - horizon + 1
        begin = end - lookback
        if begin < 0:
            continue
        windows.append(x[begin:end])
        targets.append(y[t])
        indices.append(t)
    if not windows:
        raise ValueError('Pencere oluşturulamadı: veri veya lookback yetersiz')
    return (np.asarray(windows, dtype=np.float32),
            np.asarray(targets, dtype=np.float32), np.asarray(indices))
