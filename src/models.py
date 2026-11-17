"""Tek katmanlı LSTM ile bir adım sonrası sıcaklık regresyonu."""
import tensorflow as tf


def build_lstm(lookback: int, num_features: int, units: int, dropout: float,
               learning_rate: float) -> tf.keras.Model:
    """(batch, time, feature) girişinden 1 sürekli hedef üretir.

    LSTM hücresi üç kapı (unutma, giriş, çıkış) ve hücre belleği kullanır.
    return_sequences=False yalnız son zaman adımının özetini Dense'e geçirir.
    Dropout eğitimde bazı aktivasyonları kapatarak ezberlemeyi azaltır.
    Son katmanda aktivasyon yok: sıcaklık negatif de olabileceği için
    çıkışı ReLU/sigmoid ile kısıtlamak istemiyoruz.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, num_features)),
        tf.keras.layers.LSTM(units, return_sequences=False, name='lstm'),
        tf.keras.layers.Dropout(dropout, name='dropout'),
        tf.keras.layers.Dense(1, name='temperature'),
    ], name='jena_temperature_lstm')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',  # Kayıp standartlaştırılmış sıcaklık ölçeğinde hesaplanır.
        metrics=['mae'],
    )
    return model
