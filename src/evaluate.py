"""Fiziksel birimde metrik hesapla; gerçek ve tahmini grafikle karşılaştır."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Ekran gerektirmeyen ortamlarda grafik kaydı.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred):
    """MAE/RMSE °C cinsindendir; R² boyutsuzdur."""
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    return {'mae_c': float(mean_absolute_error(true, pred)),
            'rmse_c': float(np.sqrt(mean_squared_error(true, pred))),
            'r2': float(r2_score(true, pred))}


def evaluate_and_save(model, X_test, y_test, y_scaler, history, out_dir):
    """Model tahminini ters ölçekle, naif karşılaştırma ve grafikleri kaydet.

    Naif baseline, 'yarınki sıcaklık son gördüğüm sıcaklığa eşit' der.
    Burada hedef 1 saat sonrası olduğundan ilk geçmiş sıcaklık değil,
    son geçmiş sıcaklık özelliği kullanılır (indeksi eğitici verir).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    real = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred = y_scaler.inverse_transform(predictions_scaled).ravel()
    results = {'lstm': metrics(real, pred)}
    (out_dir.parent / 'metrics.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(12, 4))
    n = min(250, len(real))
    ax.plot(real[:n], label='Gerçek sıcaklık (°C)', linewidth=1.5)
    ax.plot(pred[:n], label='LSTM tahmini (°C)', linewidth=1.2)
    ax.set(xlabel='Test örneği', ylabel='Sıcaklık (°C)', title='Test: gerçek ve tahmin')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'predictions.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'], label='Eğitim MSE')
    ax.plot(history.history['val_loss'], label='Doğrulama MSE')
    ax.set(xlabel='Epoch', ylabel='Normalize MSE', title='Eğitim geçmişi')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'training_history.png', dpi=150)
    plt.close(fig)
    return results, real, pred
