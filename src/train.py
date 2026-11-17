"""Çalıştır: python -m src.train --config config.yaml

Bu dosya proje orkestratörüdür: indirme -> ön işleme -> pencereleme ->
eğitim -> değerlendirme -> çıktı. Ayrıntılı konu anlatımı HTML rehberindedir.
"""
import argparse
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
import yaml

from src.download_data import ensure_dataset
from src.preprocessing import load_frame, split_and_scale, make_windows
from src.models import build_lstm
from src.evaluate import evaluate_and_save, metrics


def run(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    d, m, tr = cfg['data'], cfg['model'], cfg['training']

    # RNG sabitleme deneyin tekrarlanabilirliğini artırır. Farklı donanımlarda
    # deterministik sonuç %100 garanti değildir.
    seed = int(cfg['seed'])
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    csv_path = ensure_dataset(d['url'], d['csv_name'], root / 'data/raw')
    df = load_frame(csv_path, d['features'], d['target'], d['sample_every'])
    x, y, te, ve, xs, ys = split_and_scale(
        df, d['features'], d['target'], d['train_fraction'], d['val_fraction'])

    # Gözlem sayısı yeterli mi? Bölümler kronolojik, shuffle yapılmıyor.
    lookback, horizon = int(d['lookback']), int(d['horizon'])
    X_train, y_train, _ = make_windows(x, y, 0, te, lookback, horizon)
    X_val, y_val, _ = make_windows(x, y, te, ve, lookback, horizon)
    X_test, y_test, _ = make_windows(x, y, ve, len(df), lookback, horizon)
    print(f'Veri: {len(df):,} satır | Train {len(X_train):,} | '
          f'Val {len(X_val):,} | Test {len(X_test):,}')
    print(f'Giriş boyutu: {X_train.shape} (örnek, saat, özellik)')

    model = build_lstm(lookback, len(d['features']), int(m['units']),
                       float(m['dropout']), float(tr['learning_rate']))
    model.summary()
    model_dir = root / 'outputs/models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'best_lstm.keras'
    callbacks = [
        # Doğrulama iyileşmezse en iyi epoch'a geri dön.
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=int(tr['patience']), restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor='val_loss', save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        batch_size=int(tr['batch_size']), epochs=int(tr['epochs']),
        callbacks=callbacks, shuffle=False, verbose=1)
    results, real, pred = evaluate_and_save(
        model, X_test, y_test, ys, history, root / 'outputs/figures')

    # Naif kalıcılık baseline: geçmişteki EN SON sıcaklığı doğrudan kullan.
    # Bu karşılaştırma, modelin basit sezgiyi gerçekten geçip geçmediğini gösterir.
    t_feature_idx = d['features'].index(d['target'])
    last_temp_scaled = X_test[:, -1, t_feature_idx].reshape(-1, 1)
    naive = xs.inverse_transform(
        np.column_stack([last_temp_scaled.ravel() if i == t_feature_idx
                         else np.zeros(len(last_temp_scaled))
                         for i in range(len(d['features']))]))[:, t_feature_idx]
    results['persistence_baseline'] = metrics(real, naive)
    import json
    (root / 'outputs/metrics.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TEST SONUÇLARI (°C):', json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jena Climate tek katmanlı LSTM')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    run(args.config)
