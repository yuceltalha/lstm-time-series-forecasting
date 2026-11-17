"""Resmi TensorFlow örnek veri kümesini ZIP biçiminde indirip güvenli açar."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


def ensure_dataset(url: str, csv_name: str, raw_dir: Path) -> Path:
    """CSV varsa tekrar indirmez; indirme hata verirse açıkça bildirir."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / csv_name
    if csv_path.is_file():
        return csv_path

    zip_path = raw_dir / 'jena_climate.zip'
    print(f'Veri indiriliyor: {url}')
    try:
        urlretrieve(url, zip_path)
        with ZipFile(zip_path) as archive:
            # ZIP dosyasının tamamını açmıyoruz: yalnızca beklenen CSV'yi
            # seçiyoruz. Bu yaklaşım path traversal risklerini de azaltır.
            matches = [name for name in archive.namelist() if Path(name).name == csv_name]
            if len(matches) != 1:
                raise ValueError(f'ZIP içinde beklenen CSV bulunamadı: {csv_name}')
            with archive.open(matches[0]) as source, csv_path.open('wb') as target:
                import shutil
                shutil.copyfileobj(source, target)
    except Exception:
        csv_path.unlink(missing_ok=True)
        raise
    finally:
        zip_path.unlink(missing_ok=True)
    return csv_path
