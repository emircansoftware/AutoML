import pandas as pd
import os

path = r"C:\Users\emirc\OneDrive\Masaüstü\BüyükProje\Auto\data"

def read_csv(file_name: str = ""):
    # Sadece .csv ile biten ilk kelimeyi al ve boşlukları temizle
    file_name = file_name.strip().split()[0]
    if not file_name.endswith('.csv'):
        raise ValueError(".csv uzantısıyla birlikte dosya adı giriniz.")
    
    file_path = os.path.join(path, file_name)
    
    # Farklı encoding'leri dene
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    separators = [',', ';', '\t', '|']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                # Eğer sadece 1 sütun varsa ve içinde ; varsa, noktalı virgül ile tekrar dene
                if len(df.columns) == 1 and ';' in str(df.iloc[0, 0]):
                    df = pd.read_csv(file_path, sep=';', encoding=encoding)
                    return df
                elif len(df.columns) > 1:
                    return df
            except UnicodeDecodeError:
                continue  # Bu encoding ile okunamadı, diğerini dene
            except Exception:
                continue  # Bu separator ile okunamadı, diğerini dene
    
    # Hiçbiri çalışmazsa son bir deneme
    try:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        return df
    except Exception as e:
        raise ValueError(f"Dosya hiçbir encoding veya separator ile okunamadı. Hata: {e}")
