import pandas as pd
import os

path = r"C:\Users\emirc\OneDrive\Masaüstü\Projeler\BüyükProje\BüyükProje\Auto\data"

def read_csv(file_name: str = ""):
    # Take only the first word ending with .csv and strip spaces
    file_name = file_name.strip().split()[0]
    if not file_name.endswith('.csv'):
        raise ValueError(".csv uzantısıyla birlikte dosya adı giriniz.  # Please enter the file name with the .csv extension.")
    
    file_path = os.path.join(path, file_name)
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
    separators = [',', ';', '\t', '|']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=encoding)
                # If there is only 1 column and it contains a semicolon, try again with semicolon as separator
                if len(df.columns) == 1 and ';' in str(df.iloc[0, 0]):
                    df = pd.read_csv(file_path, sep=';', encoding=encoding)
                    return df
                elif len(df.columns) > 1:
                    return df
            except UnicodeDecodeError:
                continue  # Could not read with this encoding, try the next one
            except Exception:
                continue  # Could not read with this separator, try the next one
    
    # Final attempt if none work
    try:
        df = pd.read_csv(file_path, encoding='utf-8',errors='ignore')
        return df
    except Exception as e:
        raise ValueError(f"# File could not be read with any encoding or separator. Error: {e}")
