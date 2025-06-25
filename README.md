MCP Server
Bu MCP Server sayesinde istediğiniz .csv uzantılı veriyi kullanarak veri bilimi aşamalarını otomatikleştirebilirsiniz.
Bu sistem, veriyi okuma, düzenleme, modellemeye hazırlama ve modelleri test etme adımlarını kolaylıkla gerçekleştirir.

Özellikler:
-CSV veri okuma
-Veri ön işleme ve temizleme
-Model eğitimi ve test işlemleri
-Kolay yapılandırma ile hızlı entegrasyon

Nasıl Kullanılır?
1. Claude Desktop Uygulamasını İndirin
Öncelikle Claude Desktop uygulamasını sisteminize indirin ve yükleyin.

2. Claude Ayarlarını Yapılandırın
Claude Desktop'u açın:
File -> Settings -> Developer -> Edit Config

Açılan claude_desktop_config dosyasını aşağıdaki gibi düzenleyin:
{
  "mcpServers": {
    "Auto": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\emirc\\OneDrive\\Masaüstü\\BüyükProje\\Auto", -------> BU KISMI REPODA BULUNAN DOSYALARIN OLDUĞU YOL İLE DEĞİŞTİRMELİSİNİZ.
        "run",
        "main.py"
      ]
    }
  }
}

3. CSV Dosyasını Ekleyin
Repodaki data klasörünün içine analiz etmek istediğiniz .csv dosyasını ekleyin.

4. read_csv_file.py Yolunu Güncelleyin
utils/read_csv_file.py dosyasındaki path değişkenini, kendi sisteminizdeki data klasörünün tam yolu ile değiştirin.





