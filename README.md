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
![image](https://github.com/user-attachments/assets/fa226b7a-4989-47ab-a5da-88ac9356dfd7)

2. Claude Ayarlarını Yapılandırın
Claude Desktop'u açın:
File -> Settings -> Developer -> Edit Config
![image](https://github.com/user-attachments/assets/63397750-eace-48e2-aff0-63d041585afd)

Açılan claude_desktop_config dosyasını aşağıdaki gibi düzenleyin:
{
  "mcpServers": {
    "Auto": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\Users\\emirc\\OneDrive\\Masaüstü\\BüyükProje\\Auto", 
        "run",
        "main.py"
      ]
    }
  }
}
Önemli -> "C:\\Users\\emirc\\OneDrive\\Masaüstü\\BüyükProje\\Auto", -------> BU KISMI REPODA BULUNAN DOSYALARIN OLDUĞU YOL İLE DEĞİŞTİRMELİSİNİZ.

![image](https://github.com/user-attachments/assets/54ae316e-c811-4be2-acdd-a32c411cc6ad)

3. CSV Dosyasını Ekleyin:
Repodaki data klasörünün içine analiz etmek istediğiniz .csv dosyasını ekleyin.

![image](https://github.com/user-attachments/assets/16c27313-391f-4419-a278-ff6dafc9d22e)

4. read_csv_file.py Yolunu Güncelleyin:
utils/read_csv_file.py dosyasındaki path değişkenini, kendi sisteminizdeki data klasörünün tam yolu ile değiştirin.

![image](https://github.com/user-attachments/assets/f0f6c80d-1466-4d79-af21-6b99fcfb815c)




