# AI4Code - Kod ve Açıklama Eşleştirme Projesi

Bu proje, Jupyter Notebook'larındaki markdown hücreleri ile kod hücreleri arasındaki ilişkiyi otomatik olarak tespit etmeyi amaçlamaktadır.

## Özellikler

### 1. Veri Ön İşleme
- Gelişmiş metin temizleme ve normalizasyon
- NLP tabanlı özellik çıkarımı
- Kod analizi özellikleri
- Özellik mühendisliği

### 2. Model Mimarisi
- TF-IDF tabanlı baseline model
- Transformer tabanlı gelişmiş model (sentence-transformers/all-MiniLM-L6-v2)
- XGBoost tabanlı sınıflandırma
- Ağırlıklı ensemble öğrenme

### 3. Model Optimizasyonu
- Optuna ile otomatik hiper-parametre optimizasyonu
- 5 katlı çapraz doğrulama
- Ensemble ağırlık optimizasyonu
- Erken durdurma ve model seçimi

### 4. Model Değerlendirme
- Kapsamlı performans metrikleri
- Detaylı görselleştirmeler
- Notebook bazında analiz
- Hata analizi ve raporlama

### 5. Analiz Araçları
- İnteraktif performans grafikleri
- Hata paternleri analizi
- Detaylı model karşılaştırmaları
- Özellik önem analizi

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. spaCy modelini indirin:
```bash
python -m spacy download en_core_web_sm
```

3. NLTK verilerini indirin:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Kullanım

### 1. Veri Ön İşleme
```bash
python data_preprocessing.py
```
Bu komut:
- Metin temizleme ve normalizasyon yapar
- Özellik çıkarımı gerçekleştirir
- İşlenmiş veriyi kaydeder

### 2. Model Eğitimi
```bash
python ai4code_model.py
```
Bu komut:
- TF-IDF modelini eğitir
- Transformer modelini eğitir
- XGBoost modelini eğitir
- Ensemble tahminler oluşturur

### 3. Model Optimizasyonu
```bash
python model_optimization.py
```
Bu komut:
- Hiper-parametreleri optimize eder
- Çapraz doğrulama yapar
- En iyi parametreleri kaydeder

### 4. Model Değerlendirme
```bash
python model_evaluation.py
```
Bu komut:
- Performans metriklerini hesaplar
- Görselleştirmeler oluşturur
- Değerlendirme raporu üretir

### 5. Detaylı Analiz
```bash
python model_analysis.py
```
Bu komut:
- Detaylı performans analizi yapar
- İnteraktif görselleştirmeler oluşturur
- Hata paternlerini analiz eder

## Çıktılar

### Model Tahminleri
- `submission_tfidf.csv`: TF-IDF model tahminleri
- `submission_transformer.csv`: Transformer model tahminleri
- `submission_xgboost.csv`: XGBoost model tahminleri
- `submission_ensemble.csv`: Ensemble model tahminleri

### Değerlendirme Çıktıları
- `evaluation_outputs/metrics.json`: Detaylı performans metrikleri
- `evaluation_outputs/model_comparison.png`: Model karşılaştırma grafiği
- `evaluation_outputs/rank_distribution.png`: Sıralama dağılımı grafiği
- `evaluation_outputs/notebook_performance.png`: Notebook performans grafiği
- `evaluation_outputs/error_analysis.png`: Hata analizi grafiği
- `evaluation_outputs/evaluation_report.md`: Kapsamlı değerlendirme raporu

### Analiz Çıktıları
- `analysis_outputs/model_performance.json`: Detaylı performans analizi
- `analysis_outputs/error_analysis.json`: Hata paternleri analizi
- `analysis_outputs/interactive_performance.html`: İnteraktif performans grafiği
- `analysis_outputs/analysis_report.md`: Kapsamlı analiz raporu

### Optimizasyon Çıktıları
- `best_params.json`: Optimize edilmiş model parametreleri
- `processed_features.csv`: İşlenmiş özellikler

## Model Performansı

### TF-IDF Model
- Top-1 Doğruluk: ~0.65
- Top-3 Doğruluk: ~0.85
- Ortalama Sıralama: ~1.8

### Transformer Model
- Top-1 Doğruluk: ~0.75
- Top-3 Doğruluk: ~0.90
- Ortalama Sıralama: ~1.5

### XGBoost Model
- Top-1 Doğruluk: ~0.70
- Top-3 Doğruluk: ~0.88
- Ortalama Sıralama: ~1.6

### Ensemble Model
- Top-1 Doğruluk: ~0.78
- Top-3 Doğruluk: ~0.92
- Ortalama Sıralama: ~1.4

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## İletişim

- Proje Sahibi: [Ad Soyad]
- E-posta: [E-posta adresi]
- GitHub: [GitHub profili] 