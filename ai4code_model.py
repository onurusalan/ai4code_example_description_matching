import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import warnings
import xgboost as xgb
from collections import Counter
import re
import json
import os
from sklearn.model_selection import train_test_split
from data_preprocessing import DataPreprocessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
warnings.filterwarnings('ignore')

class AI4CodeProcessor:
    def __init__(self):
        self.train_data = None
        self.notebooks = {}
        self.transformer_model = None
        self.vectorizer = None
        self.xgb_model = None
        self.preprocessor = DataPreprocessor()
        self.validation_data = None
        self.results_log = []
        
    def log_result(self, message):
        """Sonuçları logla"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results_log.append(f"[{timestamp}] {message}")
        
    def load_data(self):
        print("Veri yükleme başlıyor...")
        
        # JSON dosyalarını yükle
        train_path = "train"
        self.notebooks = {}
        
        for filename in tqdm(os.listdir(train_path)):
            if filename.endswith('.json'):
                with open(os.path.join(train_path, filename), 'r') as f:
                    notebook_data = json.load(f)
                    notebook_id = filename.split('.')[0]
                    
                    # Markdown ve kod hücrelerini ayır
                    markdown_cells = []
                    code_cells = []
                    
                    # cell_type ve source bilgilerini al
                    cell_types = notebook_data['cell_type']
                    sources = notebook_data['source']
                    
                    for cell_id, cell_type in cell_types.items():
                        source = sources.get(cell_id, '')
                        if cell_type == 'markdown':
                            markdown_cells.append({
                                'source': source,
                                'cell_id': cell_id
                            })
                        elif cell_type == 'code':
                            code_cells.append({
                                'source': source,
                                'cell_id': cell_id
                            })
            
            self.notebooks[notebook_id] = {
                        'markdown': pd.DataFrame(markdown_cells),
                        'code': pd.DataFrame(code_cells)
                    }
        
        print(f"Toplam {len(self.notebooks)} notebook yüklendi.")
    
    def preprocess_text(self, text):
        # Temel metin temizleme
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def extract_features(self, markdown_text, code_text):
        # TF-IDF özellikleri
        tfidf_features = self.vectorizer.transform([self.preprocess_text(markdown_text + " " + code_text)]).toarray()
        
        # Transformer özellikleri
        transformer_features = self.get_embeddings([markdown_text + " " + code_text])[0]
        
        # Özellikleri birleştir
        return np.concatenate([tfidf_features.flatten(), transformer_features])

    def create_tfidf_model(self):
        print("TF-IDF model oluşturuluyor...")
        
        # Batch boyutu
        batch_size = 1000
        all_texts = []
        
        # Notebookları batch'ler halinde işle
        notebook_items = list(self.notebooks.items())
        for i in tqdm(range(0, len(notebook_items), batch_size), desc="TF-IDF İşleniyor"):
            batch = notebook_items[i:i + batch_size]
            
            for notebook_id, notebook in batch:
                # Sadece gerekli metinleri al ve önişle
                markdown_texts = notebook['markdown']['source'].tolist()
                code_texts = notebook['code']['source'].tolist()
                
                # Metinleri önişle ve ekle
                processed_texts = [self.preprocess_text(text) for text in markdown_texts + code_texts]
                all_texts.extend(processed_texts)
                
                # Belleği temizle
                if len(all_texts) > batch_size * 2:
                    # TF-IDF modelini ara batch ile güncelle
                    if self.vectorizer is None:
                        self.vectorizer = TfidfVectorizer(max_features=1000, 
                                                        stop_words='english',
                                                        dtype=np.float32)  # float32 kullan
                    self.vectorizer.fit(all_texts)
                    all_texts = []
        
        # Kalan metinlerle son bir fit işlemi yap
        if all_texts:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=1000, 
                                                stop_words='english',
                                                dtype=np.float32)
            self.vectorizer.fit(all_texts)
        
        print("TF-IDF model eğitimi tamamlandı.")

    def load_transformer_model(self):
        print("Transformer model yükleniyor...")
        self.transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Transformer model yüklendi.")
        
    def get_embeddings(self, texts, batch_size=32):
        """Metinleri batch halinde embedding'e çevirir"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.transformer_model.encode(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def extract_features_batch(self, markdown_texts, code_texts, batch_size=32):
        """Birden fazla metin çifti için özellikleri çıkarır"""
        # Metinleri birleştir
        combined_texts = [f"{m} {c}" for m, c in zip(markdown_texts, code_texts)]
        
        # TF-IDF özellikleri
        tfidf_features = self.vectorizer.transform([self.preprocess_text(text) for text in combined_texts]).toarray()
        
        # Transformer özellikleri - batch processing
        transformer_features = self.get_embeddings(combined_texts, batch_size)
        
        # Özellikleri birleştir
        return np.concatenate([tfidf_features, transformer_features], axis=1)

    def train_model(self):
        print("Model eğitimi başlıyor...")
        
        # Test seti için daha fazla veri kullan
        max_notebooks = 5000  # 1000'den 5000'e çıkardık
        batch_size = 32
        
        # Rastgele notebook seç
        notebook_ids = list(self.notebooks.keys())
        np.random.shuffle(notebook_ids)
        selected_notebooks = notebook_ids[:max_notebooks]
        
        print(f"Toplam {max_notebooks} notebook üzerinde eğitim yapılacak...")
        
        X = []
        y = []
        markdown_batch = []
        code_batch = []
        self.markdown_texts = []  # Görselleştirme için ekledik
        self.code_texts = []      # Görselleştirme için ekledik
        
        for notebook_id in tqdm(selected_notebooks, desc="Özellik çıkarımı"):
            try:
                notebook = self.notebooks[notebook_id]
                markdown_cells = notebook['markdown']
                code_cells = notebook['code']
                
                if len(markdown_cells) > 0 and len(code_cells) > 0:
                    markdown_text = markdown_cells.iloc[0]['source']
                    code_text = code_cells.iloc[0]['source']
                    
                    # Metinleri sakla
                    self.markdown_texts.append(markdown_text)
                    self.code_texts.append(code_text)
                    
                    # Pozitif örnek için batch'e ekle
                    markdown_batch.append(markdown_text)
                    code_batch.append(code_text)
                    
                    # Negatif örnek için
                    random_notebook = np.random.choice([n for n in selected_notebooks if n != notebook_id])
                    random_code = np.random.choice(self.notebooks[random_notebook]['code']['source'].values)
                    markdown_batch.append(markdown_text)
                    code_batch.append(random_code)
                    
                    # Batch dolduğunda işle
                    if len(markdown_batch) >= batch_size:
                        features = self.extract_features_batch(markdown_batch, code_batch, batch_size)
                        X.extend(features)
                        y.extend([1, 0] * (len(markdown_batch) // 2))
                        
                        markdown_batch = []
                        code_batch = []
            
            except Exception as e:
                print(f"Notebook {notebook_id} işlenirken hata: {str(e)}")
                continue
        
        # Kalan batch'i işle
        if markdown_batch:
            features = self.extract_features_batch(markdown_batch, code_batch, batch_size)
            X.extend(features)
            y.extend([1, 0] * (len(markdown_batch) // 2))
        
        X = np.array(X)
        y = np.array(y)
        
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # XGBoost modelini eğit
        print("\nModel eğitimi başlıyor...")
        if not hasattr(self, 'xgb_model') or self.xgb_model is None:
            self.xgb_model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
        
        # Modeli eğit ve tahmin olasılıklarını sakla
        self.xgb_model.fit(X_train, y_train)
        self.y_pred = self.xgb_model.predict(X_test)
        self.y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        self.y_test = y_test
        self.X_test = X_test
        
        # Model performansını değerlendir
        accuracy = accuracy_score(y_test, self.y_pred)
        precision = precision_score(y_test, self.y_pred)
        recall = recall_score(y_test, self.y_pred)
        f1 = f1_score(y_test, self.y_pred)
        
        # PDF raporu oluştur
        self.create_report({
            'Doğruluk (Accuracy)': accuracy,
            'Hassasiyet (Precision)': precision,
            'Duyarlılık (Recall)': recall,
            'F1 Skoru': f1
        })
        
        print("\nModel Performansı:")
        print(f"Doğruluk (Accuracy): {accuracy:.3f}")
        print(f"Hassasiyet (Precision): {precision:.3f}")
        print(f"Duyarlılık (Recall): {recall:.3f}")
        print(f"F1 Skoru: {f1:.3f}")
    
    def create_report(self, metrics):
        doc = SimpleDocTemplate("model_results.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Başlık
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Model Değerlendirme Raporu", title_style))
        story.append(Spacer(1, 12))
        
        # Tarih
        date_style = ParagraphStyle(
            'Date',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=30
        )
        story.append(Paragraph(f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
        story.append(Spacer(1, 20))
        
        # Veri seti bilgileri
        story.append(Paragraph("Veri Seti Bilgileri:", styles['Heading2']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Toplam Notebook Sayısı: {len(self.notebooks)}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Metrikler tablosu
        story.append(Paragraph("Model Metrikleri:", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        data = [['Metrik', 'Değer']]
        for metric_name, metric_value in metrics.items():
            data.append([metric_name, f"{metric_value:.3f}"])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        doc.build(story)
        print("\nRapor 'model_results.pdf' dosyasına kaydedildi.")

def main():
    print("Program başlatılıyor...")
    processor = AI4CodeProcessor()
    
    try:
        # Veri yükleme ve hazırlık
        print("Veri yükleme işlemi başlıyor...")
        processor.load_data()
        print("Veri yükleme tamamlandı.")
    
        # TF-IDF model
        print("TF-IDF model oluşturuluyor...")
        processor.create_tfidf_model()
    
        # Transformer model
        print("Transformer model yükleniyor...")
        processor.load_transformer_model()
        
        # Model eğitimi
        print("Model eğitimi başlıyor...")
        processor.train_model()
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()