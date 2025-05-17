import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import json
import os
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class NotebookAnalyzer:
    def __init__(self, notebooks_data, model, vectorizer):
        self.notebooks = notebooks_data
        self.model = model
        self.vectorizer = vectorizer
        self.stats = {}
        plt.style.use('seaborn-v0_8')
    
    def analyze_notebook_statistics(self):
        """Notebook istatistiklerini analiz eder"""
        markdown_counts = []
        code_counts = []
        total_cells = []
        
        for notebook in self.notebooks.values():
            n_markdown = len(notebook['markdown'])
            n_code = len(notebook['code'])
            
            markdown_counts.append(n_markdown)
            code_counts.append(n_code)
            total_cells.append(n_markdown + n_code)
        
        self.stats['markdown_counts'] = markdown_counts
        self.stats['code_counts'] = code_counts
        self.stats['total_cells'] = total_cells
    
    def plot_cell_distribution(self):
        """Hücre dağılımını görselleştirir"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Markdown vs Code hücresi dağılımı
        data = {
            'Markdown Hücreleri': np.mean(self.stats['markdown_counts']),
            'Kod Hücreleri': np.mean(self.stats['code_counts'])
        }
        
        colors = ['#FF9999', '#66B2FF']
        ax1.bar(data.keys(), data.values(), color=colors)
        ax1.set_title('Ortalama Hücre Dağılımı')
        ax1.set_ylabel('Ortalama Hücre Sayısı')
        
        # Toplam hücre histogramı
        sns.histplot(data=self.stats['total_cells'], ax=ax2, bins=30, color='#99FF99')
        ax2.set_title('Notebook Başına Toplam Hücre Dağılımı')
        ax2.set_xlabel('Toplam Hücre Sayısı')
        ax2.set_ylabel('Notebook Sayısı')
        
        plt.tight_layout()
        plt.savefig('cell_distribution.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Confusion matrix görselleştirmesi"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Gerçek Değer')
        plt.xlabel('Tahmin')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """ROC eğrisi görselleştirmesi"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()
    
    def plot_feature_importance(self):
        """XGBoost feature importance görselleştirmesi"""
        importance_type = 'weight'  # veya 'gain', 'cover'
        importances = self.model.get_booster().get_score(importance_type=importance_type)
        
        # En önemli 20 özelliği al
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20])
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_importances)), list(sorted_importances.values()))
        plt.xticks(range(len(sorted_importances)), list(sorted_importances.keys()), rotation=45, ha='right')
        plt.title('En Önemli 20 Özellik')
        plt.xlabel('Özellikler')
        plt.ylabel(f'Önem Skoru ({importance_type})')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def generate_detailed_metrics(self, y_true, y_pred, y_pred_proba):
        """Detaylı performans metriklerini hesaplar ve görselleştirir"""
        # Sınıflandırma raporu
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Metrik görselleştirmesi
        metrics = {
            'Doğruluk (Accuracy)': report['accuracy'],
            'Hassasiyet (Precision)': report['1']['precision'],
            'Duyarlılık (Recall)': report['1']['recall'],
            'F1 Skoru': report['1']['f1-score']
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values(), color=['#2ecc71', '#3498db', '#9b59b6', '#f1c40f'])
        
        # Değerleri çubukların üzerine ekle
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.title('Model Performans Metrikleri')
        plt.ylim(0, 1.1)  # Y ekseni 0-1 arasında
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.close()
        
        # Precision-Recall eğrisi
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='#e74c3c', lw=2)
        plt.xlabel('Duyarlılık (Recall)')
        plt.ylabel('Hassasiyet (Precision)')
        plt.title('Precision-Recall Eğrisi')
        plt.grid(True)
        plt.savefig('precision_recall_curve.png')
        plt.close()
        
        return metrics

    def generate_example_matches(self, X_test, y_test, markdown_texts, code_texts, n_examples=10):
        """Doğru ve yanlış eşleşme örneklerini gösterir (arttırılmış örnek sayısı)"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        correct_matches = []
        incorrect_matches = []
        
        # Örnekleri güven skorlarıyla birlikte sakla
        examples = []
        for i, (pred, true, proba) in enumerate(zip(y_pred, y_test, y_pred_proba)):
            examples.append({
                'markdown': markdown_texts[i],
                'code': code_texts[i],
                'prediction': pred,
                'true': true,
                'confidence': proba,
                'is_correct': pred == true
            })
        
        # Güven skoruna göre sırala
        correct_examples = sorted(
            [ex for ex in examples if ex['is_correct']],
            key=lambda x: abs(x['confidence'] - 0.5),
            reverse=True
        )[:n_examples]
        
        incorrect_examples = sorted(
            [ex for ex in examples if not ex['is_correct']],
            key=lambda x: abs(x['confidence'] - 0.5),
            reverse=True
        )[:n_examples]
        
        # HTML şablonu güncellendi
        html_template = '''
        <!DOCTYPE html>
        <html lang="tr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI4Code Eşleştirme Örnekleri ve Performans Metrikleri</title>
            <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism-tomorrow.min.css" rel="stylesheet">
            <style>
                /* Mevcut stiller korundu */
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    margin: 40px auto;
                    max-width: 1200px;
                    padding: 0 20px;
                    line-height: 1.6;
                    background-color: #f8f9fa;
                }
                .example { 
                    border: 1px solid #e9ecef;
                    padding: 25px;
                    margin: 30px 0;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                    background-color: white;
                }
                .correct { 
                    border-left: 6px solid #28a745;
                }
                .incorrect { 
                    border-left: 6px solid #dc3545;
                }
                .markdown-content {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 15px 0;
                    font-size: 16px;
                }
                pre[class*="language-"] {
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 8px;
                    font-size: 14px;
                    background-color: #282c34;
                }
                h1 {
                    color: #343a40;
                    text-align: center;
                    margin: 40px 0;
                    font-size: 2.5em;
                }
                h2 {
                    color: #495057;
                    border-bottom: 2px solid #e9ecef;
                    padding-bottom: 10px;
                    margin: 40px 0 20px;
                }
                h3 {
                    color: #6c757d;
                    margin: 20px 0 10px;
                }
                .status {
                    display: inline-block;
                    padding: 8px 16px;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 500;
                    margin-top: 15px;
                }
                .status.correct {
                    background-color: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }
                .status.incorrect {
                    background-color: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }
                
                /* Yeni stiller eklendi */
                .metrics-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }
                
                .metric-card {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }
                
                .metric-value {
                    font-size: 2em;
                    font-weight: 600;
                    color: #2c3e50;
                    margin: 10px 0;
                }
                
                .metric-label {
                    color: #666;
                    font-size: 0.9em;
                }
                
                .confidence-bar {
                    height: 6px;
                    background: #e9ecef;
                    border-radius: 3px;
                    margin-top: 10px;
                }
                
                .confidence-level {
                    height: 100%;
                    border-radius: 3px;
                    transition: width 0.3s ease;
                }
                
                .high-confidence {
                    background: #2ecc71;
                }
                
                .medium-confidence {
                    background: #f1c40f;
                }
                
                .low-confidence {
                    background: #e74c3c;
                }
            </style>
        </head>
        <body>
        <div class="container">
            <h1>📊 Model Performans Analizi</h1>
            
            <!-- Performans Metrikleri -->
            <div class="metrics-container">
                <!-- Metrikler dinamik olarak eklenecek -->
            </div>
            
            <h2>✅ Doğru Eşleşme Örnekleri</h2>
            <!-- Doğru eşleşmeler -->
            
            <h2>❌ Yanlış Eşleşme Örnekleri</h2>
            <!-- Yanlış eşleşmeler -->
        </div>
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/prism.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-python.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-markdown.min.js"></script>
        </body>
        </html>
        '''
        
        # Performans metriklerini hesapla
        metrics = self.generate_detailed_metrics(y_test, y_pred, y_pred_proba)
        
        with open('example_matches.html', 'w', encoding='utf-8') as f:
            f.write(html_template)
            
            # Metrikleri ekle
            f.write('<div class="metrics-container">')
            for metric_name, metric_value in metrics.items():
                f.write(f'''
                    <div class="metric-card">
                        <div class="metric-label">{metric_name}</div>
                        <div class="metric-value">{metric_value:.3f}</div>
                    </div>
                ''')
            f.write('</div>')
            
            # Doğru eşleşmeleri ekle
            f.write('<h2>✅ Doğru Eşleşme Örnekleri</h2>')
            for match in correct_examples:
                confidence_class = 'high-confidence' if match['confidence'] > 0.8 else 'medium-confidence' if match['confidence'] > 0.6 else 'low-confidence'
                f.write(f'''
                    <div class="example correct">
                        <h3>Markdown:</h3>
                        <div class="markdown-content">{match["markdown"]}</div>
                        <h3>Kod:</h3>
                        <pre><code class="language-python">{match["code"].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</code></pre>
                        <div class="confidence-bar">
                            <div class="confidence-level {confidence_class}" style="width: {match["confidence"]*100}%"></div>
                        </div>
                        <div class="status correct">
                            ✓ Doğru Eşleşme (Güven: {match["confidence"]:.2%})
                        </div>
                    </div>
                ''')
            
            # Yanlış eşleşmeleri ekle
            f.write('<h2>❌ Yanlış Eşleşme Örnekleri</h2>')
            for match in incorrect_examples:
                confidence_class = 'high-confidence' if match['confidence'] > 0.8 else 'medium-confidence' if match['confidence'] > 0.6 else 'low-confidence'
                f.write(f'''
                    <div class="example incorrect">
                        <h3>Markdown:</h3>
                        <div class="markdown-content">{match["markdown"]}</div>
                        <h3>Kod:</h3>
                        <pre><code class="language-python">{match["code"].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}</code></pre>
                        <div class="confidence-bar">
                            <div class="confidence-level {confidence_class}" style="width: {match["confidence"]*100}%"></div>
                        </div>
                        <div class="status incorrect">
                            Model Tahmini: {"✓ Eşleşir" if match["prediction"] == 1 else "✗ Eşleşmez"} (Güven: {match["confidence"]:.2%}),
                            Gerçek Değer: {"✓ Eşleşir" if match["true"] == 1 else "✗ Eşleşmez"}
                        </div>
                    </div>
                ''')

def main():
    # Ana AI4Code modelinden veri ve modeli al
    from ai4code_model import AI4CodeProcessor
    
    processor = AI4CodeProcessor()
    processor.load_data()
    processor.create_tfidf_model()
    processor.load_transformer_model()
    processor.train_model()
    
    # Analiz ve görselleştirme
    analyzer = NotebookAnalyzer(
        notebooks_data=processor.notebooks,
        model=processor.xgb_model,
        vectorizer=processor.vectorizer
    )
    
    # İstatistikleri hesapla
    analyzer.analyze_notebook_statistics()
    
    # Görselleştirmeleri oluştur
    analyzer.plot_cell_distribution()
    analyzer.plot_confusion_matrix(processor.y_test, processor.y_pred)
    analyzer.plot_roc_curve(processor.y_test, processor.y_pred_proba)
    analyzer.plot_feature_importance()
    
    # Örnek eşleşmeleri ve performans metriklerini göster
    analyzer.generate_example_matches(
        processor.X_test, 
        processor.y_test, 
        processor.markdown_texts, 
        processor.code_texts,
        n_examples=10  # Örnek sayısı arttırıldı
    )
    
    print("\nTüm görselleştirmeler ve analizler tamamlandı!")
    print("Oluşturulan dosyalar:")
    print("- cell_distribution.png")
    print("- confusion_matrix.png")
    print("- roc_curve.png")
    print("- feature_importance.png")
    print("- performance_metrics.png")
    print("- precision_recall_curve.png")
    print("- example_matches.html")

if __name__ == "__main__":
    main() 