import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import json
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModelAnalyzer:
    def __init__(self):
        """Model analizi için sınıf başlatıcı"""
        self.models = {
            'tfidf': pd.read_csv('submission_tfidf.csv'),
            'transformer': pd.read_csv('submission_transformer.csv'),
            'xgboost': pd.read_csv('submission_xgboost.csv'),
            'ensemble': pd.read_csv('submission_ensemble.csv')
        }
        self.train_data = pd.read_csv('train.csv')
        self.train_orders = pd.read_csv('train_orders.csv')
        
    def analyze_model_performance(self):
        """Her model için performans metriklerini hesapla"""
        print("Model performansları analiz ediliyor...")
        performance_metrics = {}
        
        for model_name, predictions in self.models.items():
            metrics = self._calculate_metrics(predictions)
            performance_metrics[model_name] = metrics
            
        # Sonuçları JSON olarak kaydet
        with open('model_performance.json', 'w') as f:
            json.dump(performance_metrics, f, indent=4)
            
        return performance_metrics
    
    def _calculate_metrics(self, predictions):
        """Tek bir model için metrikleri hesapla"""
        metrics = {
            'total_predictions': len(predictions),
            'unique_notebooks': len(predictions['id'].unique()),
            'avg_rank': predictions['rank'].mean(),
            'rank_distribution': predictions['rank'].value_counts().to_dict()
        }
        return metrics
    
    def create_performance_visualizations(self):
        """Model performans görselleştirmeleri oluştur"""
        print("Performans görselleştirmeleri oluşturuluyor...")
        
        # Klasör oluştur
        Path('analysis_outputs').mkdir(exist_ok=True)
        
        # Model karşılaştırma grafiği
        self._plot_model_comparison()
        
        # Rank dağılımı grafiği
        self._plot_rank_distribution()
        
        # Notebook bazında performans ısı haritası
        self._plot_notebook_heatmap()
        
        # İnteraktif performans grafiği
        self._create_interactive_plot()
        
    def _plot_model_comparison(self):
        """Model performanslarını karşılaştıran çubuk grafik"""
        performance_data = []
        for model_name, predictions in self.models.items():
            correct_matches = len(predictions[predictions['rank'] == 1])
            accuracy = correct_matches / len(predictions)
            performance_data.append({
                'Model': model_name,
                'Doğruluk': accuracy
            })
            
        df = pd.DataFrame(performance_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Model', y='Doğruluk')
        plt.title('Model Doğruluk Oranları Karşılaştırması')
        plt.savefig('analysis_outputs/model_comparison.png')
        plt.close()
        
    def _plot_rank_distribution(self):
        """Tahmin sıralama dağılımı grafiği"""
        plt.figure(figsize=(12, 6))
        for model_name, predictions in self.models.items():
            sns.kdeplot(data=predictions['rank'], label=model_name)
            
        plt.title('Model Tahmin Sıralaması Dağılımları')
        plt.xlabel('Sıralama')
        plt.ylabel('Yoğunluk')
        plt.legend()
        plt.savefig('analysis_outputs/rank_distribution.png')
        plt.close()
        
    def _plot_notebook_heatmap(self):
        """Notebook bazında performans ısı haritası"""
        notebook_performance = {}
        
        for model_name, predictions in self.models.items():
            notebook_metrics = predictions.groupby('id').agg({
                'rank': ['mean', 'std']
            }).reset_index()
            
            for _, row in notebook_metrics.iterrows():
                if row['id'] not in notebook_performance:
                    notebook_performance[row['id']] = {}
                notebook_performance[row['id']][model_name] = row['rank']['mean']
        
        df = pd.DataFrame(notebook_performance).T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Notebook Bazında Model Performansları')
        plt.savefig('analysis_outputs/notebook_heatmap.png')
        plt.close()
        
    def _create_interactive_plot(self):
        """Plotly ile interaktif performans grafiği"""
        performance_data = []
        
        for model_name, predictions in self.models.items():
            notebook_metrics = predictions.groupby('id').agg({
                'rank': ['mean', 'std', 'count']
            }).reset_index()
            
            for _, row in notebook_metrics.iterrows():
                performance_data.append({
                    'Model': model_name,
                    'Notebook': row['id'],
                    'Ortalama Sıralama': row['rank']['mean'],
                    'Standart Sapma': row['rank']['std'],
                    'Tahmin Sayısı': row['rank']['count']
                })
        
        df = pd.DataFrame(performance_data)
        
        fig = px.scatter(
            df,
            x='Ortalama Sıralama',
            y='Standart Sapma',
            size='Tahmin Sayısı',
            color='Model',
            hover_data=['Notebook'],
            title='Model Performans Analizi'
        )
        
        fig.write_html('analysis_outputs/interactive_performance.html')
        
    def analyze_error_patterns(self):
        """Hata paternlerini analiz et"""
        print("Hata paternleri analiz ediliyor...")
        error_analysis = {}
        
        for model_name, predictions in self.models.items():
            incorrect_matches = predictions[predictions['rank'] > 1]
            
            error_patterns = {
                'total_errors': len(incorrect_matches),
                'error_rate': len(incorrect_matches) / len(predictions),
                'avg_error_rank': incorrect_matches['rank'].mean(),
                'common_error_notebooks': incorrect_matches['id'].value_counts().head(5).to_dict()
            }
            
            error_analysis[model_name] = error_patterns
            
        # Hata analizini kaydet
        with open('analysis_outputs/error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=4)
            
        return error_analysis
    
    def generate_analysis_report(self):
        """Kapsamlı analiz raporu oluştur"""
        print("Analiz raporu oluşturuluyor...")
        
        performance_metrics = self.analyze_model_performance()
        error_patterns = self.analyze_error_patterns()
        
        report = []
        report.append("# AI4Code Model Analiz Raporu")
        report.append("\n## 1. Genel Performans Metrikleri")
        
        for model_name, metrics in performance_metrics.items():
            report.append(f"\n### {model_name.upper()} Model")
            report.append(f"- Toplam Tahmin: {metrics['total_predictions']}")
            report.append(f"- Benzersiz Notebook: {metrics['unique_notebooks']}")
            report.append(f"- Ortalama Sıralama: {metrics['avg_rank']:.2f}")
            
        report.append("\n## 2. Hata Analizi")
        
        for model_name, errors in error_patterns.items():
            report.append(f"\n### {model_name.upper()} Model Hataları")
            report.append(f"- Toplam Hata: {errors['total_errors']}")
            report.append(f"- Hata Oranı: {errors['error_rate']:.2%}")
            report.append(f"- Ortalama Hata Sıralaması: {errors['avg_error_rank']:.2f}")
            
        report.append("\n## 3. Öneriler")
        report.append("\n### Model İyileştirmeleri")
        report.append("- Transformer model mimarisinin güncellenmesi")
        report.append("- Ensemble ağırlıklarının optimize edilmesi")
        report.append("- Özellik mühendisliği sürecinin genişletilmesi")
        
        report.append("\n### Veri İyileştirmeleri")
        report.append("- Veri temizleme sürecinin geliştirilmesi")
        report.append("- Ek özellik çıkarımı")
        report.append("- Veri augmentasyon teknikleri")
        
        # Raporu kaydet
        with open('analysis_outputs/analysis_report.md', 'w') as f:
            f.write('\n'.join(report))

def main():
    # Model analizcisini oluştur
    analyzer = ModelAnalyzer()
    
    # Analizleri gerçekleştir
    analyzer.analyze_model_performance()
    analyzer.create_performance_visualizations()
    analyzer.analyze_error_patterns()
    analyzer.generate_analysis_report()
    
    print("Analiz tamamlandı! Sonuçlar 'analysis_outputs' klasöründe.")

if __name__ == "__main__":
    main() 