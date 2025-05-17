import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        """Model değerlendirme sınıfı başlatıcı"""
        self.models = {
            'tfidf': pd.read_csv('submission_tfidf.csv'),
            'transformer': pd.read_csv('submission_transformer.csv'),
            'xgboost': pd.read_csv('submission_xgboost.csv'),
            'ensemble': pd.read_csv('submission_ensemble.csv')
        }
        self.train_orders = pd.read_csv('train_orders.csv')
        
        # Değerlendirme sonuçları için klasör oluştur
        Path('evaluation_outputs').mkdir(exist_ok=True)
        
    def calculate_metrics(self):
        """Her model için performans metriklerini hesapla"""
        print("Performans metrikleri hesaplanıyor...")
        metrics = {}
        
        for model_name, predictions in self.models.items():
            model_metrics = self._evaluate_single_model(predictions)
            metrics[model_name] = model_metrics
            
        # Metrikleri JSON olarak kaydet
        with open('evaluation_outputs/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
    
    def _evaluate_single_model(self, predictions):
        """Tek bir model için metrikleri hesapla"""
        metrics = {
            'total_predictions': len(predictions),
            'unique_notebooks': len(predictions['id'].unique()),
            'rank_distribution': predictions['rank'].value_counts().to_dict(),
            'avg_rank': predictions['rank'].mean(),
            'top_1_accuracy': len(predictions[predictions['rank'] == 1]) / len(predictions),
            'top_3_accuracy': len(predictions[predictions['rank'] <= 3]) / len(predictions)
        }
        return metrics
    
    def create_visualizations(self):
        """Performans görselleştirmeleri oluştur"""
        print("Görselleştirmeler oluşturuluyor...")
        
        # Model karşılaştırma grafiği
        self._plot_model_comparison()
        
        # Rank dağılımı grafiği
        self._plot_rank_distribution()
        
        # Notebook bazında performans grafiği
        self._plot_notebook_performance()
        
        # Hata analizi grafiği
        self._plot_error_analysis()
        
    def _plot_model_comparison(self):
        """Model performanslarını karşılaştıran çubuk grafik"""
        metrics = self.calculate_metrics()
        
        accuracies = {
            model: metrics[model]['top_1_accuracy']
            for model in metrics
        }
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
        plt.title('Model Doğruluk Oranları')
        plt.ylabel('Top-1 Doğruluk')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('evaluation_outputs/model_comparison.png')
        plt.close()
        
    def _plot_rank_distribution(self):
        """Tahmin sıralama dağılımı grafiği"""
        plt.figure(figsize=(12, 6))
        
        for model_name, predictions in self.models.items():
            sns.kdeplot(data=predictions['rank'], label=model_name)
            
        plt.title('Tahmin Sıralama Dağılımları')
        plt.xlabel('Sıralama')
        plt.ylabel('Yoğunluk')
        plt.legend()
        plt.tight_layout()
        plt.savefig('evaluation_outputs/rank_distribution.png')
        plt.close()
        
    def _plot_notebook_performance(self):
        """Notebook bazında performans grafiği"""
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
        plt.tight_layout()
        plt.savefig('evaluation_outputs/notebook_performance.png')
        plt.close()
        
    def _plot_error_analysis(self):
        """Hata analizi grafiği"""
        error_rates = {}
        
        for model_name, predictions in self.models.items():
            error_rates[model_name] = {
                'top_1_error': 1 - len(predictions[predictions['rank'] == 1]) / len(predictions),
                'top_3_error': 1 - len(predictions[predictions['rank'] <= 3]) / len(predictions)
            }
        
        df = pd.DataFrame(error_rates).T
        
        plt.figure(figsize=(10, 6))
        df.plot(kind='bar')
        plt.title('Model Hata Oranları')
        plt.ylabel('Hata Oranı')
        plt.legend(['Top-1 Hata', 'Top-3 Hata'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('evaluation_outputs/error_analysis.png')
        plt.close()
        
    def generate_evaluation_report(self):
        """Kapsamlı değerlendirme raporu oluştur"""
        print("Değerlendirme raporu oluşturuluyor...")
        
        metrics = self.calculate_metrics()
        
        report = []
        report.append("# AI4Code Model Değerlendirme Raporu")
        report.append("\n## 1. Genel Performans Metrikleri")
        
        for model_name, model_metrics in metrics.items():
            report.append(f"\n### {model_name.upper()} Model")
            report.append(f"- Toplam Tahmin: {model_metrics['total_predictions']}")
            report.append(f"- Benzersiz Notebook: {model_metrics['unique_notebooks']}")
            report.append(f"- Ortalama Sıralama: {model_metrics['avg_rank']:.2f}")
            report.append(f"- Top-1 Doğruluk: {model_metrics['top_1_accuracy']:.2%}")
            report.append(f"- Top-3 Doğruluk: {model_metrics['top_3_accuracy']:.2%}")
            
        report.append("\n## 2. Model Karşılaştırması")
        report.append("\nModeller arasındaki performans karşılaştırması için 'model_comparison.png' grafiğini inceleyiniz.")
        
        report.append("\n## 3. Sıralama Dağılımı")
        report.append("\nTahmin sıralama dağılımları için 'rank_distribution.png' grafiğini inceleyiniz.")
        
        report.append("\n## 4. Notebook Bazında Analiz")
        report.append("\nNotebook bazında performans analizi için 'notebook_performance.png' grafiğini inceleyiniz.")
        
        report.append("\n## 5. Hata Analizi")
        report.append("\nModel hata oranları için 'error_analysis.png' grafiğini inceleyiniz.")
        
        report.append("\n## 6. Öneriler")
        report.append("\n### Model İyileştirmeleri")
        report.append("- En iyi performans gösteren modelin mimarisini geliştir")
        report.append("- Ensemble ağırlıklarını optimize et")
        report.append("- Özellik mühendisliğini genişlet")
        
        report.append("\n### Veri İyileştirmeleri")
        report.append("- Veri temizleme sürecini geliştir")
        report.append("- Veri augmentasyon teknikleri uygula")
        report.append("- Ek özellikler ekle")
        
        # Raporu kaydet
        with open('evaluation_outputs/evaluation_report.md', 'w') as f:
            f.write('\n'.join(report))

def main():
    # Değerlendirme sınıfını oluştur
    evaluator = ModelEvaluator()
    
    # Metrikleri hesapla
    evaluator.calculate_metrics()
    
    # Görselleştirmeleri oluştur
    evaluator.create_visualizations()
    
    # Raporu oluştur
    evaluator.generate_evaluation_report()
    
    print("Değerlendirme tamamlandı! Sonuçlar 'evaluation_outputs' klasöründe.")

if __name__ == "__main__":
    main() 