import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import optuna
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    def __init__(self):
        """Model optimizasyonu için sınıf başlatıcı"""
        self.train_data = pd.read_csv('train.csv')
        self.train_orders = pd.read_csv('train_orders.csv')
        self.best_params = {}
        
    def optimize_tfidf(self, n_trials=50):
        """TF-IDF modelinin hiperparametrelerini optimize et"""
        print("TF-IDF parametreleri optimize ediliyor...")
        
        def objective(trial):
            params = {
                'max_features': trial.suggest_int('max_features', 1000, 10000),
                'min_df': trial.suggest_float('min_df', 0.0, 0.1),
                'max_df': trial.suggest_float('max_df', 0.5, 1.0),
                'ngram_range': (1, trial.suggest_int('max_ngram', 1, 3))
            }
            
            vectorizer = TfidfVectorizer(**params)
            scores = []
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(self.train_data):
                train_docs = self.train_data.iloc[train_idx]['source'].values
                val_docs = self.train_data.iloc[val_idx]['source'].values
                
                vectorizer.fit(train_docs)
                train_vectors = vectorizer.transform(train_docs)
                val_vectors = vectorizer.transform(val_docs)
                
                # Benzerlik skorlarını hesapla
                similarities = (train_vectors @ val_vectors.T).toarray()
                scores.append(np.mean(similarities))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['tfidf'] = study.best_params
        return study.best_params
    
    def optimize_transformer(self, n_trials=30):
        """Transformer modelinin hiperparametrelerini optimize et"""
        print("Transformer parametreleri optimize ediliyor...")
        
        def objective(trial):
            params = {
                'max_length': trial.suggest_int('max_length', 128, 512),
                'batch_size': trial.suggest_int('batch_size', 16, 64),
                'pooling': trial.suggest_categorical('pooling', ['mean', 'max', 'cls'])
            }
            
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            scores = []
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(self.train_data):
                train_docs = self.train_data.iloc[train_idx]['source'].values
                val_docs = self.train_data.iloc[val_idx]['source'].values
                
                # Embeddingler
                train_embeddings = model.encode(
                    train_docs,
                    batch_size=params['batch_size'],
                    max_length=params['max_length'],
                    show_progress_bar=False
                )
                
                val_embeddings = model.encode(
                    val_docs,
                    batch_size=params['batch_size'],
                    max_length=params['max_length'],
                    show_progress_bar=False
                )
                
                # Benzerlik skorları
                similarities = np.dot(train_embeddings, val_embeddings.T)
                scores.append(np.mean(similarities))
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['transformer'] = study.best_params
        return study.best_params
    
    def optimize_xgboost(self, X, y, n_trials=50):
        """XGBoost modelinin hiperparametrelerini optimize et"""
        print("XGBoost parametreleri optimize ediliyor...")
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss'
            }
            
            scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False
                )
                
                scores.append(model.best_score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['xgboost'] = study.best_params
        return study.best_params
    
    def optimize_ensemble_weights(self, predictions_dict, true_labels):
        """Ensemble model ağırlıklarını optimize et"""
        print("Ensemble ağırlıkları optimize ediliyor...")
        
        def objective(trial):
            weights = {
                'tfidf': trial.suggest_float('tfidf_weight', 0.0, 1.0),
                'transformer': trial.suggest_float('transformer_weight', 0.0, 1.0),
                'xgboost': trial.suggest_float('xgboost_weight', 0.0, 1.0)
            }
            
            # Ağırlıkları normalize et
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Ağırlıklı tahminler
            weighted_preds = np.zeros_like(true_labels, dtype=float)
            for model_name, preds in predictions_dict.items():
                weighted_preds += weights[model_name] * preds
            
            # Performans metriği
            accuracy = np.mean(np.round(weighted_preds) == true_labels)
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
        self.best_params['ensemble'] = study.best_params
        return study.best_params
    
    def save_best_params(self):
        """En iyi parametreleri kaydet"""
        with open('best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)
    
    def load_best_params(self):
        """Kaydedilmiş en iyi parametreleri yükle"""
        try:
            with open('best_params.json', 'r') as f:
                self.best_params = json.load(f)
            return True
        except FileNotFoundError:
            print("Kaydedilmiş parametre bulunamadı.")
            return False
    
    def cross_validate_models(self, X, y, n_splits=5):
        """Modelleri çapraz doğrulama ile değerlendir"""
        print("Çapraz doğrulama yapılıyor...")
        cv_results = {}
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for model_name in ['tfidf', 'transformer', 'xgboost']:
            scores = []
            print(f"\n{model_name.upper()} model değerlendiriliyor...")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if model_name == 'tfidf':
                    vectorizer = TfidfVectorizer(**self.best_params['tfidf'])
                    vectorizer.fit(X_train)
                    train_vectors = vectorizer.transform(X_train)
                    val_vectors = vectorizer.transform(X_val)
                    score = np.mean((train_vectors @ val_vectors.T).toarray())
                
                elif model_name == 'transformer':
                    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    train_embeddings = model.encode(X_train, **self.best_params['transformer'])
                    val_embeddings = model.encode(X_val, **self.best_params['transformer'])
                    score = np.mean(np.dot(train_embeddings, val_embeddings.T))
                
                else:  # xgboost
                    model = xgb.XGBClassifier(**self.best_params['xgboost'])
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                
                scores.append(score)
                print(f"Fold {fold}: {score:.4f}")
            
            cv_results[model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
            
            print(f"{model_name.upper()} ortalama skor: {np.mean(scores):.4f} (±{np.std(scores):.4f})")
        
        return cv_results

def main():
    # Optimizasyon sınıfını oluştur
    optimizer = ModelOptimizer()
    
    # TF-IDF optimizasyonu
    tfidf_params = optimizer.optimize_tfidf()
    print("\nEn iyi TF-IDF parametreleri:", tfidf_params)
    
    # Transformer optimizasyonu
    transformer_params = optimizer.optimize_transformer()
    print("\nEn iyi Transformer parametreleri:", transformer_params)
    
    # XGBoost için örnek veri
    X = np.random.rand(1000, 10)  # Örnek özellikler
    y = np.random.randint(0, 2, 1000)  # Örnek etiketler
    
    # XGBoost optimizasyonu
    xgboost_params = optimizer.optimize_xgboost(X, y)
    print("\nEn iyi XGBoost parametreleri:", xgboost_params)
    
    # Ensemble optimizasyonu için örnek tahminler
    predictions = {
        'tfidf': np.random.rand(1000),
        'transformer': np.random.rand(1000),
        'xgboost': np.random.rand(1000)
    }
    true_labels = np.random.randint(0, 2, 1000)
    
    # Ensemble ağırlıklarını optimize et
    ensemble_weights = optimizer.optimize_ensemble_weights(predictions, true_labels)
    print("\nEn iyi Ensemble ağırlıkları:", ensemble_weights)
    
    # En iyi parametreleri kaydet
    optimizer.save_best_params()
    
    # Çapraz doğrulama
    cv_results = optimizer.cross_validate_models(X, y)
    print("\nÇapraz doğrulama sonuçları:", cv_results)

if __name__ == "__main__":
    main() 