import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from tqdm import tqdm
import spacy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        """Veri ön işleme ve özellik mühendisliği sınıfı"""
        # NLTK gerekli dosyaları indir
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # spaCy modelini yükle
        self.nlp = spacy.load('en_core_web_sm')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.scaler = StandardScaler()
        
    def clean_code(self, code_text):
        """Kod hücresini temizle ve normalize et"""
        # Yorumları kaldır
        code_text = re.sub(r'#.*$', '', code_text, flags=re.MULTILINE)
        code_text = re.sub(r'""".*?"""', '', code_text, flags=re.DOTALL)
        code_text = re.sub(r"'''.*?'''", '', code_text, flags=re.DOTALL)
        
        # Gereksiz boşlukları temizle
        code_text = re.sub(r'\s+', ' ', code_text)
        
        # Özel karakterleri normalize et
        code_text = code_text.replace('\\n', ' ').replace('\\t', ' ')
        
        return code_text.strip()
    
    def clean_markdown(self, markdown_text):
        """Markdown hücresini temizle ve normalize et"""
        # Markdown formatını kaldır
        markdown_text = re.sub(r'#+ ', '', markdown_text)  # Başlıkları temizle
        markdown_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown_text)  # Linkleri temizle
        markdown_text = re.sub(r'`[^`]+`', '', markdown_text)  # Kod bloklarını temizle
        markdown_text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', markdown_text)  # Bold/italic temizle
        
        # Özel karakterleri temizle
        markdown_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', markdown_text)
        
        return markdown_text.strip()
    
    def extract_code_features(self, code_text):
        """Kod hücresinden özellikler çıkar"""
        cleaned_code = self.clean_code(code_text)
        
        features = {
            # Kod yapısı özellikleri
            'has_function': 1 if re.search(r'def\s+\w+\s*\(', code_text) else 0,
            'has_class': 1 if re.search(r'class\s+\w+', code_text) else 0,
            'has_import': 1 if re.search(r'import\s+\w+|from\s+\w+\s+import', code_text) else 0,
            'has_loop': 1 if re.search(r'for\s+|while\s+', code_text) else 0,
            'has_conditional': 1 if re.search(r'if\s+|elif\s+|else:', code_text) else 0,
            
            # Kod metrikleri
            'line_count': len(code_text.split('\n')),
            'char_count': len(code_text),
            'word_count': len(cleaned_code.split()),
            
            # Değişken ve fonksiyon sayıları
            'variable_count': len(re.findall(r'=\s*(?![=])', code_text)),
            'function_count': len(re.findall(r'def\s+\w+\s*\(', code_text)),
            
            # Yaygın kütüphane kullanımı
            'uses_pandas': 1 if re.search(r'pd\.|pandas\.', code_text) else 0,
            'uses_numpy': 1 if re.search(r'np\.|numpy\.', code_text) else 0,
            'uses_sklearn': 1 if re.search(r'sklearn\.', code_text) else 0,
            'uses_matplotlib': 1 if re.search(r'plt\.|matplotlib\.', code_text) else 0,
            
            # Kod karmaşıklığı
            'indentation_levels': max([len(line) - len(line.lstrip()) for line in code_text.split('\n')]) // 4
        }
        
        return features
    
    def extract_markdown_features(self, markdown_text):
        """Markdown hücresinden özellikler çıkar"""
        cleaned_text = self.clean_markdown(markdown_text)
        doc = self.nlp(cleaned_text)
        
        features = {
            # Metin yapısı özellikleri
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text),
            'sentence_count': len(list(doc.sents)),
            'avg_word_length': np.mean([len(word) for word in cleaned_text.split()]),
            
            # Markdown formatı özellikleri
            'heading_count': len(re.findall(r'#+\s', markdown_text)),
            'code_block_count': len(re.findall(r'```.*?```', markdown_text, re.DOTALL)),
            'link_count': len(re.findall(r'\[([^\]]+)\]\([^\)]+\)', markdown_text)),
            'list_item_count': len(re.findall(r'^\s*[-*]\s', markdown_text, re.MULTILINE)),
            
            # NLP özellikleri
            'noun_count': len([token for token in doc if token.pos_ == 'NOUN']),
            'verb_count': len([token for token in doc if token.pos_ == 'VERB']),
            'adj_count': len([token for token in doc if token.pos_ == 'ADJ']),
            'entity_count': len(doc.ents),
            
            # İçerik türü göstergeleri
            'is_question': 1 if '?' in markdown_text else 0,
            'has_numbers': 1 if bool(re.search(r'\d', markdown_text)) else 0,
            'has_code_reference': 1 if bool(re.search(r'`.*?`', markdown_text)) else 0
        }
        
        return features
    
    def calculate_similarity_features(self, markdown_text, code_text):
        """Markdown ve kod arasındaki benzerlik özelliklerini hesapla"""
        clean_markdown = self.clean_markdown(markdown_text)
        clean_code = self.clean_code(code_text)
        
        markdown_words = set(word_tokenize(clean_markdown.lower()))
        code_words = set(word_tokenize(clean_code.lower()))
        
        # Kelime örtüşmesi özellikleri
        common_words = markdown_words.intersection(code_words)
        
        features = {
            'common_word_count': len(common_words),
            'word_overlap_ratio': len(common_words) / (len(markdown_words) + len(code_words)) if (len(markdown_words) + len(code_words)) > 0 else 0,
            'markdown_coverage': len(common_words) / len(markdown_words) if len(markdown_words) > 0 else 0,
            'code_coverage': len(common_words) / len(code_words) if len(code_words) > 0 else 0
        }
        
        return features
    
    def process_notebook(self, notebook_data):
        """Notebook verilerini işle ve özellikleri çıkar"""
        processed_data = []
        
        markdown_cells = notebook_data[notebook_data['cell_type'] == 'markdown']
        code_cells = notebook_data[notebook_data['cell_type'] == 'code']
        
        for _, markdown in markdown_cells.iterrows():
            markdown_features = self.extract_markdown_features(markdown['source'])
            
            for _, code in code_cells.iterrows():
                code_features = self.extract_code_features(code['source'])
                similarity_features = self.calculate_similarity_features(
                    markdown['source'], 
                    code['source']
                )
                
                # Tüm özellikleri birleştir
                features = {
                    'cell_id': markdown['cell_id'],
                    'matched_code_id': code['cell_id'],
                    **markdown_features,
                    **code_features,
                    **similarity_features
                }
                
                processed_data.append(features)
        
        return pd.DataFrame(processed_data)
    
    def fit_transform(self, train_data):
        """Eğitim verisini işle ve dönüştür"""
        print("Veri ön işleme ve özellik çıkarımı yapılıyor...")
        processed_notebooks = []
        
        for notebook_id in tqdm(train_data['id'].unique()):
            notebook_cells = train_data[train_data['id'] == notebook_id]
            processed_df = self.process_notebook(notebook_cells)
            processed_df['notebook_id'] = notebook_id
            processed_notebooks.append(processed_df)
        
        # Tüm notebookları birleştir
        final_df = pd.concat(processed_notebooks, ignore_index=True)
        
        # Sayısal özellikleri ölçeklendir
        numeric_columns = final_df.select_dtypes(include=[np.number]).columns
        final_df[numeric_columns] = self.scaler.fit_transform(final_df[numeric_columns])
        
        return final_df
    
    def transform(self, test_data):
        """Test verisini dönüştür"""
        print("Test verisi işleniyor...")
        processed_notebooks = []
        
        for notebook_id in tqdm(test_data['id'].unique()):
            notebook_cells = test_data[test_data['id'] == notebook_id]
            processed_df = self.process_notebook(notebook_cells)
            processed_df['notebook_id'] = notebook_id
            processed_notebooks.append(processed_df)
        
        # Tüm notebookları birleştir
        final_df = pd.concat(processed_notebooks, ignore_index=True)
        
        # Sayısal özellikleri ölçeklendir
        numeric_columns = final_df.select_dtypes(include=[np.number]).columns
        final_df[numeric_columns] = self.scaler.transform(final_df[numeric_columns])
        
        return final_df

def main():
    # Örnek kullanım
    train_data = pd.read_csv('train.csv')
    
    # Önişleyici oluştur
    preprocessor = DataPreprocessor()
    
    # Veriyi işle
    processed_data = preprocessor.fit_transform(train_data)
    
    # İşlenmiş veriyi kaydet
    processed_data.to_csv('processed_features.csv', index=False)
    print("İşlenmiş özellikler 'processed_features.csv' dosyasına kaydedildi.")
    
    # Özellik istatistiklerini göster
    feature_stats = processed_data.describe()
    feature_stats.to_csv('feature_statistics.csv')
    print("Özellik istatistikleri 'feature_statistics.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    main() 