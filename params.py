"""
Contem todos os parâmetros usados no programa.
"""

DATA_DIR: str = 'data'
TARGET_COLUMN_NAME: str = 'sentiment'
FILE_NAME: str = 'IMDB Dataset.csv'

epochs: int = 10
metric: str = 'accuracy'

sizes: dict = {
    'full': 50000,
    'train': 25000,
    'test': 20000
}
