from pycaret.nlp import *
import pandas as pd

CSV_PATH = 'data.csv'

import warnings
warnings.filterwarnings("ignore")

def startpy():

    data = pd.read_csv(CSV_PATH)

    data = data[:200]

    setup(data = data, target = 'description')

    lda = create_model('lda')

    processed_data = assign_model(lda)

    processed_data.to_csv('processed_data.csv')

if __name__ == '__main__':
    startpy()


