from pycaret.nlp import *
import pandas as pd

CSV_PATH = 'data.csv'

import warnings
warnings.filterwarnings("ignore")

def startpy():

    data = pd.read_csv(CSV_PATH)

    data = data[:200]

    setup(data = data, target = 'description')

    lda = create_model('lda', num_topics = 6, multi_core = True)

    assign_model(lda)

    plot_model(lda, plot='wordcloud',save=True)
    plot_model(lda, plot='bigram',save=True)
    plot_model(lda, plot='frequency',save=True)
    plot_model(lda, plot='trigram',save=True)

if __name__ == '__main__':
    startpy()


