import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

books_df = pd.read_csv('books.csv')

# print(books_df)

df = books_df[['original_title','authors','language_code']]
df=df.dropna(subset=['original_title','authors','language_code'])
df = df.reset_index()
df['authors'] = df['authors'].str.replace(' ','')
df['authors'] = df['authors'].str.replace(',',' ')
df['language_code'] = df['language_code'].str.replace('-','')
df['features'] = df['authors'] + ' ' + df['original_title'] + ' ' + df['language_code']

from sklearn.feature_extraction.text import CountVectorizer
model=CountVectorizer(tokenizer=lambda x:x.split(' '))
matrixFeature=model.fit_transform(df['features'])

from sklearn.metrics.pairwise import cosine_similarity
score=cosine_similarity(matrixFeature)


andi = ['The Hunger Games', 'Catching Fire', 'Mockingjay', 'The Hobbit or There and Back Again']
budi = ["Harry Potter and the Philosopher's Stone", "Harry Potter and the Chamber of Secrets", "Harry Potter and the Prisoner of Azkaban"]
ciko = ['Robots and Empire']
dedi = ['Nine Parts of Desire: The Hidden World of Islamic Women', "A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam", "No god but God: The Origins, Evolution, and Future of Islam"]
ello = ['Doctor Sleep','The Story of Doctor Dolittle',"Bridget Jones's Diary"]

import random
def get_similar_books(books):
    recomend = []
    for book in books:
        # print(book)
        try:
            indexSuka = df[df['original_title'] == book].index.values[0]
            daftarScore = list(enumerate(score[indexSuka]))
            sortDaftarScore = sorted(daftarScore,key = lambda j: j[1],reverse = True)
            for i in sortDaftarScore:
                if i[1] > .5:
                    recomend.append(i)
        except:
            pass
    final =[]
    # rekomendasi = shuffle(recomend)
    rekomendasi = sorted(recomend, key=lambda k: random.random())

    for i in rekomendasi[:5]:
        data = df.iloc[i[0]].values
        final.append(data[1])
        
    return final
    # for i in recomend:
    #     data = df.iloc[i[0]].values
    #     final.append(data[1])
    # return final
    

rekomen = {}
rekomen['andi'] = get_similar_books(andi)
rekomen['budi'] = get_similar_books(budi)
rekomen['ciko'] = get_similar_books(ciko)
rekomen['dedi'] = get_similar_books(dedi)
rekomen['ello'] = get_similar_books(ello)

pelanggan = ['andi','budi','ciko','dedi','ello']

for orang in pelanggan:
    print(f"Rekomendasi buku bagus untuk {orang}")
    for i in rekomen[orang]:
        print(f"-{i}")
    

