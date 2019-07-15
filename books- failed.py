##############################################
##############################################
##[Failed attempt to extract genre from tags]#
##############################################
##############################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

books_df = pd.read_csv('books.csv')

tags = pd.read_csv('tags.csv')
bt = pd.read_csv('book_tags.csv')
bt = bt.merge( tags, on = 'tag_id' )
bt = bt.merge( books_df[[ 'goodreads_book_id', 'title','authors','original_title']], on = 'goodreads_book_id' )

andi = ['The Hunger Games', 'Catching Fire', 'Mockingjay', 'The Hobbit or There and Back Again']
budi = ["Harry Potter and the Philosopher's Stone", "Harry Potter and the Chamber of Secrets", "Harry Potter and the Prisoner of Azkaban"]
ciko = ['Robots and Empire']
dedi = ['Nine Parts of Desire: The Hidden World of Islamic Women', "A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam", "No god but God: The Origins, Evolution, and Future of Islam"]
ello = ['Doctor Sleep','The Story of Doctor Dolittle',"Bridget Jones's Diary"]

universe = andi + budi + ciko + dedi + ello

bks = bt[bt.original_title.isin(universe)]

#get genre from tags
tag_counts = bks.groupby( 'tag_name' ).tag_name.count().sort_values( ascending = False )
top_tags = tag_counts.head(100).index
print(top_tags)

genres = ['fiction', 'ebook', 'audiobook','novel', 'fantasy', 'series', 'sci-fi-fantasy','young-adult','classics', 'ya-fantasy', 'young-adult-fiction',
         'childrens','classic', '20th-century', 'childhood-favorites','mystery','nonfiction','magic','paranormal']

book_universe = bt[bt.tag_name.isin(genres)]
book_universe = book_universe.dropna(subset = ['original_title'])
book_universe.count()

grouped = book_universe.groupby('original_title')

a =[]
quack = ""
for key, item in grouped:
  for tag in item['tag_name']:
      quack = quack + tag + '//'
  a.append({'original_title': key,'tags': quack,'authors': item['authors'].iloc[0]})
booksdata = pd.DataFrame(a)

booksdata['authors'] = booksdata['authors'].str.replace(', ','//')
booksdata['tag_auth'] = booksdata['authors'] + '//' + booksdata['tags']

from sklearn.feature_extraction.text import CountVectorizer
model = CountVectorizer(tokenizer=lambda x: x.split('//'))
matrixFeature = model.fit_transform(booksdata['tag_auth'])

from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixFeature)

from sklearn.metrics.pairwise import cosine_similarity
score = cosine_similarity(matrixFeature)

buku = "The Hunger Games"
indexSuka = booksdata[booksdata['original_title'] == buku].index.values[0]

daftarScore = list(enumerate(score[indexSuka]))
sortDaftarScore = sorted(
    daftarScore,
    key = lambda j: j[1],
    reverse = True
)

similarBooks = []
for i in sortDaftarScore:
    if i[1] > 0.9999999:
        similarBooks.append(i)
len(similarBooks)

import random
rekomendasi = random.choices(similarBooks, k=5)
# print(rekomendasi)

for i in rekomendasi:
    data = booksdata.iloc[i[0]].values
    print(data)

print("AAA")