import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


job_df = pd.read_csv("Combined_Jobs_Final.csv")

job_df = job_df[['Status', 'Title', 'Position', 'Company', 'Job.Description']]

job_df.fillna('',inplace=True)

job_df = job_df.sample(n=1000,random_state=42)

from nltk.corpus import stopwords
import nltk
import re
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def cleaning(txt):
    txt = re.sub(r'[^a-zA-Z0-9\s]','',txt)
    tokens = nltk.word_tokenize(txt.lower())

    stemming = [ps.stem(w) for w in tokens if w not in stopwords.words('english')]
    return " ".join(stemming)
job_df['Job.Description'] = job_df['Job.Description'].astype(str).apply(lambda x: cleaning(x))
job_df['Title'] = job_df['Title'].astype(str).apply(lambda x: cleaning(x))
job_df['Position'] = job_df['Position'].astype(str).apply(lambda x: cleaning(x))

job_df['clean_text'] = job_df['Job.Description']+" "+job_df['Title']+job_df['Position']

tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(job_df['clean_text'])
similarity = cosine_similarity(matrix)

def recommend(title):

        indx = job_df[job_df['Title'] == title].index[0]
        indx = job_df.index.get_loc(indx)
        distances = sorted(list(enumerate(similarity[indx])), key=lambda x: x[1], reverse=True)[1:20]

        jobs = []
        for i in distances:
            jobs.append(job_df.iloc[i[0]].Title)
        return jobs
import pickle
pickle.dump(job_df,open('df.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
        
    
