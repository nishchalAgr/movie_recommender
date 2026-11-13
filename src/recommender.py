import os
from pandas.api.types import is_numeric_dtype
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from similarity import Similarity

class Recommender():
    def __init__(self, col_names, col_weights):
        load_dotenv()
        self.col_names = col_names
        self.col_weights = col_weights
        self.df = pd.read_csv(os.getenv("DATASET_PATH"))
        self.df = self.df.dropna(subset=self.col_names, how="any", axis=0)
        self.df = self.replace_nums(self.df, self.col_names)

        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.similarities = [Similarity(n, self.df, model) for n in self.col_names]
             
    def replace_nums(self, df, cols):
        for c in cols:
            if is_numeric_dtype(df[c].dtype):
                df[c] = df[c].astype(str)
    
        return df

    def get_similarity_score(self, search, col_name):
        vectors = []
        series = self.df.loc[self.df[col_name] == search].iloc[0]
        for i, s in enumerate(self.similarities):
            vectors.append(s.get_similarity_score(series) * self.col_weights[i])

        return [series, torch.sum(torch.stack(vectors), dim=0)]