import os
import pickle
from pandas.api.types import is_numeric_dtype, is_list_like
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch


class Data():
    def __init__(self):
        load_dotenv()
        self.cols = ["duration", "méta_score", "description", "writers", "directors", "stars", "genres"]
        self.df = pd.read_csv(os.getenv("DATASET_PATH"))
        self.df = self.df.dropna(subset=self.cols, how="any", axis=0)
        self.df = self.replace_nums(self.df, self.cols)
        self.df["val"] = self.comb_row(self.df, self.cols)    

    def replace_nums(self, df, cols):
        for c in cols:
            if is_numeric_dtype(df[c].dtype):
                df[c] = df[c].astype(str)

        return df

    def comb_row(self, df, cols, sep=" | "):
        s = ""
        for c in cols:
            s += sep + c + ":" + df[c]

        return s

    def get_vals(self):
        return self.df["val"].to_list()

    def get_val(self, col, search):
        return self.df.loc[self.df[col] == search]["val"].to_list()[0]

def load_from_pickle(var_name, func, *args, **kwargs):
    try:
        content = pickle.load(open(var_name + ".pickle", "rb"))
    except(OSError, IOError) as e:
        content = func(*args, **kwargs)
        pickle.dump(content, open(var_name + ".pickle", "wb"))

obj = Data()
docs = obj.get_vals()

search = "Batman"
query = obj.get_val("title", search)

print(query)

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print(model.prompts)

try:
    document_embeddings = pickle.load(open("doc_embed.pickle", "rb"))
except(OSError, IOError) as e:
    document_embeddings = model.encode(docs)
    pickle.dump(document_embeddings, open("doc_embed.pickle", "wb"))

query_embeddings = model.encode(query, prompt_name = "document")

similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
top_sim, top_is = torch.topk(similarity, k = 20)
top_sim = top_sim.flatten()
top_is = top_is.flatten()
for i in range(len(top_is)):
    index = top_is[i].item()
    sim_score = top_sim[i].item()
    movie = obj.df.iloc[index]
    print(movie["title"] + " " + movie["release_date"] + " " + str(sim_score))






