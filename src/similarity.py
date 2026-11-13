import pickle

class Similarity():
    def __init__(self, col_name, df, model):
        self.model = model
        self.col_name = col_name

        f_name = col_name + "_embeddings.pickle"

        try:
            self.doc_embeddings = pickle.load(open(f_name, "rb"))
        except(OSError, IOError) as e:
            self.doc_embeddings = self.model.encode(df[col_name].to_list())
            pickle.dump(self.doc_embeddings, open(f_name, "wb"))

    def get_similarity_score(self, series):
        query = series[self.col_name]
        embedding = self.model.encode(query, prompt_name = "document")
        return self.model.similarity(embedding, self.doc_embeddings)
