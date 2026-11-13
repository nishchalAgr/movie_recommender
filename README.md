# MOVIE RECOMMENDER 

Movie recommender based on applying cosine similarity to embeddings of the dataset and user input. Allows similarity to be calculated per movie descriptor. For example, the recommender can be configured to give more weight to genre when calculating the similarity matrix for a given input. Example usage is in `main.py`.

### Overview
- Dataset: https://www.kaggle.com/datasets/raedaddala/top-500-600-movies-of-each-year-from-1960-to-2024
- Embedding model: Qwen/Qwen3-Embedding-0.6B

### TODO
- [X] minimum implementation
- [ ] some sort of UI
- [X] movie data weights (ex. genre contributes more to similarity than cast)
- [ ] filters (ex. only return similar movies that were released after 2010)
