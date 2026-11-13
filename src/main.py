from recommender import Recommender
import torch
import torch.nn.functional as F
from prettytable import PrettyTable

def print_movie(movie, sim_score = ""):
    print(movie["title"] + " " + movie["release_date"] + " " + str(sim_score))

cols = ["duration", "méta_score", "description", "writers", "directors", "stars", "genres"]
weights = [0, 0, 10, 0, 10, 3, 1]
recommender = Recommender(cols, weights)

search = "Whiplash"
[input_series, sim_scores] = recommender.get_similarity_score(search, "title")
sim_scores = sim_scores / torch.max(sim_scores)
print(sim_scores)
top_sim, top_is = torch.topk(sim_scores, k = 50)
top_sim = top_sim.flatten()
top_is = top_is.flatten()

t = PrettyTable(['Title', 'Release Date', 'Meta Score', 'Sim_Score'])
for i in range(len(top_is)):
    index = top_is[i].item()
    sim_score = top_sim[i].item()
    movie = recommender.df.iloc[index]
    t.add_row([movie["title"], movie["release_date"], float(movie["méta_score"]), sim_score])

print(t)