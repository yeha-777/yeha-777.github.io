---
layout: post
title: Example for using the OpenAI API
#subtitle: There's lots to learn!
gh-repo: openai/openai-cookbook
gh-badge: [star, fork, follow]
#tags: [test]
comments: false
#author: Bill Smith
---


# Using Embeddings

```
import openai

embedding = openai.Embedding.create(
    input="Your text goes here", model="text-embedding-3-small"
)["data"][0]["embedding"]
len(embedding)
```

It's recommended to use the 'tenacity' package or another exponential backoff implementation to better manage API rate limits, as hitting the API too much too fast can trigger rate limits. Using the following function ensures you get your embeddings as fast as possible.

```
# Negative example (slow and rate-limited)
import openai

num_embeddings = 10000 # Some large number
for i in range(num_embeddings):
    embedding = openai.Embedding.create(
        input="Your text goes here", model="text-embedding-3-small"
    )["data"][0]["embedding"]
    print(len(embedding))


# Best practice
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-3-small") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

embedding = get_embedding("Your text goes here", model="text-embedding-3-small")
print(len(embedding))
```


# 2D Visualization
We will use t-SNE to reduce the dimensionality of the embeddings from 1536 to 2. Once the embeddings are reduced to two dimensions, we can plot them in a 2D scatter plot.

## Dimensionality Reduction
We reduce the dimensionality to 2 dimensions using t-SNE decomposition.
```
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from ast import literal_eval

# Load the embeddings
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)

# Convert to a list of lists of floats
matrix = np.array(df.embedding.apply(literal_eval).to_list())

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)
vis_dims.shape
```

## Plotting
```
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
x = [x for x,y in vis_dims]
y = [y for x,y in vis_dims]
color_indices = df.Score.values - 1

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
for score in [0,1,2,3,4]:
    avg_x = np.array(x)[df.Score-1==score].mean()
    avg_y = np.array(y)[df.Score-1==score].mean()
    color = colors[score]
    plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)

plt.title("Amazon ratings visualized in language using t-SNE")
```

# 3D Visualization
```
# Load the dataset and query embeddings
import pandas as pd
samples = pd.read_json("data/dbpedia_samples.jsonl", lines=True)
categories = sorted(samples["category"].unique())
print("Categories of DBpedia samples:", samples["category"].value_counts())

from utils.embeddings_utils import get_embeddings
# NOTE: The following code will send a query of batch size 200 to /embeddings
matrix = get_embeddings(samples["text"].to_list(), model="text-embedding-3-small")

# Reduce the embedding dimensionality
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(matrix)
samples["embed_vis"] = vis_dims.tolist()

# Plot the embeddings of lower dimensionality
%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection='3d')
cmap = plt.get_cmap("tab20")

# Plot each sample category individually such that we can set label name.
for i, cat in enumerate(categories):
    sub_matrix = np.array(samples[samples["category"] == cat]["embed_vis"].to_list())
    x=sub_matrix[:, 0]
    y=sub_matrix[:, 1]
    z=sub_matrix[:, 2]
    colors = [cmap(i/len(categories))] * len(sub_matrix)
    ax.scatter(x, y, zs=z, zdir='z', c=colors, label=cat)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(bbox_to_anchor=(1.1, 1))
```


Source of this post: https://github.com/openai/openai-cookbook