# Natural Language Processing Analysis of Philosophical Texts

This project is our third iteration of analyzing philosophical texts using natural language processing. We download texts from Project Gutenberg, find instances where authors reference each other, turn that data into a graph, and then visualize and analyze the results. This data is then used to create PhilBERT, an interactive tool where users can visualize our networks, which you can find it here: https://ogreowl.github.io/PhilBERT/

## Data Pipeline Overview

### Step 1: Downloading + Preparing Data

In `download.py`, we downloaded texts from Project Gutenberg by category. To maintain a focus on philosophy, while also showing how it interacted with other fields, we downloaded:

* 2819 texts classified as 'philosophy'
* 218 texts classified as 'literature'
* 54 texts classified as 'mathematics'
* 157 texts classified as 'physics'
* 215 texts classified as 'politics'
* 220 texts classified as 'science'
* 206 texts classified as 'religion'

We saved information about each text in a csv file, including:
* Title
* File Path
* Author
* Author Birth Year
* Author Death Year

In `processing.ipynb`, we:
1. Combined the csv files
2. Filtered out duplicate texts
3. Removed authors without birth and death years
4. Filtered out texts that added noise (e.g., 'William Wake')
5. Applied special filtering for important authors with common names (e.g., 'Adam Smith')

### Step 2: Fetching References

In `reference_fetcher.py`, we fetched references from each text through:
1. Loading books and authors from the csv file
2. Finding author references and their context in each book
3. Processing texts in batches to manage memory

Additional processing in `processing.ipynb` included:
1. Removing duplicate references
2. Filtering noisy author names
3. Removing authors with missing data

### Step 3: Embedding, Classification, + Matrix Generation

In `embedding.py`, we used DistilBERT to create embeddings for reference contexts. We matched these embeddings to 8 topics:

* Ethics
* Metaphysics
* Epistemology
* Art
* Religion
* Politics
* Science
* Mathematics

We generated 11 different matrices in `processing.ipynb`:

**Main Matrices:**
1. `main.csv`: 163 authors, filtered for relevance and visualization
2. `strong_filter.csv`: filtered version of main.csv removing anachronistic references
3. `expanded.csv`: full list of 1,088 authors

**Topic Matrices:**
* Separate matrices for each of the 8 topics

### Step 4: Analysis

We computed key network metrics including:
1. Incoming and outgoing reference counts
2. Various centrality measures:
   * In-degree
   * Out-degree
   * Betweenness
   * Eigenvector
   * PageRank
3. Network density

### Step 5: Visualization Tool

The final visualization tool is available in our [separate repository](https://github.com/ogreowl/nlp-phil-v3), allowing users to interact with and explore our networks.
