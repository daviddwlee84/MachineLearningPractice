# Information Retrieval - Topic Modelling Relevent Part

* [Wiki - Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval)
* [Gensim - Introduction (Concepts)](https://radimrehurek.com/gensim/intro.html)
* [Gensim - Tutorial (example about TF-IDF)](https://radimrehurek.com/gensim/tutorial.html)

## Overview

### [Information Retrieval Model](#IR-Models)

![Wiki IR Model](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Information-Retrieval-Models.png/800px-Information-Retrieval-Models.png)

#### Classical Information Retrieval Model

(usually work with unstructured text document)

**Set-theoretic Models**:

Set-theoretic models represent documents as sets of words or phrases. Similarities are usually derived from set-theoretic operations on those sets.

* [Boolean Model](#Boolean-Model) (布爾)
* Extended Boolean Model (擴展布爾)
* Fuzzy Retrieval (模糊)

**Algebraic Models**:

Algebraic models represent documents and queries usually as vectors, matrices, or tuples. The similarity of the query vector and document vector is represented as a scalar value.

* [Vector Space Model](#Vector-Space-Model) (向量空間)
* Generalized Vector Space Model (廣義向量)
* (Enhanced) Topic-based Vector Space Model
* Extended Boolean Model <-
* [Latent Semantic Indexing (LSI) aka. Latent Semantic Analysis (LSA)](#Latent-Semantic-Analysis-(Latent-Semantic-Indexing)) (潛在語意分析)

**Probabilistic Models**:

Probabilistic models treat the process of document retrieval as a probabilistic inference. Similarities are computed as probabilities that a document is relevant for a given query. Probabilistic theorems like the Bayes' theorem are often used in these models.

* Binary Independence Model
* Probabilistic Relevance Model (on which is based the okapi (BM25) relevance function)
* Uncertain Inference
* Language Models
* Divergence-from-randomness Model
* Latent Dirichlet Allocation

#### Half-structured Document Information Retrieval Model

* XML

#### Link-based Information Retrieval Model

* Page Rank
* Hubs & Authorities

#### Multimedia Information Retrieval Model

* Image
* Audio and Music
* Video

### [How to represent text](#Text-Representation)

* [Bag-of-words approach](#Bag-of-words-Approach)
* [Vector representation](#Vector-Representation)

### [Term Weights](#Term-Weight)

* [TF-IDF](#TF-IDF)

### [Performance and Correctness Measures](#Evaluation-Measures)

* Precision
* Recall
* Fall-out
* F-score / F-measure
* Average precision
* R-Precision
* Mean average precision
* Discounted cumulative gain

## Text Representation

### Bag-of-words Approach

[Wiki - Bag-of-words model](https://en.wikipedia.org/wiki/Bag-of-words_model)

* Treat all the words in a document as *index terms* for that document
* Assign a *weight* to each term based on its *importance*
* *Disregard order, structure, meaning, etc.* of the words

Conclusion

* This approach think IR is all (and only) about mathcing words in documents with words in queries (which is not true)
* But it works pretty well

#### Vector Representation

- [Wiki - Word2vec](https://en.wikipedia.org/wiki/Word2vec)
- [Tensorflow - Vector Representations of Words](https://www.tensorflow.org/tutorials/representation/word2vec)

* "Bags of words" can be represented as vectors
    * For computational efficiency, easy of manipulation
    * Geometric metaphor: "arrows"
* A vector is a set of values recorded in any consistent order

## Term Weight

**Term**:

The definition of **term** depends on the application. (Typically terms are single words, keywords, or longer phrases.)

* Each dimension corresponds to a separate term.
* If a term occurs in the document, its value in the vector is non-zero.
* If words are chosen to be the terms, the dimensionality of the vector is the number of words in the vocabulary (the number of distinct words occurring in the corpus).

**Example**:

document|text                       |terms
--------|---------------------------|-------------------
doc1    |ant ant bee                |ant bee
doc2    |dog bee dog hog dog ant dog|ant bee dog hog
doc3    |cat gnu dog eel fox        |cat dog eel fox gnu

query|content
-----|-------
q    |ant dog

**Term incidence matrix (in Term vector space (No weighting))**:

doc |ant|bee|cat|dog|eel|fox|gnu|hog
----|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
doc1| 1 | 1 |   |   |   |   |   |
doc2| 1 | 1 |   | 1 |   |   |   | 1
doc3|   |   | 1 | 1 | 1 | 1 | 1 |

**Unnormalized Form of Term Frequency (TF) (weighting)**:

(Weight of term td) = frequency that term j occurs in document i

doc |ant|bee|cat|dog|eel|fox|gnu|hog|length
----|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|-----:
doc1| 2 | 1 |   |   |   |   |   |   |$\sqrt{5}$
doc2| 1 | 1 |   | 4 |   |   |   | 1 |$\sqrt{19}$
doc3|   |   | 1 | 1 | 1 | 1 | 1 |   |$\sqrt{5}$

Calculate Ranking:

Similarity of documents

-   |doc1|doc2|doc3
----|:--:|:--:|:--:
doc1| 1  |0.31| 0
doc2|0.31| 1  |0.41
doc3| 0  |0.41| 1

Similarity of query to documents

-   |doc1|doc2|doc3
----|:--:|:--:|:--:
q   |0.63|0.81|0.32

**Methods for Selecting Weights**:

* Empirical

    Test a large number of possible weighting schemes with actual data

* Model based

    Develoop a mathematical model of word distribution and derive weighting scheme theoretically. (e.g Probabilistic model)

* Intuition
    * Terms that appear often in a document should get higher weights
        * The more often a document contains the term "dog", the more likely that the document is "about" dogs
    * Terms that appear in many documents should get low weights
        * Words like "the", "a", "of" appear in (nearly) all documents

* Weighting scheme
    * bianry (i.e. term incidence matrix)
    * row count (i.e. unnormalized form of term frequency)
    * term frequency (the following thing...)
    * log normalization
    * double normalization 0.5
    * double normalization K
    * ...

### TF-IDF

* TF stands for Term Frequency
* IDF stands for Inverse Document Frequency

#### Term Frequency

($t$ stands for term; $d$ stands for document)

$$
tf(t, d) = f_{t, d}
$$

**Concept**:

* A term that appears many times within a document is likely to be more important than a term that appears only once

**Normalization** (for Free-text document):

Key: Length of document

* a term frequency in different length of document may have different importance

Maximum Normalization:

$$
tf(t, d)  = \frac{f_{t, d}}{\text{maximum frequency of any term in document}_d}= \frac{f_{t, d}}{\max_{t' \in d}(f_{t, d})} = \frac{f_{t, d}}{m_d}
$$

Augmented Maximum Normalization: (for Structured Text)

(Salton and Buckley recommend K = 0.5)

$$
tf(t, d) = K + (1-K) \times \frac{f_{t, d}}{m_d}, K \in [0, 1]
$$

Cosine Normalization:

$$
tf(t, d) = \frac{f_{t, d}}{\sqrt{\sum_{d}(f_{t, d})^2}}
$$

#### Inverse Document Frequency

**Concept**:

* A term that occurs in only a few documents is likely to be a better discriminator that a term that appears in most or all documents

**A Simple Method**: use document frequency

$$
idf(t, D) = \frac{1}{df} = \frac{1}{\frac{\text{number of document has term}_j}{\text{number of total documents}}} = \frac{|D|}{n_t}
$$

(the simple method over-emphasizes small differences) => use logarithm

**A Standard Method**: use log form

$$
idf(t, D) = \log(\frac{N}{n_t}) + 1
$$

#### Full weighting of TF-IDF

The weight assigned to term $t$ in document $d$:

$$
\mathit{tf.idf}(t, d, D) = tf(t, d) \times idf(t, D)
$$

Example:

$$
\mathit{tf.idf}(t, d, D) = \underbrace{\frac{f_{t, d}}{m_t}}_{tf(t, d)} \times \underbrace{(\log(\frac{N}{n_t}) + 1)}_{idf(t, D)}
$$

## IR Models

### Boolean Model

* [Wiki - Boolean model of information retrieval](https://en.wikipedia.org/wiki/Boolean_model_of_information_retrieval)
* [Wiki - DNF (Disjunctive normal form)](https://en.wikipedia.org/wiki/Disjunctive_normal_form)

**Brief description**:

* Based on notion of sets
* Documents are retrieved *only if* they satisfy boolean conditions specified in the query
* *No ranking* on retrieved documents
* *Exact match*

**Similarity of documnet and query**:

$$
\mathit{Sim}(\mathbf{doc}_i, \mathbf{query})
\begin{cases}
1, \exists c(\mathbf{query})|c(\mathbf{query}) = c(\mathbf{doc}_i) \\
0, \text{otherwise}
\end{cases}
$$

#### Boolean Retreival

(Boolean operators *approximate* natural language)

* AND: discover relationships between concepts
* OR: discover alternate terminology
* NOT: discover alternate meaning

The Perfect Query Paradox

* Every information need has a perfect set of documents
    * If not, there would be no sense doing retrieval
* Every document set has a perfect query
    * AND every word in a document to get a query for it
    * Repeat for each document in the set
    * OR every document query to get the set query
> but can users realistically be expected to formulate this perfect query?
> => perfact query formulation is hard

Why Boolean Retrieval Fails

* Natural language is way more complex
* AND "discovers" nonexistent relationships
    * Terms in different sentences, paragraphs, ...
* Guessing terminology for OR is hard
    * e.g. good, nice, excellent, outstanding, awesome, ...
* Guessing terms to exclude is even harder
    * e.g. democratic party, party to a lawsuit, ...

Pros and Cons

* Strengths
    * Precise
        * if you know the right strategies
        * if you have an idea of what you're looking for
    * Efficient for the computer
* Weaknesses
    * User must learn boolean logic
    * Boolean logic insufficient to capture the richness of language
    * No control over size of result set (either too many documents or none)
    * When do you stop reading? All documents in the result set are considered "equally good"
    * What about partial matches? Documents that "don't quite match" the query may be useful also

#### Ranked Retrieval

Arranging documents by relevance

* Closer to how humans think
    * some documents are "better" than others
* Closer to user behavior
    * users can decide when to stop reading
* Best (partial) match
    * documents need not have all query terms

Similarity-Based Queries

* Replace *relevance* with *similarity*
    * rank documents by their similarity with the query
* Treat the query as if it were a document
    * Create a query bag-of-words
    * Find its similarity to each document
    * Rank order the document by similarity

### Vector Space Model

* [Wiki - Vector space model](https://en.wikipedia.org/wiki/Vector_space_model)

**Brief description**:

* Based on geometry, the notion of *vectors in high dimensional space*
* Documents are *ranked* based on their similarity to the query (ranked retrieval)
* *Best / partial match*

> Postulate: Documents that are "close together" in vector space "talk about" the same things

Therefore, retrieve documents based on how close the document is to the query (i.e. similarity ~ "closeness")

**Similarity of document and query**:

* [Wiki - Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

$$
\mathit{Sim}(\mathbf{doc}_i, \mathbf{query}) = \cos{\theta} = \frac{\mathbf{doc}_i \cdot \mathbf{query}}{\left\| \mathbf{doc}_i \right\| \left \| \mathbf{query} \right\|}
$$

### Extended Boolean Model

* [Wiki - Extended Boolean model](https://en.wikipedia.org/wiki/Extended_Boolean_model)

### Latent Semantic Analysis (Latent Semantic Indexing)

* [Wiki - Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

Dimensionality reduction using truncated SVD (aka LSA)

This transformer performs linear dimensionality reduction by means of truncated singular value decomposition (SVD). Contrary to PCA, this estimator does not center the data before computing the singular value decomposition. This means it can work with scipy.sparse matrices efficiently.

In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA).

This estimator supports two algorithms: a fast randomized SVD solver, and a “naive” algorithm that uses ARPACK as an eigensolver on (X * X.T) or (X.T * X), whichever is more efficient.

## Evaluation Measures

## Reference

### Book

* **Modern Information Retrieval - The Concepts and Technology behind Search**
    * Ch3 Modelling
        * Ch3.2.2 Boolean Model
        * Ch3.2.6 Vector Space Model
            * Ch3.2.5 Document Length Normalization
    * Ch3.2.3 Term weight
        * Ch3.2.4 TF-IDF
