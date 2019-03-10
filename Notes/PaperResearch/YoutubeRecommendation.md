# YouTube Recommendation

## Background

### Personalized Recommendation

In today's information-rich environment, personalized recommendation are a key method for *information retrieval* and *content discovery*.

And it is combined with,

* Pure search (querying)
* Browsing (directed / non-directed)

### Goals of YouTube

There are some scenario

* User watch a single video that they found elsewhere
  * Direct navigation
* User find specific videos around a topic
  * Search
  * Goal-oriented browse
* User just be entertained by content that they find interesting
  * Personalized video recommendation (THIS IS THE MAIN PART!!): dub *unarticulated want*

### Challenges / Difficulty

* Videos which are uploaded by users often have no or very poor metadata
* Videos on YouTube are mostly short form (under 10 mins)
  * => User interactions are relatively short and noisy
    * vs. Netflix: renting a movie (long video)
    * vs. Amazon: purchasing an item are very clear declarations of intent

Three major perspectives

* Scale
* Freshness
* Noise

## Recommendation System Design

Goals: Recent and Fresh (i.e. diverse and relevant to user's recent actions)

> it's important that user understand why a video was recommended to them

* User's personal activity as seed
  * watched videos
  * favorited videos
  * liked videos
* Expanding the set of videos by traversing a *co-visitation based graph of videos*
  * the set of videos is then *ranked* using a variety of signals for *relevance* and *diversity*

System overview

![img from the 2016 paper](https://cdn-images-1.medium.com/max/800/1*fcsodWL98sYqyUhQtJlNPA.png)

Two neural network

* candidate generation (collaborative filtering)
* ranking

### Input of System

* Content data
  * raw video streams
  * video metadata
    * title
    * description
    * ...
* User activity data
  * Explicit
    * rating a video
    * favoriting/liking a video
    * subscribing to an uploader
  * Implicit (datum generated)
    * user watching and interacting with videos
      * e.g. user started to watch a video, user watched a large portion of the video (long watch)

### Calculate Related Videos

> Construction of a *mapping* from a video $v_i$ to a set of similar or related videos $R_i$

The mapping technique: **Association Rule Mining** (**Co-visitation Counts**)

Definition

* Similar videos: a user is likely to watch after having watched the given *seed video* $v$
* Co-visitation count: $c_{ij}$
* Relatedness score of video $v_j$ to base video $v_i$
    $$
    r(v_i, v_j) = \frac{c_{ij}}{f(v_i, v_j)}
    $$
    * $f(v_i, v_j)$ is a normalization function (e.g. $f(v_i, v_j)=c_i \cdot c_j$)
      * it takes the "global popularity" of both the seed video and the candidate video into account

--

> Before here is mostly from the 2010 paper, and after here is mostly from the 2016 paper.

--

### Candidate Generation

#### Recommendation => Extreme Multiclass Classification

> Prediction problem => Classifying a specific video watch $w_t$ at time $t$ among millions of videos $i$ (classes) from a corpus $V$ based on a user $U$ and context $C$

$$
P(w_t = i|U, C) = \frac{e^{v_i u}}{\sum_{j\in V} e^{v_j u}}
$$

* Embedding the *user context pair* and the *candidate video*.
  * It's inspired by continuous bag of words language models
* YouTube use the implicit feetback of watches to train the model. (instead of explicit feedback such as thumbs up/down etc.)

![deep candidate generation model 2016 paper](https://adriancolyer.files.wordpress.com/2016/09/dnn-youtube-fig-3.png?w=600)

### Ranking

#### Feature Representation (Feature Engineering)

* Embedding Categorical Features
* Normalizing Continuous Features

![deep ranking network 2016 paper](https://adriancolyer.files.wordpress.com/2016/09/dnn-youtube-fig-7.png?w=600)

## Resources

### Paper

* [The YouTube video recommendation system](https://dl.acm.org/citation.cfm?id=1864770) (2010)
  * [pdf](https://www.researchgate.net/profile/Sujoy_Gupta2/publication/221140967_The_YouTube_video_recommendation_system/links/53e834410cf21cc29fdc35d2/The-YouTube-video-recommendation-system.pdf)
* [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/citation.cfm?id=2959190) (2016)
  * [pdf](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)

### Article

* [Reddit - Learned about factors in the YouTube algorithm, and how to help your videos trigger them](https://www.reddit.com/r/youtube/comments/5npala/learned_about_factors_in_the_youtube_algorithm/)
* [Medium - How YouTube Recommends Videos](https://towardsdatascience.com/how-youtube-recommends-videos-b6e003a5ab2f)
* [the morning paper - Deep neural networks for YouTube recommendations](https://blog.acolyer.org/2016/09/19/deep-neural-networks-for-youtube-recommendations/)
