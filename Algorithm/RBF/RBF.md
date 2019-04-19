# Radial Basis Function

> RBF (Network) has some relationhsip with SVM (kernel), kNN, k-Means, Neural Network

## RBF Basis

### The Local Representation

In multilayer perceptron, the input is encoded by the simultaneous activation of many hidden units. This is called a **Distributed Representation**.

But in RBF, for a given input, only one or a few units are active. This is called a **Local Representation**.

**Receptive Field**: The part of the input space where a unit has nonzero response. (Like in SVM, only Support Vector (some of the decisive input data) will participate in the decision.)

### Regularization

> smaller $M$ and larger $\lambda$

### Choosing Prototype by [k-Means](../KMeans/KMeans.md) Clustering

> using unsupervised learning (k-Means) to assist *feature transform* (like autoencoder)

## RBF Application

### RBF Kernel in Gaussion SVM

Gaussian SVM: find $\alpha_n$ to combine Gaussians centered at $x_n$ => achieve large margin in *infinite-dimensional space*

$$
g_{\text{SVM}}(\mathbf{x}) = \operatorname{sign}(\sum_{\text{support vector}}\alpha_n y_n \exp(-\gamma||\mathbf{x} - x_n||^2) + b)
$$

The Gaussian kernel is also called RBF kernel

* Radial: only depends on distance between $x$ and 'center' $x_n$
* Basis Function: to be 'combined'

let $g_n(\mathbf{x}) = y_n\exp(-\gamma||\mathbf{x}-x_n||^2)$ then

$$
g_{\text{SVM}}(\mathbf{x}) = \operatorname{sign}(\sum_{\text{support vector}}\alpha_n g_n(\mathbf{x}) + b)
$$

> Linear *aggregation* of selected *radial* hypothesis

### RBF and Similarity/Distance

> **kernel**: similarity via inner product in $\mathcal{Z}$-space
>
> RBF: similarity via $\mathcal{X}$-space distance (often *monotonically non-increasing* to distance)

### RBF Network

> Linear *aggregation* of *radial* hypotheses

![Wiki - RBF Network](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Radial_funktion_network.svg/465px-Radial_funktion_network.svg.png)

> It's a simple (old-fashioned) model

The difference between normal neural network

| -            | Normal Neural Network               | RBF Network                                       |
| ------------ | ----------------------------------- | ------------------------------------------------- |
| hidden layer | inner product + activation function | distance of centers + Gaussian                    |
| output layer | linear aggregation                  | linear aggregation                                |
| layer number | may have multiple layers            | normally no more than one layer of Gaussian units |

> RBF Network is historically a type of neural netowrk

$$
h(\mathbf{x}) = \sum_{m = 1}^M \beta_m \operatorname{RBF}(\mathbf{x}, \mu_m) + b
$$

The Learning Variables:

* **centers** $\mu_m$
* (signed) **votes** $\beta_m$ (the linear aggregation weight)

When $M = N$ we called it Full RBF Network (lazy way to decide $\mu_m$) => aggregate *each sample*'s opinion subject to *similarity*

> Full RBF Network has some relation with [kNN](../kNN/kNN.md)

#### Comparision of RBF Network and RBF in Gaussion SVM and Similarity

Formula Corresponding

* RBF vs. Gaussian
* Output activation function vs. Sign function (for binary classification)
* $M$ vs. number of support vector
* $\mu_m$ vs. SVM support vector $x_m$
* $\beta_m$ vs. $\alpha_my_m$ from SVM Dual

> RBF Network: distance *similarity-to-centers* as *feature transform*

Parameters

* $M$: prototypes (centroid)
* RBF: such as $\gamma$ of Gaussian

### Interpolation by Full RBF Network

Non-regularized Full RBF Network

> called **exact interpolation** for function approximation.
> this is bad in machine learning => overfitting

Regularized Full RBF Network

... (around 15 mins in Hsuan-Tien Lin Machine Learning Techniques RBF Network Learning)

## RBF Derivation

> Basically the Exercises 3 and 4 in Introduction to Machine Learning 3rd Ch12.11

Figure 12.8

![Figure 12.8](https://images.slideplayer.com/16/4882459/slides/slide_9.jpg)

The RBF network where $p_h$ are hidden units using the bell-shaped activation funciton.
$\mathbf{m}_h$, $s_h$ are the first-layer parameters, and $w_i$ are the second-layer weights.

### Derive the update equations for the RBF netowrk for classification

> Equation 12.20 - the softmax
>
> $$
> y_{i}^{t}=\frac{\exp \left[\sum_{h} w_{i h} p_{h}^{t}+w_{i 0}\right]}{\sum_{k} \exp \left[\sum_{h} w_{k h} p_{h}^{t}+w_{k 0}\right]}
> $$
>
> Equation 12.21 - the cross-entropy error
>
> $$
> E\left(\left\{\mathbf{m}_{h}, s_{h}, w_{i h}\right\}_{i, h} | X\right)=-\sum_{t} \sum_{i} r_{i}^{t} \log y_{i}^{t}
> $$
>
> Because of the use of cross-entropy and softmax, the update equations will be the same with equations 12.17, 12.18, and 12.19 (see
equation 10.33 for a similar derivation).

Equation 12.17 - the update rule for the second layer weights

$$
\Delta w_{i h}=\eta \sum_{t}\left(r_{i}^{t}-y_{i}^{t}\right) p_{h}^{t}
$$

Equation 12.18, 12.19 - the update equations for the **centers** and **spreads** by backpropagation (chain rule)

$$
\Delta m_{h j}=\eta \sum_{t}\left[\sum_{i}\left(r_{i}^{t}-y_{i}^{t}\right) w_{i h}\right] p_{h}^{t} \frac{\left(x_{j}^{t}-m_{h j}\right)}{s_{h}^{2}}
$$

$$
\Delta s_{h}=\eta \sum_{t}\left[\sum_{i}\left(r_{i}^{t}-y_{i}^{t}\right) w_{i h}\right] p_{h}^{t} \frac{\left\|x^{t}-m_{h}\right\|^{2}}{s_{h}^{3}}
$$

Equation 10.33

$$
\begin{aligned} \Delta w_{j} &=\eta \sum_{t} \sum_{i} \frac{r_{i}^{t}}{y_{i}^{t}} y_{i}^{t}\left(\delta_{i j}-y_{j}^{t}\right) x^{t} \\ &=\eta \sum_{t} \sum_{i} r_{i}^{t}\left(\delta_{i j}-y_{j}^{t}\right) x^{t} \\ &=\eta \sum_{t}\left[\sum_{i} r_{i}^{t} \delta_{i j}-y_{j}^{t} \sum_{i} r_{i}^{t}\right] x^{t} \\ &=\eta \sum_{t}\left(r_{j}^{t}-y_{j}^{t}\right) x^{t} \\ \Delta w_{j 0} &=\eta \sum_{t}\left(r_{j}^{t}-y_{j}^{t}\right) \end{aligned}
$$

### Show how the system given in equation 12.22 can be trained

> Equation 12.22
>
> $$
> y^{t}=\underbrace{\sum_{h=1}^{H} w_{h} p_{h}^{t}}_{\textit{exceptions}}+\underbrace{\mathbf{v}^{T} \mathbf{x}^{t}+v_{0}}_{rule}
> $$
>
> There are two sets of parameters: $\mathbf{v}$, $v_0$ of the default model and
> $w_h$, $\mathbf{m}_h$, $s_h$ of the exceptions. Using gradient-descent and starting from
> random values, we can update both iteratively. We update $\mathbf{v}$, $v_0$ as if
> we are training a linear model and $w_h$, $\mathbf{m}_h$, $s_h$ as if we are training a
> RBF network.
>
> Another possibility is to separate their training: First, we train the
> linear default model and then once it converges, we freeze its weights
> and calculate the residuals, that is, differences not explained by the
> default. We train the RBF on these residuals, so that the RBF learns
> the exceptions, that is, instances not covered by the default “rule.”

## Resources

### Book

* Intorduction to Machine Learning 3rd Ch12 Local Models
  * Ch12.3 Radial Basis Functions
  * Ch12.11 Exercises
    * 3
    * 4

### Wikipedia

* [Radial basis function](https://en.wikipedia.org/wiki/Radial_basis_function) - 徑向基函數
* [Radial basis function kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)
* [Radial basis function network](https://en.wikipedia.org/wiki/Radial_basis_function_network)
* [Radial basis function interpolation](https://en.wikipedia.org/wiki/Radial_basis_function_interpolation)

### Tutorial

* Hsuan-Tien Lin Machine Learning Techniques (機器學習技法) - Radial Basis Function Network
  * [X] [RBF Network Hypothesis](https://youtu.be/7lHhnpdPVr0)
  * [X] [RBF Network Learning](https://youtu.be/dEYdx2rS66c)
  * [X] [k-Means Algorithm](https://youtu.be/ker9RF2TDUU)
  * [X] [k-Means and RBFNet in Action](https://youtu.be/D5elADTz1vk)
  * Notes
    * [林軒田教授機器學習技法 Machine Learning Techniques 第 14 講學習筆記](https://blog.fukuball.com/lin-xuan-tian-jiao-shou-ji-qi-xue-xi-ji-fa-machine-learning-techniques-di-14-jiang-xue-xi-bi-ji/)
    * [機器學習技法 學習筆記 (7)：Radial Basis Function Network與Matrix Factorization](https://www.ycc.idv.tw/ml-course-techniques_7.html)
    * [[ML] 機器學習技法：第十四講 Radial Basis Function Network](https://zwindr.blogspot.com/2017/08/ml-radial-basis-function-network.html)
* [ ] [CS 156 Lecture 16 - Radial Basis Functions](https://youtu.be/O8CfrnOPtLc)
