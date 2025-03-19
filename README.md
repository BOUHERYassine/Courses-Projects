Courses-Projects, Internships

Here is a list of my courses over the last 3 years. In some courses, there are projects are mentioned in $${\color{red}red}$$

Next, I present my internship topics in detail.

## MVA

### Large Language Models

### Genrative AI for images

### Graphs in ML

### Data generation by transport and denoising

### Convex Optimisatoin

### Reinforcment Learnine

### Stopping times and online algorithms

### Deep learning

### Statistical Learning with extreme values

### Time Series
- Pattern recognition and detection
- Feature extraction and selection
- Representation learning
- Data enhancement and preproceesing
- Anomaly detection
- Multivariate time series
- Presentation of the following article $${\color{red}\text{Soft-DTW: a Differentiable Loss Function for Time-Series}}$$ [ARTICLE](https://arxiv.org/abs/1703.01541)
    - DTW loss allows to compare time series that are not of the same length, but it is not differentiable
    - Soft-DTW is a variant of DTW that is differentiable. It can be useful for methods that use gradients
    - Time series averaging using soft-DTW (on some time series from the M4 dataset)
    - Time series clustering using soft-DTW
    - Study of the influence of the $\gamma$ parameter of the soft-DTW loss.

## Ecole Polytechnique

### Collaborative machine learning and decentralized learning
- Fully research-axed course
- Study of decentralized learning, gossip matrix
- Presentation of the following article : $${\color{red}\text{Accelerated Decentralized Optimization with Local Updates for Smooth and Strongly Convex Objectives}}$$ [ARTICLE](https://arxiv.org/abs/1810.02660)
    - Presentation of the article
    - We have shown that this algorithm cannot be Chebyshev accelerated : That means that we cannot trade between computation time (compute the gradient) and communication time (propagate the gradient to its neighbors)
    - We have shown that, at the moment the article was published, in general, asynchronous algorithms cannot compete against synchronous one
    - Here, asynchonous means that the nodes are not all updated simultaneously.

### Statistical Learning
- Rademacher complexity, VC dimension, PAC learning
- Bandit algorithms
- High dimensional statistics

### Random and statistical process modelling
-Overview of a lot of statistical methods
- Central Limit theorem adapted for medians
- Stastics of extreme values
- AR and MA processes, and all of its variants (ARIMA, ARFIMA, SARIMAX, SARIMA etc..)
- $${\color{red}\text{Study of statistical methods}}$$
    - Extreme values on CAC40 data with Picklands and Hill estimators (to caracterize the type of the extreme)
    - GARCH and ARCH processes on CAC40 data to study the variance/volatility
    - I do not develop the results obtained because they were deceiving, the idea of the project was rather to play with the concepts

### Monte Carlo methods
- Review of all advanced Monte Carlo's methods
- MCMC (Markov chains Monte Carlo), Metropolis Hastings algorithm
- Variance reduction and stratification
- Rejection sampling
- Quasi-Monte Carlo

### Foundation of Machine Learning
- In-depth study of theoretical guarantees of Machine Leaning (wth Hoeffding's and McDiarmid's inequalities)
- Overview of most of machine learning algorithms, Ensemble methods, SVM, Clustering, Gradient descent, Deep learning
### Stochastic calculus
- Brownian motion
- Ito's formula
- Stochastic integrals
- Stochastic differential equations (SDE)
- Black-Scholes model

### Research course (PCA guarantees)
- Fully research-axed course
- $${\color{red}\text{Theoretical guarantees of PCA estimators}}$$
  
    - In machine learning, we use a lot the PCA methods, however, few of us know the link between the estimated covariance matrix, and the true covariance matrix, and we studied it.
    - In our case, we considered an distribution of features X (can be an "infinite matrix") evolving on a Hilbert space $\mathbb{H}$, its covariance matrix is $\Sigma = \mathbb{E}(X \otimes X)$
    - We sample $X_1$, ..., $X_n$ from X, we obtain empirical estimators of the covariance matrix and the eingenvalues : $\hat{\Sigma_n} = \sum_{i=1}^n X_i \otimes X_i = \sum_{j\ge 1} \hat{\lambda_j} \hat{\theta_j} \otimes \hat{\theta_j}$
    - We want to know how close, in terms of n, are the empirical estimators close to the real values.
    - We reproduce the Proof of **Vershynin** (2011). If the $\mathbb{H} = \mathbb{R}^p$, and if X is sub-gaussian ans isotropic ($\Sigma = I$), we have $$\lVert \hat{\Sigma_n} - \Sigma \lVert < C (\sqrt{\frac{p}{n}} \vee \frac{p}{n} \vee \sqrt{\frac{t}{n}} \vee \frac{t}{n})$$   , with a probability of at least $1-e^{-t}$
    - Now that the covariance matrix convergence is bounded, we can bound the convergence of eigenvalues with Lidskii's inequality : $$\max_{j \ge 1} |\hat{\lambda_j} - \lambda_j| < \lVert \hat{\Sigma} - \Sigma \rVert$$
    - An application to this study was to do a simple spectral clustering setup with gaussian mixtures

### Random numerical simulation of rare events
- Monte Carlo methods : Naive, Importance Sampling and more
- $${\color{red}\text{Prediction of a Premier League season using previous seasons and Monte Carlo methods}}$$
  
    - In 2015-2016, Leicester won their Premier League despite being far from favourites. The bookmakers haven't even given them a 0.5% chance.
    - In our modelling of Premier Ligue, we used previous seasons to modelise the strength of a team i $V_i$. This is the **Bradley-Terry** model. In this case, when the team i faces the team j, their probability to win is $\frac{V_i}{V_i+V_j}$ (note that ties are not allowed).
    - With that, we case modelise Premier League, but Leicester has a winrate of 0% when we use Naive Monte Carlo
    - We use importance sampling to modify the strenghts of the teams (the more we modify, the bigger the corrective term will be)
    - With that, we predised the chances for Leicester to Win, and it was a frequency of the order of $10^{-4}$

### Statistics
- Hypothesis test
- p-value
- Statistical estimators including bias, method of moments, and EMV (Method of Maximum Likelihood Estimation)
- Introduction to Machine learning with theoretical guarantees

### Differential calculus and holomorphic functions
- Introduction to holomophic function
- Residue theorem
- Complex logarithm

### Modelling random phenomena
- Introduction to martingals, central limit theorem
- Convergence : In probability, almost surely, in $L^1$, in $L^2$, in distribution


# Intenships

## Ekimetrics
