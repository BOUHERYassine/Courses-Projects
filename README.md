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

## Ecole Polytechnique

### Collaborative machine learning and decentralized learning

### Statistical Learning

### Random and statistical process modelling

### Monte Carlo methods

### Foundation of Machine Learning

### Stochastic calculus

### Research course (PCA guarantees)
- Fully research-axed course
- $${\color{red}\text{Theoretical guarantees of PCA estimators}}$$
    - In machine learning, we use a lot the PCA methods, however, few of us know the link between the estimated covariance matrix, and the true covariance matrix, and we studied it.
    - In our case, we considered an distribution of features X (can be an "infinite matrix") evolving on a Hilbert space $\mathbb{H}$, its covariance matrix is $\Sigma = \mathbb{E}(X \otimes X)$
    - We sample $X_1$, ..., $X_n$ from X, we obtain empirical estimators of the covariance matrix and the eingenvalues : $\hat{\Sigma_n} = \sum_{i=1}^n X_i \otimes X_i = \sum_{j\ge 1} \hat{\lambda_j} \hat{\theta_j} \otimes \hat{\theta_j}$
    - We want to know how close, in terms of n, are the empirical estimators close to the real values.
    - We reproduce the Proof of **Vershynin** (2011). If the $\mathbb{H} = \mathbb{R}^p$, and if X is sub-gaussian ans isotropic ($\Sigma = I$), we have $$\lVert \hat{\Sigma_n} - \Sigma \lVert < C (\sqrt{\frac{p}{n}} \vee \frac{p}{n} \vee \sqrt{\frac{t}{n}} \vee \frac{t}{n})$$   , with a probability of at least $1-e^{-t}$
    - Now that the covariance matrix convergence is bounded, we can bound the convergence of eigenvalues with Lidskii's inequality : $$\max_{j \ge 1} |\hat{\lambda_j} - \lambda_j| < \lVert \hat{\Sigma} - \Sigma \rVert$$
    - We also prove that the empirical projectors converge $$\mathbb{E}(\| \hat{P_1} - P_1 \|_2^2) = \frac{1}{n}\sum_{j\geq2}\frac{\lambda_1 \lambda_j}{(\lambda_1-\lambda_j)^2} (1 + o(1)) \lesssim \frac{\lambda_1 \lambda_2}{g_1^2} \frac{r(\Sigma)}{n}$$

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
