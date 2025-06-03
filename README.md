# bayesian-vs-ridge-regression
Comparison of Bayesian Ridge Regression and standard Ridge Regression for continuous target prediction using real-valued features.


##  Bayesian vs Ridge Regression

In this mini project, I explored how **Bayesian Ridge Regression** works compared to standard Ridge Regression.

The dataset had 4 numerical columns — I used the first 3 as input features and the last one as the target. After normalizing the data, I trained both models using scikit-learn.

###  What I did:
- Standardized the input features using `StandardScaler`
- Trained a **regular Ridge Regression** as baseline
- Trained a **Bayesian Ridge Regression** to model uncertainty
- Compared their predictions and learned weights

###  Why Bayesian?
The Bayesian model doesn’t just give predictions — it tells you **how confident it is** about each one. That’s a big deal when working with noisy or sensitive data.

###  Conclusion
Both models gave reasonable predictions, but Bayesian Ridge added extra value by capturing uncertainty in model parameters. It’s a great step toward more interpretable machine learning.

---

## Tools used: `scikit-learn`, `numpy`, `matplotlib`, `pymc`, `StandardScaler`
