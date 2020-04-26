## MatrixFactorization

Base file for MatrixFactorization repo.

In this repo, I tried implementing the gradient descent-based matrix factorization, in order to compute the SVD-like decomposition A = U * V^T
for a user-item matrix. 
These two matrices U (which is supposed to incorporate the singular values) and V can then be used to suggest to a user an item that he hasn't still seen yet, as in a recommender system, with a dot product: 
* ![$r_{ij} = u_i \cdot i_j^T$](https://render.githubusercontent.com/render/math?math=%24r_%7Bij%7D%20%3D%20u_i%20%5Ccdot%20i_j%5ET%24)

This falls inside the category of Collaborative Filtering methods, in fact it can't be used for a new user, it suffers of the cold start problem.


I based the implementation of the Charu Aggarwal's book on Recommender Systems by Springer.
