class matrixFactorization: 
    def __init__(self, ratings, k, lr, lam):
    
        
        self.ratings = ratings 
        m,n = self.ratings.shape
        
        self.k = k
        self.lr = lr 
        self.lam = lam
        self.U = np.random.rand(m,k)
        self.V = np.random.rand(n,k)
        self.isTrained = False

    def get_ratings(self,idxs):
        
        return self.ratings[idxs]
    
    def get_diff_in_mean(self):
        
        try:
            if self.isTrained:
                return self.diff_in_mean
            else:
                raise Exception("Not trained yet.")
        except Exception as e:
            print(e)
    
    def fit(self, n_epochs):
    
        self.diff_in_mean = []
        indices_list = (list(zip(*(np.where(np.isfinite(self.ratings))))))    
        print("Starting to train.")
        for k in range(n_epochs):
            random.shuffle(indices_list)
            for idx in indices_list:
                i, j = idx
                r_ij = self.get_ratings(idx)
                e_ij = r_ij - np.dot(self.U[i,:],self.V[j,:])
        
                u_ij = self.U[i,:] + self.lr*(e_ij*self.V[j,:]-self.lam*self.U[i,:])
                v_ij = self.V[j,:] + self.lr*(e_ij*self.U[i,:]-self.lam*self.V[j,:])

                self.U[i,:] = u_ij
                self.V[j,:] = v_ij
            current_diff = np.abs(np.nanmean(self.ratings)-np.mean(np.dot(self.U,self.V.T)))
            print(f"Current diff at epoch {k+1} is {current_diff}")
            self.diff_in_mean.append(current_diff)
            
        print("Training completed.")    
        self.isTrained = True
