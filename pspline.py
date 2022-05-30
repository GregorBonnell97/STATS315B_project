import numpy as np

class PSplineRegressor(object):
    """
    Implementation of a Spline
    """
    
    def __init__(self, Lambda=0, deg=3):
        """
        Lambda: penalization parameter
        deg: degree of the polynomial function 
        """
        self.Lambda=Lambda
        self.deg=deg
    
    def B(self, x, k, i, t):
        """
        to be commented
        """
        if k == 0:
            return np.where(np.abs(x-0.5*(t[i]+t[i+1]))<0.5*(t[i+1]-t[i]),1,0)
        if t[i+k] == t[i]:
            c1 = np.zeros(x.size)
        else:
            c1 = (x - t[i])/(t[i+k] - t[i]) * self.B(x, k-1, i, t)
        if t[i+k+1] == t[i+1]:
            c2 = np.zeros(x.size)
        else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * self.B(x, k-1, i+1, t)
        return c1 + c2
   
    def D(self,p,N):
        """
        returns a matrix of size (N-p,N)
        1 on the diagonal 
        -2 on the upper diagonal
        1 on the upper upper diagonal
        """
        output=np.zeros((N-p,N))
        for i in range(N-p):
            output[i,i+2]=1
            output[i,i]=1
            output[i,i+1]=-2
        return output
    
    def BS(self,X,t,nb_nots=20):
        """
        hello
        """
        N=nb_nots-1-self.deg
        Bs=np.zeros((X.size,N))
        for j in range(N):
            Bs[:,j]=self.B(X,self.deg,j,t)
        return Bs
    
    def fit(self, X, Y, nb_nots=20, p=2):
        t=np.linspace(min(X), max(X), nb_nots)
        Bs=self.BS(X,t,nb_nots)
        N=nb_nots-1-self.deg
        d=self.D(p,N)
        inv_Bs=np.linalg.inv(Bs.T.dot(Bs)+self.Lambda*d.T.dot(d))
        a=inv_Bs.dot(Bs.T.dot(Y))
        return a,Bs.dot(a),t
    
    def predict(self,X_test,t,a,nb_nots=20):
        Bs=self.BS(X_test,t,nb_nots)
        return Bs.dot(a)
        