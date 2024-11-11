#LAB-1-EXTRA
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.prio_probability = {}
        self.feature_params = {}
        self.feature_unique = []
    def feature_probability(self, X, y):
        '''Unique features'''
        self.feature_unique = np.unique(y)
        '''Probability of 1 or -1'''
        self.prio_probability = {c: np.mean(y == c) for c in self.feature_unique}
        
        '''Mean and Variance'''
        for f in self.feature_unique:
            X_f = X[y == f]
            self.feature_params[f] = (np.mean(X_f, axis=0), np.var(X_f, axis=0))
    
    def predict(self, X):
        prediction = []
        for x in X:
            probs = [] #Place holder to store probabilities
            for f in self.feature_unique:
                prior = np.log(self.prio_probability[f])
                likelihood = np.sum(self.pdf(x, f))
                probs.append(prior + likelihood)
            prediction.append(self.feature_unique[np.argmax(probs)]) #Most likely feature
        return np.array(prediction)
        
    def pdf(self, x, uf):
        '''Log version instead of exp to avoid underflow'''
        mean, var = self.feature_params[uf]
        return -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x - mean) ** 2) / var)

datas = np.array([
    [0.21835, 0.81884, 1],
    [0.14115, 0.83535, 1],
    [0.37022, 0.8111, 1],
    [0.31565, 0.83101, 1],
    [0.36484, 0.8518, 1],
    [0.46111, 0.82518, 1],
    [0.55223, 0.83449, 1],
    [0.16975, 0.84049, 1],
    [0.49187, 0.80889, 1],
    [0.14913, 0.77104, -1],
    [0.18474, 0.6279, -1],
    [0.08838, 0.62068, -1],
    [0.098166, 0.79092, -1]
])

X = datas[:, :-1]
y = datas[:, -1]
test = NaiveBayes()
test.feature_probability(X, y)
predictions = test.predict(X)
is_correct = predictions == y
print("Predictions:", predictions, "Accuracy:", np.mean(is_correct))
#AV