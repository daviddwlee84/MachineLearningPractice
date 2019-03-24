import numpy as np
import pickle # for model preservation
from Epinions_Preprocessing import load_epinions, get_sample_users

class sigmoid: # callable sigmoid function class
    def __init__(self, x):
        self.x = x

    def __call__(self):
        return 1/(1+np.exp(-self.x))
    
    def derivative(self):
        return np.exp(self.x)/(1+np.exp(self.x))**2

# i = user
# j, k = item
class CLiMF:
    def __init__(self, data, lamb=0.001, gamma=0.0001, dimension=10, max_iters=25):
        self.__data = data # Scipy sparse metrix => user->(item, count)
        self.__lambda = lamb # Regularization constant lambda
        self.__gamma = gamma # Learning rate
        self.__max_iters = max_iters
        self.U = 0.01 * np.random.random_sample((data.shape[0], dimension))
        self.V = 0.01 * np.random.random_sample((data.shape[1], dimension))
    
    def load(self, filename="CLiMF_model.pickle"):
        with open(filename, 'rb') as model_file:
            model_dict = pickle.load(model_file)
        self.__dict__.update(model_dict)
    
    def save(self, filename="CLiMF_model.pickle"):
        with open(filename, 'wb') as model_file:
            pickle.dump(self.__dict__, model_file)
    
    def __f(self, i):
        items = self.__data[i].indices
        fi = dict((j, np.dot(self.U[i], self.V[j])) for j in items)
        return fi # Get <U[i], V[j]> for all j in data[i]

    # Objective function (predict)
    # U: user latent factor
    # V: item latent factor
    def F(self):
        F = 0
        for i in range(len(self.U)):
            fi = self.__f(i)
            for j in fi:
                F += np.log(sigmoid(fi[j])())
                for k in fi:
                    F += np.log(1 - sigmoid(fi[k]-fi[j])())
        F -= 0.5 * self.__lambda * (np.sum(self.U * self.U) + np.sum(self.V * self.V)) # Forbenius norm
        return F
    
    # Stochastic gradient ascent (maximize the objective function)
    def __train_one_round(self):
        for i in range(len(self.U)):
            dU = -self.__lambda * self.U[i]
            fi = self.__f(i)
            for j in fi:
                # Calculate dV
                dV = sigmoid(-fi[j])() - self.__lambda * self.V[j]
                for k in fi:
                    dV += sigmoid(fi[j]-fi[k]).derivative() * (1/(1-sigmoid(fi[k] - fi[j])())) - (1/(1-sigmoid(fi[j] - fi[k])())) * self.U[i]
                self.V[j] += self.__gamma * dV

                # Calculate dU
                dU += sigmoid(-fi[j])() * self.V[j]
                for k in fi:
                    dU += (self.V[j] - self.V[k]) * sigmoid(fi[k] - fi[j]).derivative() / (1-sigmoid(fi[k] - fi[j])())
            self.U[i] += self.__gamma * dU

    def train(self, verbose=False, sample_users=None, max_iters=-1):
        if max_iters <= 0:
            max_iters = self.__max_iters
            
        for time in range(max_iters):
            self.__train_one_round()
            if verbose:
                print('iteration:', time+1)
                print('F(U, V) =', self.F())
                print('Train MRR =', aMRRevaluate(self.__data, self, sample_users))

# average Mean Reciprocal Rank
def aMRRevaluate(data, climf_model, sample_users=None):
    MRR = []
    if not sample_users:
        sample_users = range(len(climf_model.U))
    for i in sample_users:
        items = set(data[i].indices)
        predict = np.sum(np.tile(climf_model.U[i], (len(climf_model.V), 1)) * climf_model.V, axis=1) 
        for rank, item in enumerate(np.argsort(predict)[::-1]):
            if item in items:
                MRR.append(1.0/(rank+1))
                break
    return np.mean(MRR)

def main():
    TRAIN = True # Train or Load the model

    print("Loading Epinions dataset...")
    train_data, test_data = load_epinions()
    train_sample_users, test_sample_users = get_sample_users(train_data, test_data)

    print("Before training:")
    CF_model = CLiMF(train_data)
    print("aMRR of training data:", aMRRevaluate(train_data, CF_model, train_sample_users))
    print("aMRR of test data:", aMRRevaluate(test_data, CF_model, test_sample_users))

    if TRAIN:
        print("Training...")
        CF_model.train(verbose=True, sample_users=train_sample_users)
    else:
        print("Load pre-trained model...")
        CF_model.load()

    print("After training:")
    print("aMRR of training data:", aMRRevaluate(train_data, CF_model, train_sample_users))
    print("aMRR of test data:", aMRRevaluate(test_data, CF_model, test_sample_users))

    print("Result of U, V")
    print("U:", CF_model.U)
    print("V:", CF_model.V)

    CF_model.save() # save model

if __name__ == "__main__":
    # Test sigmoid callable class
    # print(sigmoid(-87)())
    # print(sigmoid(87).derivative())
    main()
