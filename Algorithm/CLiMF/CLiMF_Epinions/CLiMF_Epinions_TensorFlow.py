import tensorflow as tf
tf.enable_eager_execution() # Eager Execution
import tensorflow.contrib.eager as tfe

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
        self.__dim = dimension
        # EagerTensor does not support item assignment, thus transform it into tf.Variable
        self.U = tfe.Variable(tf.convert_to_tensor(0.01 * np.random.random_sample((data.shape[0], dimension))))
        self.V = tfe.Variable(tf.convert_to_tensor(0.01 * np.random.random_sample((data.shape[1], dimension))))
    
    def load(self, filename="CLiMF_TF_model.pickle"):
        with open(filename, 'rb') as model_file:
            model_dict = pickle.load(model_file)
        self.__dict__.update(model_dict)
    
    def save(self, filename="CLiMF_TF_model.pickle"):
        with open(filename, 'wb') as model_file:
            pickle.dump(self.__dict__, model_file)
    
    def __f(self, i):
        items = self.__data[i].indices
        # take notice to axes
        fi = dict((j, tf.tensordot(self.U[i], self.V[j], axes=1)) for j in items)
        return fi # Get <U[i], V[j]> for all j in data[i]

    # Objective function (predict)
    # U: user latent factor
    # V: item latent factor
    def F(self):
        F = 0
        num_of_rows = self.U.get_shape()[0]
        for i in range(num_of_rows):
            fi = self.__f(i)
            for j in fi:
                F += np.log(sigmoid(fi[j])())
                for k in fi:
                    F += np.log(1 - sigmoid(fi[k]-fi[j])())
        F -= 0.5 * self.__lambda * (np.sum(tf.multiply(self.U, self.U)) + np.sum(tf.multiply(self.V, self.V))) # Forbenius norm
        return F
    
    # Stochastic gradient ascent (maximize the objective function)
    def __train_one_round(self):
        num_of_rows = self.U.get_shape()[0]
        for i in range(num_of_rows):
            dU = -self.__lambda * self.U[i]
            fi = self.__f(i)
            for j in fi:
                # Calculate dV
                dV = sigmoid(-fi[j])() - self.__lambda * self.V[j]
                for k in fi:
                    dV += sigmoid(fi[j]-fi[k]).derivative() * (1/(1-sigmoid(fi[k] - fi[j])())) - (1/(1-sigmoid(fi[j] - fi[k])())) * self.U.numpy()[i]
                # original: self.V[j] += self.__gamma * dV
                # Method 1
                self.V[j].assign(self.V[j] + self.__gamma * dV)
                # Method 2
                # V_indices = tf.constant([[j, z] for z in range(self.__dim)], dtype=tf.int32)
                # self.V = tf.scatter_nd_update(self.V, V_indices, self.V[j] + self.__gamma * dV)
                # (Both methods will work)

                # Calculate dU
                dU += sigmoid(-fi[j])() * self.V[j]
                for k in fi:
                    dU += (self.V[j] - self.V[k]) * sigmoid(fi[k] - fi[j])() / (1-sigmoid(fi[k] - fi[j])())
            # original: self.U[i] += self.__gamma * dU
            # Method 1
            self.U[i].assign(self.U[i] + self.__gamma * dU)
            # Method 2
            # U_indices = tf.constant([[i, z] for z in range(self.__dim)], dtype=tf.int32)
            # self.U = tf.scatter_nd_update(self.U, U_indices, self.U[i] + self.__gamma * dV)
            # (Both methods will work)

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
        sample_users = range(climf_model.U.get_shape()[0])
        
    for i in sample_users:
        items = set(data[i].indices)
        predict = np.sum(np.tile(climf_model.U[i], (climf_model.V.get_shape()[0], 1)) * climf_model.V, axis=1) 
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
    main()
