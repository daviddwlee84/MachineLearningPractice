import numpy as np

class ReciprocalRank:
    def __init__(self, R, Y):
        self.R = R # Rank (user, item)
        self.Y = Y # Binary Relevance Score (user, item)
        self.N = np.shape(R)[1] # number of items

    def RR(self, i):
        RRi = 0.0
        for j in range(self.N):
            prod_temp = 1
            for k in range(self.N):
                prod_temp *= (1 - self.Y[i, k] * (self.R[i, k] < self.R[i, j]))
                # print(k, (self.R[i, k] < self.R[i, j]))
            RRi += self.Y[i, j]/self.R[i, j] * prod_temp
            print("j={}: {}/{} * {}"
                    .format(j, self.Y[i, j], self.R[i, j], prod_temp))
        return RRi

# In CLiMF paper
class SmoothReciprocalRank:
    def __init__(self):
        pass

def main():
    R = np.array([
        [3, 4, 2, 5, 1], # user 0
    ]) 

    R2 = np.array([[3, 4, 5, 2, 1]])
    R3 = np.array([[3, 4, 2, 1, 5]])

    Y = np.array([
        [1, 1, 0, 0, 1], # user 0
    ])

    print("RR =", ReciprocalRank(R, Y).RR(0))
    print("RR =", ReciprocalRank(R2, Y).RR(0))
    print("RR =", ReciprocalRank(R3, Y).RR(0))

if __name__ == "__main__":
    main()
