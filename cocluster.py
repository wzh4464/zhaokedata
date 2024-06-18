import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds


class CoCluster:
    """A class for co-clustering a given matrix.

    Explanation:
    - Contains methods to perform co-clustering on a given matrix.
    - Includes functions to create the co-cluster matrix, minimize a given matrix, and perform the co-clustering process.

    Methods:
    - minimize(Y, A): Minimizes a given matrix based on input matrices Y and A.
    - create_A(B): Creates a co-cluster matrix based on the input matrix B.
    - co_cluster(): Performs the co-clustering process and returns the co-clustered matrix.
    """

    def __init__(self, B, k):
        """Initializes the CoCluster class with input matrix B and integer k.

        Explanation:
        - Sets the attributes B, k, m, n, and N based on the input values.
        """

        self.B = B
        self.k = k
        self.m, self.n = B.shape
        self.N = self.m + self.n

    def create_A(self, B):
        """Creates a co-cluster matrix based on the input matrix B.

        Explanation:
        - Constructs a co-cluster matrix by stacking and concatenating zero matrices with the input matrix B.

        Args:
        - B: Input matrix used to create the co-cluster matrix.

        Returns:
        - The constructed co-cluster matrix.
        """

        B_T = B.T
        upper_zeros = np.zeros((self.m, self.m))
        right_zeros = np.zeros((self.n, self.n))
        return np.vstack((np.hstack((upper_zeros, B)), np.hstack((B_T, right_zeros))))

    def co_cluster(self):
        """Performs the co-clustering process.

        Explanation:
        - Executes the co-clustering algorithm on the input matrix B using the minimize and create_A methods.

        Returns:
        - The co-clustered matrix after the co-clustering process.
        """

        A = self.create_A(self.B)
        D = np.diag(np.sum(A, axis=1))
        
        D_u = D[: self.m, : self.m]
        D_v = D[self.m :, self.m :]
        
        M = np.dot(fractional_matrix_power(D_u, -0.5), np.dot(B, fractional_matrix_power(D_v, -0.5)))
        
        # SVD decomposition, first k singular vectors
        U, S, Vh = svds(M, k=k)
        
        F = np.vstack((U, Vh.T))

        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(F)
        label = kmeans.labels_

        labele_u = label[: self.m]
        labele_v = label[self.m :]

        vis_B = self.B[labele_u.argsort()]
        vis_B = vis_B[:, labele_v.argsort()]

        return vis_B, labele_u, labele_v

if __name__ == "__main__":
    B = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 2, 2, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 3, 3, 0],
        ]
    )

    k = 3
    cocluster = CoCluster(B, k)
    rearranged_B, label_u, label_v = cocluster.co_cluster()
    print(f"Rearranged B: \n{rearranged_B}")
    print(f"Label u: {label_u}")
    print(f"Label v: {label_v}")
