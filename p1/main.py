import numpy as np
from scipy.sparse import block_diag, csc_array, lil_matrix, eye, kron

def generate_laplacian_1D(N_mesh:int, dx:float):
    
    L = np.zeros((N_mesh, N_mesh), dtype = np.float64)
    
    for idx_i in range(0,N_mesh):
        if idx_i > 0:
            L[idx_i, idx_i - 1] = 1.0

        if idx_i < N_mesh - 1:
            L[idx_i, idx_i + 1] = 1.0

        L[idx_i, idx_i] = -2.0

    L[0,N_mesh-1] = 1.0
    L[N_mesh-1,0] = 1.0
    L /= dx ** 2
    return L

def generate_sparse_laplacian_1D(N_mesh:int, dx:float):

    L = lil_matrix((N_mesh, N_mesh))
    coeffs = [-1.0 / 12, 16.0 / 12, -30.0 / 12, 16.0 / 12, -1.0 / 12]

    for offset, coeff in zip([-2, -1, 0, 1, 2], coeffs):
        L.setdiag(coeff, offset)

    L[1, 0:5] = [11/12, -20/12, 6/12, 4/12, -1/12]
    L[0, 0:5] = [35/12, -104/12, 114/12, -56/12, 11/12]
    
    L[-2, -5:] = [-1/12, 4/12, 6/12, -20/12, 11/12]
    L[-1, -5:] = [11/12, -56/12, 114/12, -104/12, 35/12]
    L /= dx ** 2
    return L.tocsr()

def generate_laplacian_2D(N_mesh:int, dx:float):
    Lx = generate_sparse_laplacian_1D(N_mesh, dx)
    Ly = generate_sparse_laplacian_1D(N_mesh, dx)

    Ix = eye(N_mesh, format = 'csr')
    Iy = eye(N_mesh, format = 'csr')

    # https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
    L = kron(Lx, Iy) + kron(Ix, Ly)
    return L

if __name__ == "__main__":
    N = 10
    L = generate_laplacian_2D(N**2, dx = 1.0).toarray()
