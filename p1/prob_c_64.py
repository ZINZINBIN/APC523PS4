import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, eye, kron
from scipy.linalg import lu_factor, lu_solve

def generate_sparse_laplacian_1D(N_mesh:int, dx:float):
    L = lil_matrix((N_mesh, N_mesh))
    coeffs = [1.0, -2.0, 1.0]

    for offset, coeff in zip([-1, 0, 1], coeffs):
        L.setdiag(coeff, offset)

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

def generate_Jacobian(u:np.ndarray, laplacian:np.ndarray):
    J = laplacian - 4 * np.diag(np.power(u.ravel(),3))
    return J

def compute_inverse_Jacobian(J:np.ndarray):
    LU, Pinv = lu_factor(J)
    I = np.eye(J.shape[0])
    J_inv = lu_solve((LU, Pinv), I)
    return J_inv

def compute_g(u:np.ndarray, L:np.ndarray, dx:float, N_mesh:int):
    res = L@u - np.power(u, 4)
    res = res.reshape(N_mesh, N_mesh)

    # Boundary condition: u(x,y) = 1 at the boundary
    res[:,0] += 1.0 / dx ** 2
    res[0,:] += 1.0 / dx ** 2
    res[-1,:] += 1.0 / dx ** 2
    res[:,-1] += 1.0 / dx ** 2
    
    res = res.reshape(-1,1)
    return res

def GaussSeidel(A:np.ndarray, b:np.ndarray, x0:np.ndarray,eps:float, n_epoch:int):
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()

    for k in range(n_epoch):
        x_new = x.copy()
        
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            break
        
        x = x_new
        
    return x_new

def compute_inverse_Jacobian_iterative(J:np.ndarray, eps:float, n_epoch:int):

    n = J.shape[0]
    J_inv = np.zeros_like(J)
    
    for i in range(n):
        b = np.zeros(n).reshape(-1,1)
        b[i] = 1.0
        x_init = np.zeros(n).reshape(-1,1)
        r = GaussSeidel(J, b, x_init, n_epoch=n_epoch, eps=eps)
        J_inv[:,i] = r

    return J_inv

def compute_l2_error(x:np.ndarray):
    err = np.sqrt(np.sum(np.power(x,2)))
    return err

def plot_contourf(X: np.ndarray, Y: np.ndarray, u:np.ndarray, filename: str, title:str, dpi: int = 160):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), facecolor="white", dpi=dpi)

    ax.contourf(X, Y, u)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

def solve_c(u_init:np.ndarray, L:np.ndarray, dx:float, N:int,  N_epoch:int, N_iter:int, eps:float, verbose : int):

    u = np.copy(u_init.reshape(-1,1))

    for n_epoch in range(N_epoch):

        J = generate_Jacobian(u, L)
        Jinv = compute_inverse_Jacobian_iterative(J, eps, N_iter)
        u = u - Jinv @ compute_g(u, L, dx, N)

        # error analysis
        l2_err = compute_l2_error(compute_g(u, L, dx, N))
        inf_err = np.linalg.norm(compute_g(u, L, dx, N), ord=np.inf)

        if l2_err < eps:
            break

        if n_epoch % verbose == 0:
            print("Epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    return u, l2_err, inf_err, n_epoch

if __name__ == "__main__":

    # setup
    eps = 1e-8
    N_epoch = 128
    N_iter = 12
    verbose = 4

    print("\n=============== Problem (c): N = 64 ==================")

    # Case: N = 64
    N = 64
    dx = 1.0 / N
    L = generate_laplacian_2D(N, dx=dx)

    lin = np.linspace(0, 1.0, N, endpoint=True)
    X, Y = np.meshgrid(lin, lin)

    u_init = np.zeros((N, N))

    u, l2_err, inf_err, n_epoch = solve_c(u_init, L, dx, N, N_epoch, N_iter, eps, verbose)

    plot_contourf(X, Y, u.reshape(N,N), filename = "p1_c_64.png", title = "u(x,y) with N = 64", dpi = 120)
    print("N = 64 | Final epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    np.save("./p1/u_64.npy", u.reshape(N,N))