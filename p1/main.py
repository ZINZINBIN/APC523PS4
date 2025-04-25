import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.sparse import lil_matrix, eye, kron, diags, identity
from scipy.sparse.linalg import gmres
from scipy.linalg import lu_factor, lu_solve
from scipy.ndimage import zoom

def generate_sparse_laplacian_1D(N_mesh:int, dx:float):
    L = lil_matrix((N_mesh, N_mesh), dtype=np.float32)
    coeffs = [1.0, -2.0, 1.0]

    for offset, coeff in zip([-1, 0, 1], coeffs):
        L.setdiag(coeff, offset)

    L /= dx ** 2
    return L.tocsr()

def generate_laplacian_2D(N_mesh:int, dx:float):
    Lx = generate_sparse_laplacian_1D(N_mesh, dx)
    Ly = generate_sparse_laplacian_1D(N_mesh, dx)

    Ix = eye(N_mesh, format = 'csr', dtype = np.float32)
    Iy = eye(N_mesh, format = 'csr', dtype = np.float32)

    # https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
    L = kron(Lx, Iy) + kron(Ix, Ly)
    return L

def generate_Jacobian(u:np.ndarray, laplacian:np.ndarray):
    J = laplacian - 4 * np.diag(np.power(u.ravel(),3))
    return J

def generate_Jacobian_large_N(u:np.ndarray, laplacian:np.ndarray):
    J = laplacian - 4 * diags(np.power(u.ravel(),3))
    return J

def compute_inverse_Jacobian(J:np.ndarray):
    LU, Pinv = lu_factor(J)
    I = np.eye(J.shape[0], dtype = np.float32)
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
    
    res = res.ravel()
    return res

def Jacobi(A:np.ndarray, b:np.ndarray, x0:np.ndarray,eps:float, n_epoch:int):
    
    n = A.shape[0]
    x = identity(n) * 0 if x0 is None else x0.copy()
    w = 2/3

    D_inv = diags(1.0 / A.diagonal())

    is_converged = False

    for k in range(n_epoch):
        
        x_new = x + w * D_inv - w * D_inv @ A @ x
        if np.linalg.norm((x_new - x).toarray()) / n < eps:
            is_converged = True
            break
        x = x_new

    return x_new, is_converged

def compute_inverse_Jacobian_iterative(J:np.ndarray, eps:float, n_epoch:int, J_inv_prev = None):
    n = J.shape[0]
    J_inv, is_converged = Jacobi(J, np.eye(n, dtype = np.float32), J_inv_prev, eps, n_epoch)
    return J_inv, is_converged

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

def solve_b(u_init:np.ndarray, L:np.ndarray, dx:float, N:int,  N_epoch:int, eps:float, verbose : Optional[int]):
    u = np.copy(u_init.reshape(-1,1))
    
    for n_epoch in range(N_epoch):

        J = generate_Jacobian(u, L)
        Jinv = compute_inverse_Jacobian(J)
        u = u - Jinv @ compute_g(u, L, dx, N).reshape(-1,1)

        # error analysis
        l2_err = compute_l2_error(compute_g(u,L,dx,N))
        inf_err = np.linalg.norm(compute_g(u,L,dx,N), ord=np.inf)

        if l2_err < eps:
            break
        
        if verbose is not None:
            if n_epoch % verbose == 0:
                print("Epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    return u, l2_err, inf_err, n_epoch

def solve_c(u_init:np.ndarray, L:np.ndarray, dx:float, N:int,  N_epoch:int, N_iter:int, eps:float, verbose : int):

    u = np.copy(u_init)

    for n_epoch in range(N_epoch):

        J = generate_Jacobian(u, L)
        
        # Use scipy sparse iterative solver: generalized minimal residual iteration
        du, _ = gmres(J, compute_g(u, L, dx, N), rtol=eps, maxiter=N_iter)
        u -= du
        
        # error analysis
        l2_err = compute_l2_error(compute_g(u, L, dx, N))
        inf_err = np.linalg.norm(compute_g(u, L, dx, N), ord=np.inf)

        if l2_err < eps:
            break

        if n_epoch % verbose == 0:
            print("Epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    return u, l2_err, inf_err, n_epoch

def solve_c_large_N(u_init:np.ndarray, L:np.ndarray, dx:float, N:int,  N_epoch:int, N_iter:int, eps:float, verbose : int):
    
    u = np.copy(u_init)
    
    for n_epoch in range(N_epoch):

        J = generate_Jacobian_large_N(u, L)
        
        # Use scipy sparse iterative solver: generalized minimal residual iteration
        du, _ = gmres(J, compute_g(u, L, dx, N), rtol=eps, maxiter=N_iter)
        u -= du
        
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
    N = 64
    N_epoch =64
    N_iter = 16
    dx =  1.0 / N
    verbose = 1

    # Laplacian 2D
    L = generate_laplacian_2D(N, dx=dx)

    # mesh
    lin = np.linspace(0, 1.0, N, endpoint = True)
    X,Y = np.meshgrid(lin,lin)

    # Initial u(x,y)
    u_init = np.zeros((N,N))
    u = np.copy(u_init.reshape(-1,1))

    # Problem (b)
    print("\n=============== Problem (b) ==================")
    u, l2_err, inf_err, n_epoch = solve_b(u_init, L, dx, N, N_epoch, eps, verbose)

    print("Final epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))
    plot_contourf(X, Y, u.reshape(N,N), filename = "p1_b.png", title = "u(x,y)", dpi = 120)

    np.save("./p1/u_64_b.npy", u.reshape(N, N))

    # Problem (c)
    print("\n=============== Problem (c): N = 64 ==================")

    # Case: N = 64
    N = 64
    dx = 1.0 / N
    L = generate_laplacian_2D(N, dx=dx)

    lin = np.linspace(0, 1.0, N, endpoint=True)
    X, Y = np.meshgrid(lin, lin)

    u_init = np.zeros((N, N), dtype=np.float32).ravel()

    u, l2_err, inf_err, n_epoch = solve_c_large_N(u_init, L, dx, N, N_epoch, N_iter, eps, verbose)

    plot_contourf(X, Y, u.reshape(N,N), filename = "p1_c_64.png", title = "u(x,y) with N = 64", dpi = 120)
    print("N = 64 | Final epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    np.save("./p1/u_64.npy", u.reshape(N,N))

    # remove variables for saving the memory
    del L
    del u
    del u_init
    del X
    del Y

    print("\n=============== Problem (c): N = 128  =================")

    # Case: N = 128
    N = 128
    dx = 1.0 / N
    L = generate_laplacian_2D(N, dx=dx)

    lin = np.linspace(0, 1.0, N, endpoint=True)
    X, Y = np.meshgrid(lin, lin)

    u_init = np.zeros((N, N), dtype=np.float32).ravel()

    u, l2_err, inf_err, n_epoch = solve_c_large_N(u_init, L, dx, N, N_epoch, N_iter, eps, verbose)

    plot_contourf(X, Y, u.reshape(N,N), filename = "p1_c_128.png", title = "u(x,y) with N = 128", dpi = 120)
    print("N = 128 | Final epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    np.save("./p1/u_128.npy", u.reshape(N,N))

    # remove variables for saving the memory
    del L
    del u
    del u_init
    del X
    del Y

    print("\n=============== Problem (c): N = 256  =================")

    # Case: N = 256
    N = 256
    dx = 1.0 / N
    L = generate_laplacian_2D(N, dx=dx)

    lin = np.linspace(0, 1.0, N, endpoint=True)
    X, Y = np.meshgrid(lin, lin)

    u_init = np.zeros((N, N), dtype=np.float32).ravel()

    u, l2_err, inf_err, n_epoch = solve_c_large_N(u_init, L, dx, N, N_epoch, N_iter, eps, verbose)

    plot_contourf(X, Y, u.reshape(N,N), filename = "p1_c_256.png", title = "u(x,y) with N = 256", dpi = 120)
    print("N = 256 | Final epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    np.save("./p1/u_256.npy", u.reshape(N, N))    

    # remove variables for saving the memory
    del L
    del u
    del u_init
    del X
    del Y

    # The OOM error happens when N = 512
    # So, I just skipped this process and set the ground-truth with the solution on problem b instead
    print("\n=============== Problem (c): N = 512  =================")

    # Case: N = 512
    N = 512
    dx = 1.0 / N
    L = generate_laplacian_2D(N, dx=dx)

    lin = np.linspace(0, 1.0, N, endpoint=True)
    X, Y = np.meshgrid(lin, lin)

    u_init = np.zeros((N, N), dtype=np.float32).ravel()

    u, l2_err, inf_err, n_epoch = solve_c_large_N(u_init, L, dx, N, N_epoch, N_iter, eps, verbose)

    plot_contourf(X, Y, u.reshape(N,N), filename = "p1_c_512.png", title = "u(x,y) with N = 512", dpi = 120)
    print("N = 512 | Final epoch: {} | L2 norm:{:.4f} | Inf norm:{:.4f}".format(n_epoch + 1, l2_err, inf_err))

    np.save("./p1/u_512.npy", u.reshape(N, N))

    # remove variables for saving the memory
    del L
    del u
    del u_init
    del X
    del Y
    
    
    u_64 = np.load("./p1/u_64.npy")
    u_128 = zoom(np.load("./p1/u_128.npy"), (0.5, 0.5))
    u_256 = zoom(np.load("./p1/u_256.npy"), (0.25, 0.25))
    u_gt =  zoom(np.load("./p1/u_512.npy"), (0.125, 0.125))

    N_list = [64, 128, 256, 512]
    err_list = [
        np.linalg.norm(u_64 - u_gt, ord="fro"),
        np.linalg.norm(u_128 - u_gt, ord="fro"),
        np.linalg.norm(u_256 - u_gt, ord="fro"),
        0,
    ]

    # Error plot with original and log scale
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white", dpi=120)
    axes = axes.ravel()
    axes[0].plot(N_list, err_list, "ro-")
    axes[0].set_xlabel("Number of cells")
    axes[0].set_ylabel("L2 norm")
    axes[0].set_title("N vs Accuracy")

    axes[1].plot(N_list[:-1], err_list[:-1], "ro-")
    axes[1].set_xlabel("Number of cells (log-scale)")
    axes[1].set_ylabel("L2 norm (log-scale)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("N vs Accuracy")

    fig.tight_layout()
    fig.savefig("p1_c_err.png", dpi=120)
    plt.close(fig)
