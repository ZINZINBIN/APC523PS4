import numpy as np
import matplotlib.pyplot as plt

def init_condition(x:float, y:float):
    return np.exp(-((x-1/2)**2 + (y-1/2)**2) / (3/20)**2)

def analytic_solution(x:float, y:float, t:float, a:float, b:float):
    eta = x - a * t
    nu = y - b * t
    
    # double periodic condtion
    eta = np.mod(eta, 1.0)
    nu = np.mod(nu, 1.0)
    
    return init_condition(eta, nu)

def CTU(u: np.ndarray, dx: float, dy: float, dt: float, a: float, b: float):
    mu = a * dt / dx
    nu = b * dt / dy

    L = 1
    uxl = np.roll(u, shift=L, axis=0)
    uyl = np.roll(u, shift=L, axis=1)
    u_next = (1-mu-nu) * u + mu * uxl + nu * uyl
    return u_next

def LaxWendroff(u:np.ndarray, dx:float, dy:float, dt:float, a:float, b:float):
    u_next = np.copy(u)

    R = -1
    L = 1

    mu = a * dt / dx
    nu = b * dt / dy

    uxr = np.roll(u, shift = R, axis = 0)
    uxl = np.roll(u, shift = L, axis = 0)

    uyr = np.roll(u, shift = R, axis = 1)
    uyl = np.roll(u, shift = L, axis = 1)

    uxryr = np.roll(uxr, shift = R, axis = 1)
    uxryl = np.roll(uxr, shift = L, axis = 1)
    uxlyr = np.roll(uxl, shift = R, axis = 1)
    uxlyl = np.roll(uxl, shift = L, axis = 1)

    u_next += (-1) * 0.5 * mu * (uxr - uxl) \
            + (-1) * 0.5 * nu * (uyr - uyl) \
            + 0.5 * mu ** 2 * (uxr - 2*u + uxl) \
            + 0.5 * nu ** 2 * (uyr - 2*u + uyl) \
            + 0.25 * mu * nu * (uxryr - uxryl - uxlyr + uxlyl)

    return u_next

def run_simulation(N:int, a:float, b:float, T:float, CFL:float, filename:str):

    dx = dy = 1.0 / N
    lin = np.linspace(0, 1.0, N)
    X, Y = np.meshgrid(lin,lin)

    dt = CFL / (a/dx + b/dy)
    Nt = int(T / dt)
    ts = np.linspace(0, T, Nt+1, endpoint = True)

    u_init = init_condition(X,Y)
    u_ctu = np.copy(u_init)
    u_lax = np.copy(u_init)

    l2_ctu_list = []
    l2_lax_list = []
    
    print("Simulation start")
    for t in ts[1:]:
        u_gt = analytic_solution(X, Y, t, a, b)
        u_lax = LaxWendroff(u_lax, dx, dy, dt, a, b)
        u_ctu = CTU(u_ctu, dx, dy, dt, a, b)

        l2_ctu = np.sqrt(np.sum((u_gt - u_ctu) ** 2))
        l2_lax = np.sqrt(np.sum((u_gt - u_lax) ** 2))

        l2_ctu_list.append(l2_ctu)
        l2_lax_list.append(l2_lax)

    print("Simulation end")
    plot_contourf(X, Y, u_gt, u_ctu, u_lax, filename, 120)
    
    return ts[1:], u_gt, u_ctu, u_lax, l2_ctu_list, l2_lax_list

def plot_contourf(X:np.ndarray, Y:np.ndarray, u_gt, u_ctu, u_lax, filename:str, dpi:int = 160):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=dpi)
    axes = axes.ravel()

    axes[0].contourf(X, Y, u_gt)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Analytic solution")

    axes[1].contourf(X, Y, u_ctu)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("CTU")

    axes[2].contourf(X, Y, u_lax)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Lax-Wendroff")

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":

    a = 1.0
    b = 2.0
    T = 10.0
    CFL = 0.4

    N_list = [128, 256, 512]
    
    # u(1,x,y)
    _, _, _, _, _, _ = run_simulation(N_list[0], a, b, 1.0, CFL, "p3_c_contour_128_t_1.png")
    _, _, _, _, _, _ = run_simulation(N_list[1], a, b, 1.0, CFL, "p3_c_contour_256_t_1.png")
    _, _, _, _, _, _ = run_simulation(N_list[2], a, b, 1.0, CFL, "p3_c_contour_512_t_1.png")

    # u(10,x,y)
    ts1, _, _, _, l2_ctu_list_1, l2_lax_list_1 = run_simulation(N_list[0], a, b, T, CFL, "p3_c_contour_128_t_10.png")
    ts2, _, _, _, l2_ctu_list_2, l2_lax_list_2 = run_simulation(N_list[1], a, b, T, CFL, "p3_c_contour_256_t_10.png")
    ts3, _, _, _, l2_ctu_list_3, l2_lax_list_3 = run_simulation(N_list[2], a, b, T, CFL, "p3_c_contour_512_t_10.png")

    # Plot the L2 norm
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.ravel()

    axes[0].plot(ts1, l2_ctu_list_1, "r", label="CTU")
    axes[0].plot(ts1, l2_lax_list_1, "g", label="Lax-Wendroff")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("L2 error")
    axes[0].legend(loc="upper right")
    axes[0].set_title("L2 error with N = {}".format(N_list[0]))

    axes[1].plot(ts2, l2_ctu_list_2, "r", label="CTU")
    axes[1].plot(ts2, l2_lax_list_2, "g", label="Lax-Wendroff")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("L2 error")
    axes[1].legend(loc="upper right")
    axes[1].set_title("L2 error with N = {}".format(N_list[1]))

    axes[2].plot(ts3, l2_ctu_list_3, "r", label="CTU")
    axes[2].plot(ts3, l2_lax_list_3, "g", label="Lax-Wendroff")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("L2 error")
    axes[2].legend(loc="upper right")
    axes[2].set_title("L2 error with N = {}".format(N_list[2]))

    fig.tight_layout()
    fig.savefig("./p3_error.png")
