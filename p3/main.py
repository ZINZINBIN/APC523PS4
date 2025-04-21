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
    
    uxr = np.roll(u, shift = R, axis = 0)
    uxl = np.roll(u, shift = L, axis = 0)
    
    uyr = np.roll(u, shift = R, axis = 1)
    uyl = np.roll(u, shift = L, axis = 1)
    
    uxryr = np.roll(uxr, shift = R, axis = 1)
    uxryl = np.roll(uxr, shift = L, axis = 1)
    uxlyr = np.roll(uxl, shift = R, axis = 1)
    uxlyl = np.roll(uxl, shift = L, axis = 1)
    
    u_next += (-1) * 0.5 * a * dt / dx * (uxr - uxl) \
            + (-1) * 0.5 * b * dt / dy * (uyr - uyl) \
            + a ** 2 * dt ** 2 * 0.5 / dx ** 2 * (uxr - 2*u + uxl) \
            + b ** 2 * dt ** 2 * 0.5 / dy ** 2 * (uyr - 2*u + uyl) \
            + a * b * dt ** 2 * 0.25 / dx / dy * (uxryr - uxryl - uxlyr + uxlyl)
    
    return u_next

def plot_contourf(X:np.ndarray, Y:np.ndarray, u_gt, u_ctu, u_lax, filename:str, dpi:int = 160):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white", dpi=dpi)
    axes = axes.ravel()

    axes[0].contourf(X, Y, u_gt, np.linspace(0,1,50))
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Analytic solution")

    axes[0].contourf(X, Y, u_ctu, np.linspace(0, 1, 50))
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("CTU")

    axes[0].contourf(X, Y, u_lax, np.linspace(0, 1, 50))
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Lax-Wendroff")

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":
    a = 1
    b = 2
    Nt = 100000
    ts = np.linspace(0, 10, Nt)
    dt = ts[1] - ts[0]

    N_list = [128, 256, 512]

    X1, Y1 = np.meshgrid(np.linspace(0, 1, N_list[0]), np.linspace(0, 1, N_list[0]))
    X2, Y2 = np.meshgrid(np.linspace(0, 1, N_list[1]), np.linspace(0, 1, N_list[1]))
    X3, Y3 = np.meshgrid(np.linspace(0, 1, N_list[2]), np.linspace(0, 1, N_list[2]))

    u_gt_1 = analytic_solution(X1,Y1,10.0, a,b)
    u_init_1 = init_condition(X1,Y1)

    u_gt_2 = analytic_solution(X2, Y2, 10.0, a, b)
    u_init_2 = init_condition(X2, Y2)

    u_gt_3 = analytic_solution(X3,Y3,10.0, a,b)
    u_init_3 = init_condition(X3,Y3)

    dx1 = dy1 = 1.0 / N_list[0]
    dx2 = dy2 = 1.0 / N_list[1]
    dx3 = dy3 = 1.0 / N_list[2]

    u_lax_1 = u_ctu_1 = np.copy(u_init_1)
    u_lax_2 = u_ctu_2 = np.copy(u_init_2)
    u_lax_3 = u_ctu_3 = np.copy(u_init_3)

    l2_ctu_list_1 = []
    l2_lax_list_1 = []

    l2_ctu_list_2 = []
    l2_lax_list_2 = []

    l2_ctu_list_3 = []
    l2_lax_list_3 = []

    for t in ts[1:]:
        u_gt_1 = analytic_solution(X1, Y1, t, a, b)
        u_lax_1 = LaxWendroff(u_lax_1, dx1, dy1, dt, a, b)   
        u_ctu_1 = CTU(u_ctu_1, dx1, dy1, dt, a, b) 

        l2_ctu_1 = np.sqrt(np.sum((u_gt_1 - u_ctu_1) ** 2))
        l2_lax_1 = np.sqrt(np.sum((u_gt_1 - u_lax_1) ** 2))

        l2_ctu_list_1.append(l2_ctu_1)
        l2_lax_list_1.append(l2_lax_1)

        u_gt_2 = analytic_solution(X2, Y2, t, a, b)
        u_lax_2 = LaxWendroff(u_lax_2, dx2, dy2, dt, a, b)
        u_ctu_2 = CTU(u_ctu_2, dx2, dy2, dt, a, b)

        l2_ctu_2 = np.sqrt(np.sum((u_gt_2 - u_ctu_2) ** 2))
        l2_lax_2 = np.sqrt(np.sum((u_gt_2 - u_lax_2) ** 2))

        l2_ctu_list_2.append(l2_ctu_2)
        l2_lax_list_2.append(l2_lax_2)

        u_gt_3 = analytic_solution(X3, Y3, t, a, b)
        u_lax_3 = LaxWendroff(u_lax_3, dx3, dy3, dt, a, b)
        u_ctu_3 = CTU(u_ctu_3, dx3, dy3, dt, a, b)

        l2_ctu_3 = np.sqrt(np.sum((u_gt_3 - u_ctu_3) ** 2))
        l2_lax_3 = np.sqrt(np.sum((u_gt_3 - u_lax_3) ** 2))

        l2_ctu_list_3.append(l2_ctu_3)
        l2_lax_list_3.append(l2_lax_3)

    plot_contourf(X1, Y1, u_gt_1, u_ctu_1, u_lax_1, "p3_c_contour_128.png", 120)
    plot_contourf(X2, Y2, u_gt_2, u_ctu_2, u_lax_2, "p3_c_contour_256.png", 120)
    plot_contourf(X3, Y3, u_gt_3, u_ctu_3, u_lax_3, "p3_c_contour_512.png", 120)

    # Plot the L2 norm
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.ravel()

    axes[0].plot(ts[1:], l2_ctu_list_1, "r", label="CTU")
    axes[0].plot(ts[1:], l2_lax_list_1, "g", label="Lax-Wendroff")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("L2 error")
    axes[0].legend(loc="upper right")
    axes[0].set_title("L2 error with N = {}".format(N_list[0]))

    axes[1].plot(ts[1:], l2_ctu_list_2, "r", label="CTU")
    axes[1].plot(ts[1:], l2_lax_list_2, "g", label="Lax-Wendroff")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("L2 error")
    axes[1].legend(loc="upper right")
    axes[1].set_title("L2 error with N = {}".format(N_list[1]))

    axes[2].plot(ts[1:], l2_ctu_list_3, "r", label="CTU")
    axes[2].plot(ts[1:], l2_lax_list_3, "g", label="Lax-Wendroff")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("L2 error")
    axes[2].legend(loc="upper right")
    axes[2].set_title("L2 error with N = {}".format(N_list[2]))

    fig.tight_layout()
    fig.savefig("./p3_error.png")
