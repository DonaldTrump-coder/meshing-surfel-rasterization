import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero vector")
    return v / n

def ellipse_Sigma2_from_axis_dirs_3d(a, b,
                                     axis1_3, axis2_3,
                                     e1, e2 # 平面基向量
                                     ):
    """
    axis1_3, axis2_3: unit axis direction vectors of the ellipse in 3D (they should be orthonormal and lie in the plane)
    e1,e2: plane basis (unit)
    返回平面2D下的 Sigma2 (2x2)
    证明/推导：在平面基 (e1,e2) 下，投影后的轴向量 v1,v2 (2D)，
    椭圆 Sigma = a^2 v1 v1^T + b^2 v2 v2^T
    """
    v1 = np.array([np.dot(axis1_3, e1), np.dot(axis1_3, e2)])
    v2 = np.array([np.dot(axis2_3, e1), np.dot(axis2_3, e2)])
    # Sigma2 = a^2 v1 v1^T + b^2 v2 v2^T
    Sigma2 = (a*a) * np.outer(v1, v1) + (b*b) * np.outer(v2, v2)
    return Sigma2

def support_point_2d(c2, Sigma2, u2):
    denom = np.sqrt(float(u2.T @ Sigma2 @ u2))
    x2 = c2 + (Sigma2 @ u2) / denom
    h = float(u2.T @ x2)
    return x2, h

def mvee(points, tol=1e-7, max_iter=1000):
    P = np.asarray(points)
    N, d = P.shape
    Q = np.hstack((P, np.ones((N,1))))
    u = np.ones(N) / N
    for _ in range(max_iter):
        X = (Q.T * u) @ Q
        try:
            M = np.diag(Q @ np.linalg.inv(X) @ Q.T)
        except np.linalg.LinAlgError:
            X += 1e-12 * np.eye(d+1)
            M = np.diag(Q @ np.linalg.inv(X) @ Q.T)
        j = np.argmax(M)
        max_M = M[j]
        step = (max_M - d - 1) / ((d+1) * (max_M - 1.0))
        new_u = (1 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    c = (P.T @ u).reshape(d,)
    P_centered = P - c
    S = (P_centered.T * u) @ P_centered
    A = np.linalg.inv(S) / d
    return c, A

def ellipse_params_from_A_2d(c2, A):
    Sigma = np.linalg.inv(A)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    a = np.sqrt(eigvals[1])
    b = np.sqrt(eigvals[0])
    v = eigvecs[:,1]
    theta = np.arctan2(v[1], v[0])
    return c2, a, b, theta, Sigma

def verify_coverage_simple(ell1, ell2, merged):
    """
    merged: {"center3", "axis_vec1_3", "axis_vec2_3"}
    返回 True/False 是否覆盖
    """
    def sample_points(ell, n=360):
        C, ax1, ax2, a, b = ell
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        pts = np.array([C + a*np.cos(tt)*ax1 + b*np.sin(tt)*ax2 for tt in t])
        return pts
    pts = np.vstack([sample_points(ell1), sample_points(ell2)])
    # 检查是否在 merged 椭圆内 (x-C)^T Σ^{-1} (x-C) <= 1
    # 先构造 Σ
    ax1 = merged["axis_vec1_3"]
    ax2 = merged["axis_vec2_3"]
    center = merged["center"]
    Sigma = np.outer(ax1, ax1) + np.outer(ax2, ax2)
    Sigma_inv = np.linalg.pinv(Sigma)
    dif = pts - center
    vals = np.einsum('ij,jk,ik->i', dif, Sigma_inv, dif)
    ok = np.all(vals <= 1.0 + 1e-9)
    return ok, np.max(vals)

def merge_two_ellipses_simple_3d(ell1, ell2, m=64, safety_factor=1.0):
    """
    ell = (C3, axis1_3, axis2_3, a, b)
    axis1_3, axis2_3: *unit* direction vectors in 3D
    输出: center3, axis_vec1_3, axis_vec2_3, a, b
    """
    C1, ax11, ax12, a1, b1 = ell1
    C2, ax21, ax22, a2, b2 = ell2

    # 公共平面基
    e1 = ax11
    e2 = ax12
    if np.linalg.norm(e2) < 1e-12:
        return None

    # 投影到平面坐标
    def to_plane_coords(C):
        return np.array([np.dot(C, e1), np.dot(C, e2)])
    c1_2 = to_plane_coords(C1)
    c2_2 = to_plane_coords(C2)

    # Sigma2
    Sigma1 = ellipse_Sigma2_from_axis_dirs_3d(a1, b1, ax11, ax12, e1, e2)
    Sigma2 = ellipse_Sigma2_from_axis_dirs_3d(a2, b2, ax21, ax22, e1, e2)

    # 支撑点合并
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    pts2 = []
    for th in thetas:
        u2 = np.array([np.cos(th), np.sin(th)])
        x1, h1 = support_point_2d(c1_2, Sigma1, u2)
        x2, h2 = support_point_2d(c2_2, Sigma2, u2)
        pts2.append(x1 if h1 >= h2 else x2)
    pts2 = np.array(pts2)

    # MVEE
    c_mve2, A_mve = mvee(pts2, tol=1e-7)
    if safety_factor != 1.0:
        A_mve = A_mve / (safety_factor**2)

    # 提取参数
    c2d, a_out, b_out, theta_out, _ = ellipse_params_from_A_2d(c_mve2, A_mve)
    center = e1*c2d[0] + e2*c2d[1]

    # 输出主轴向量（带长度）
    dir1_2 = np.array([np.cos(theta_out), np.sin(theta_out)])
    dir2_2 = np.array([-np.sin(theta_out), np.cos(theta_out)])
    axis_vec1_3 = a_out * (dir1_2[0]*e1 + dir1_2[1]*e2)
    axis_vec2_3 = b_out * (dir2_2[0]*e1 + dir2_2[1]*e2)

    return {
        "center": center,
        "axis_vec1_3": normalize(axis_vec1_3),
        "axis_vec2_3": normalize(axis_vec2_3),
        "scale1": a_out,
        "scale2": b_out
    }

if __name__ == "__main__":
    ell1 = (np.array([0.0,0.0,0.0]), normalize(np.array([1,0,0])), normalize(np.array([0,1,0])), 3.0, 1.0)
    ell2 = (np.array([4.0,0.2,0.0]), normalize(np.array([0.98,0.1,0])), normalize(np.array([-0.1,0.98,0])), 2.0, 0.8)

    res = merge_two_ellipses_simple_3d(ell1, ell2, m=128, safety_factor=1.01)
    ok, max_v = verify_coverage_simple(ell1, ell2, res)
    print("covered?", ok, "max value:", max_v)