"""
2D FLIP/PIC fluid solver on NVIDIA Warp.

MAC grid layout:
    - u velocity: (nx+1, ny)  at x-faces (cell left/right)
    - v velocity: (nx, ny+1)  at y-faces (cell bottom/top)
    - p pressure: (nx, ny)    at cell centers
    - cell type:  (nx, ny)    0=air, 1=fluid, 2=solid

Particles carry position (vec2 in world units) and velocity (vec2).

Pressure solver is Jacobi (simple, slow-converging, but correct).
Expect ~60-200 Jacobi iters for visually plausible incompressibility.

TODO next iteration:
    - PCG pressure solve (much faster than Jacobi)
    - Sub-stepping with CFL
    - Better particle reseeding to fix voids
    - APIC instead of FLIP/PIC for less noise
"""
import warp as wp
import numpy as np

wp.init()


# -----------------------------------------------------------------------------
# Warp kernels
# -----------------------------------------------------------------------------

@wp.func
def clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo: return lo
    if v > hi: return hi
    return v


@wp.kernel
def k_clear_grid(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    uw: wp.array2d(dtype=float),
    vw: wp.array2d(dtype=float),
    marker: wp.array2d(dtype=int),
):
    i, j = wp.tid()
    if i < u.shape[0] and j < u.shape[1]:
        u[i, j] = 0.0
        uw[i, j] = 0.0
    if i < v.shape[0] and j < v.shape[1]:
        v[i, j] = 0.0
        vw[i, j] = 0.0
    if i < marker.shape[0] and j < marker.shape[1]:
        # keep solid markers, reset others to air
        if marker[i, j] != 2:
            marker[i, j] = 0


@wp.kernel
def k_p2g(
    particles_pos: wp.array(dtype=wp.vec2),
    particles_vel: wp.array(dtype=wp.vec2),
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    uw: wp.array2d(dtype=float),
    vw: wp.array2d(dtype=float),
    marker: wp.array2d(dtype=int),
    dx: float,
    nx: int,
    ny: int,
):
    pid = wp.tid()
    p = particles_pos[pid]
    vel = particles_vel[pid]

    # Mark containing cell as fluid (unless solid).
    ci = clamp_int(int(p[0] / dx), 0, nx - 1)
    cj = clamp_int(int(p[1] / dx), 0, ny - 1)
    if marker[ci, cj] != 2:
        marker[ci, cj] = 1

    # ---------------- scatter to u (x-faces, sampled at (i*dx, (j+0.5)*dx)) -----------------
    fx = p[0] / dx
    fy = p[1] / dx - 0.5
    i0 = int(wp.floor(fx))
    j0 = int(wp.floor(fy))
    sx = fx - float(i0)
    sy = fy - float(j0)
    # 4 weights
    w00 = (1.0 - sx) * (1.0 - sy)
    w10 = sx * (1.0 - sy)
    w01 = (1.0 - sx) * sy
    w11 = sx * sy
    # bounds: u shape (nx+1, ny)
    if i0 >= 0 and i0 <= nx and j0 >= 0 and j0 < ny:
        wp.atomic_add(u, i0, j0, vel[0] * w00)
        wp.atomic_add(uw, i0, j0, w00)
    if i0 + 1 >= 0 and i0 + 1 <= nx and j0 >= 0 and j0 < ny:
        wp.atomic_add(u, i0 + 1, j0, vel[0] * w10)
        wp.atomic_add(uw, i0 + 1, j0, w10)
    if i0 >= 0 and i0 <= nx and j0 + 1 >= 0 and j0 + 1 < ny:
        wp.atomic_add(u, i0, j0 + 1, vel[0] * w01)
        wp.atomic_add(uw, i0, j0 + 1, w01)
    if i0 + 1 >= 0 and i0 + 1 <= nx and j0 + 1 >= 0 and j0 + 1 < ny:
        wp.atomic_add(u, i0 + 1, j0 + 1, vel[0] * w11)
        wp.atomic_add(uw, i0 + 1, j0 + 1, w11)

    # ---------------- scatter to v (y-faces, sampled at ((i+0.5)*dx, j*dx)) -----------------
    fx = p[0] / dx - 0.5
    fy = p[1] / dx
    i0 = int(wp.floor(fx))
    j0 = int(wp.floor(fy))
    sx = fx - float(i0)
    sy = fy - float(j0)
    w00 = (1.0 - sx) * (1.0 - sy)
    w10 = sx * (1.0 - sy)
    w01 = (1.0 - sx) * sy
    w11 = sx * sy
    # bounds: v shape (nx, ny+1)
    if i0 >= 0 and i0 < nx and j0 >= 0 and j0 <= ny:
        wp.atomic_add(v, i0, j0, vel[1] * w00)
        wp.atomic_add(vw, i0, j0, w00)
    if i0 + 1 >= 0 and i0 + 1 < nx and j0 >= 0 and j0 <= ny:
        wp.atomic_add(v, i0 + 1, j0, vel[1] * w10)
        wp.atomic_add(vw, i0 + 1, j0, w10)
    if i0 >= 0 and i0 < nx and j0 + 1 >= 0 and j0 + 1 <= ny:
        wp.atomic_add(v, i0, j0 + 1, vel[1] * w01)
        wp.atomic_add(vw, i0, j0 + 1, w01)
    if i0 + 1 >= 0 and i0 + 1 < nx and j0 + 1 >= 0 and j0 + 1 <= ny:
        wp.atomic_add(v, i0 + 1, j0 + 1, vel[1] * w11)
        wp.atomic_add(vw, i0 + 1, j0 + 1, w11)


@wp.kernel
def k_normalize_uv(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    uw: wp.array2d(dtype=float),
    vw: wp.array2d(dtype=float),
    u_saved: wp.array2d(dtype=float),
    v_saved: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if i < u.shape[0] and j < u.shape[1]:
        if uw[i, j] > 1.0e-8:
            u[i, j] = u[i, j] / uw[i, j]
        else:
            u[i, j] = 0.0
        u_saved[i, j] = u[i, j]
    if i < v.shape[0] and j < v.shape[1]:
        if vw[i, j] > 1.0e-8:
            v[i, j] = v[i, j] / vw[i, j]
        else:
            v[i, j] = 0.0
        v_saved[i, j] = v[i, j]


@wp.kernel
def k_add_gravity(
    v: wp.array2d(dtype=float),
    g: float,
    dt: float,
):
    i, j = wp.tid()
    if i < v.shape[0] and j < v.shape[1]:
        v[i, j] = v[i, j] + g * dt


@wp.kernel
def k_enforce_solid_bc(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    marker: wp.array2d(dtype=int),
    nx: int,
    ny: int,
):
    i, j = wp.tid()
    # zero normal velocities at solid faces
    # u face between (i-1,j) and (i,j); v face between (i,j-1) and (i,j)
    if i <= nx and j < ny:
        left_solid = (i == 0) or (i > 0 and marker[i - 1, j] == 2)
        right_solid = (i == nx) or (i < nx and marker[i, j] == 2)
        if left_solid or right_solid:
            u[i, j] = 0.0
    if i < nx and j <= ny:
        bot_solid = (j == 0) or (j > 0 and marker[i, j - 1] == 2)
        top_solid = (j == ny) or (j < ny and marker[i, j] == 2)
        if bot_solid or top_solid:
            v[i, j] = 0.0


@wp.kernel
def k_jacobi_pressure(
    p_in: wp.array2d(dtype=float),
    p_out: wp.array2d(dtype=float),
    div: wp.array2d(dtype=float),
    marker: wp.array2d(dtype=int),
):
    i, j = wp.tid()
    nx = p_in.shape[0]
    ny = p_in.shape[1]
    if i >= nx or j >= ny:
        return
    if marker[i, j] != 1:
        p_out[i, j] = 0.0
        return
    # neighbours: air (=0) → p=0, solid (=2) → skip (Neumann), fluid → use current
    pl = float(0.0); pr = float(0.0); pb = float(0.0); pt = float(0.0)
    diag = float(0.0)
    if i > 0:
        m = marker[i - 1, j]
        if m != 2:
            diag = diag + 1.0
            if m == 1: pl = p_in[i - 1, j]
    else:
        # outside domain treated as solid → Neumann
        pass
    if i < nx - 1:
        m = marker[i + 1, j]
        if m != 2:
            diag = diag + 1.0
            if m == 1: pr = p_in[i + 1, j]
    if j > 0:
        m = marker[i, j - 1]
        if m != 2:
            diag = diag + 1.0
            if m == 1: pb = p_in[i, j - 1]
    if j < ny - 1:
        m = marker[i, j + 1]
        if m != 2:
            diag = diag + 1.0
            if m == 1: pt = p_in[i, j + 1]
    if diag < 0.5:
        p_out[i, j] = 0.0
        return
    p_out[i, j] = (pl + pr + pb + pt - div[i, j]) / diag


@wp.kernel
def k_compute_divergence(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    div: wp.array2d(dtype=float),
    marker: wp.array2d(dtype=int),
    dx: float,
    dt: float,
    rho: float,
):
    i, j = wp.tid()
    if i >= div.shape[0] or j >= div.shape[1]:
        return
    if marker[i, j] != 1:
        div[i, j] = 0.0
        return
    # rhs for Poisson eq: Lap(p) = (rho/dt)*div(u)
    # discretized as: (sum_nb - diag*p) / dx^2 = (rho/dt) * (u_diff + v_diff)/dx
    # → Jacobi:  p = (sum_nb - rhs) / diag   where rhs = (rho*dx/dt)*(u_diff + v_diff)
    d = (u[i + 1, j] - u[i, j] + v[i, j + 1] - v[i, j])
    div[i, j] = d * (rho * dx / dt)


@wp.kernel
def k_subtract_pressure_grad(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    p: wp.array2d(dtype=float),
    marker: wp.array2d(dtype=int),
    dx: float,
    dt: float,
    rho: float,
):
    i, j = wp.tid()
    nx = marker.shape[0]
    ny = marker.shape[1]
    scale = dt / (rho * dx)
    # u face at (i,j): between cells (i-1,j) and (i,j)
    if i >= 1 and i <= nx - 1 and j < ny:
        m_left = marker[i - 1, j]
        m_right = marker[i, j]
        if m_left == 1 or m_right == 1:
            if m_left != 2 and m_right != 2:
                pl = float(0.0); pr = float(0.0)
                if m_left == 1: pl = p[i - 1, j]
                if m_right == 1: pr = p[i, j]
                u[i, j] = u[i, j] - scale * (pr - pl)
    # v face at (i,j): between cells (i,j-1) and (i,j)
    if j >= 1 and j <= ny - 1 and i < nx:
        m_bot = marker[i, j - 1]
        m_top = marker[i, j]
        if m_bot == 1 or m_top == 1:
            if m_bot != 2 and m_top != 2:
                pb = float(0.0); pt = float(0.0)
                if m_bot == 1: pb = p[i, j - 1]
                if m_top == 1: pt = p[i, j]
                v[i, j] = v[i, j] - scale * (pt - pb)


@wp.func
def sample_u(u: wp.array2d(dtype=float), px: float, py: float, dx: float, nx: int, ny: int) -> float:
    fx = px / dx
    fy = py / dx - 0.5
    i0 = clamp_int(int(wp.floor(fx)), 0, nx)
    j0 = clamp_int(int(wp.floor(fy)), 0, ny - 1)
    i1 = clamp_int(i0 + 1, 0, nx)
    j1 = clamp_int(j0 + 1, 0, ny - 1)
    sx = fx - float(i0)
    sy = fy - float(j0)
    sx = wp.clamp(sx, 0.0, 1.0)
    sy = wp.clamp(sy, 0.0, 1.0)
    u00 = u[i0, j0]; u10 = u[i1, j0]; u01 = u[i0, j1]; u11 = u[i1, j1]
    return (1.0 - sx) * (1.0 - sy) * u00 + sx * (1.0 - sy) * u10 + (1.0 - sx) * sy * u01 + sx * sy * u11


@wp.func
def sample_v(v: wp.array2d(dtype=float), px: float, py: float, dx: float, nx: int, ny: int) -> float:
    fx = px / dx - 0.5
    fy = py / dx
    i0 = clamp_int(int(wp.floor(fx)), 0, nx - 1)
    j0 = clamp_int(int(wp.floor(fy)), 0, ny)
    i1 = clamp_int(i0 + 1, 0, nx - 1)
    j1 = clamp_int(j0 + 1, 0, ny)
    sx = fx - float(i0)
    sy = fy - float(j0)
    sx = wp.clamp(sx, 0.0, 1.0)
    sy = wp.clamp(sy, 0.0, 1.0)
    v00 = v[i0, j0]; v10 = v[i1, j0]; v01 = v[i0, j1]; v11 = v[i1, j1]
    return (1.0 - sx) * (1.0 - sy) * v00 + sx * (1.0 - sy) * v10 + (1.0 - sx) * sy * v01 + sx * sy * v11


@wp.kernel
def k_g2p_and_advect(
    pos: wp.array(dtype=wp.vec2),
    vel: wp.array(dtype=wp.vec2),
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    u_saved: wp.array2d(dtype=float),
    v_saved: wp.array2d(dtype=float),
    dx: float,
    dt: float,
    nx: int,
    ny: int,
    flip_blend: float,
    domain_size: wp.vec2,
):
    pid = wp.tid()
    p = pos[pid]
    old_v = vel[pid]
    # PIC: sample fresh grid; FLIP: old particle vel + delta
    pic_u = sample_u(u, p[0], p[1], dx, nx, ny)
    pic_v = sample_v(v, p[0], p[1], dx, nx, ny)
    du = pic_u - sample_u(u_saved, p[0], p[1], dx, nx, ny)
    dv = pic_v - sample_v(v_saved, p[0], p[1], dx, nx, ny)
    flip_u = old_v[0] + du
    flip_v = old_v[1] + dv
    new_u = flip_blend * flip_u + (1.0 - flip_blend) * pic_u
    new_v = flip_blend * flip_v + (1.0 - flip_blend) * pic_v
    vel[pid] = wp.vec2(new_u, new_v)
    # advect with new velocity (semi-implicit Euler)
    new_pos = p + wp.vec2(new_u, new_v) * dt
    # clamp into open domain leaving a 1-cell solid wall buffer
    eps = dx * 1.001
    new_pos[0] = wp.clamp(new_pos[0], eps, domain_size[0] - eps)
    new_pos[1] = wp.clamp(new_pos[1], eps, domain_size[1] - eps)
    pos[pid] = new_pos


# -----------------------------------------------------------------------------
# Solver class
# -----------------------------------------------------------------------------

class FlipSolver2D:
    def __init__(self, nx=64, ny=64, dx=None, gravity=-9.81, flip_blend=0.95, rho=1.0):
        self.nx = nx
        self.ny = ny
        self.dx = dx if dx is not None else 1.0 / nx
        self.domain_size = wp.vec2(self.nx * self.dx, self.ny * self.dx)
        self.gravity = gravity
        self.flip_blend = flip_blend
        self.rho = rho
        self.device = "cuda:0"

        # Grid arrays
        self.u = wp.zeros((nx + 1, ny), dtype=float, device=self.device)
        self.v = wp.zeros((nx, ny + 1), dtype=float, device=self.device)
        self.uw = wp.zeros((nx + 1, ny), dtype=float, device=self.device)
        self.vw = wp.zeros((nx, ny + 1), dtype=float, device=self.device)
        self.u_saved = wp.zeros((nx + 1, ny), dtype=float, device=self.device)
        self.v_saved = wp.zeros((nx, ny + 1), dtype=float, device=self.device)
        self.p = wp.zeros((nx, ny), dtype=float, device=self.device)
        self.p_tmp = wp.zeros((nx, ny), dtype=float, device=self.device)
        self.div = wp.zeros((nx, ny), dtype=float, device=self.device)
        self.marker = wp.zeros((nx, ny), dtype=int, device=self.device)

        # set boundary cells to solid
        marker_host = np.zeros((nx, ny), dtype=np.int32)
        marker_host[0, :] = 2
        marker_host[-1, :] = 2
        marker_host[:, 0] = 2
        marker_host[:, -1] = 2
        wp.copy(self.marker, wp.array(marker_host, dtype=int, device=self.device))

        # particles
        self.pos = None
        self.vel = None
        self.n_particles = 0

    def seed_box(self, x0, y0, x1, y1, ppc=4):
        """Seed particles uniformly in axis-aligned rect [x0,y0]x[x1,y1] world units."""
        i0 = int(x0 / self.dx); i1 = int(x1 / self.dx)
        j0 = int(y0 / self.dx); j1 = int(y1 / self.dx)
        per_axis = int(np.sqrt(ppc))  # ppc=4 → 2x2 per cell
        positions = []
        for i in range(i0, i1):
            for j in range(j0, j1):
                for sx in range(per_axis):
                    for sy in range(per_axis):
                        # jittered placement
                        px = (i + (sx + 0.5 + 0.3 * (np.random.rand() - 0.5)) / per_axis) * self.dx
                        py = (j + (sy + 0.5 + 0.3 * (np.random.rand() - 0.5)) / per_axis) * self.dx
                        positions.append([px, py])
        positions = np.array(positions, dtype=np.float32)
        velocities = np.zeros_like(positions)
        self.pos = wp.array(positions, dtype=wp.vec2, device=self.device)
        self.vel = wp.array(velocities, dtype=wp.vec2, device=self.device)
        self.n_particles = len(positions)

    def step(self, dt, pressure_iters=80):
        nx, ny, dx = self.nx, self.ny, self.dx
        # clear grid (keep solids)
        wp.launch(k_clear_grid, dim=(max(nx + 1, nx), max(ny + 1, ny)),
                  inputs=[self.u, self.v, self.uw, self.vw, self.marker], device=self.device)
        # P2G
        wp.launch(k_p2g, dim=self.n_particles,
                  inputs=[self.pos, self.vel, self.u, self.v, self.uw, self.vw, self.marker, dx, nx, ny],
                  device=self.device)
        # normalize and save pre-pressure grid for FLIP
        wp.launch(k_normalize_uv, dim=(max(nx + 1, nx), max(ny + 1, ny)),
                  inputs=[self.u, self.v, self.uw, self.vw, self.u_saved, self.v_saved], device=self.device)
        # gravity (on v only)
        wp.launch(k_add_gravity, dim=(nx, ny + 1),
                  inputs=[self.v, self.gravity, dt], device=self.device)
        # solid BCs
        wp.launch(k_enforce_solid_bc, dim=(nx + 1, ny + 1),
                  inputs=[self.u, self.v, self.marker, nx, ny], device=self.device)
        # divergence
        wp.launch(k_compute_divergence, dim=(nx, ny),
                  inputs=[self.u, self.v, self.div, self.marker, dx, dt, self.rho], device=self.device)
        # Jacobi pressure
        self.p.zero_()
        self.p_tmp.zero_()
        for _ in range(pressure_iters):
            wp.launch(k_jacobi_pressure, dim=(nx, ny),
                      inputs=[self.p, self.p_tmp, self.div, self.marker], device=self.device)
            self.p, self.p_tmp = self.p_tmp, self.p
        # subtract gradient
        wp.launch(k_subtract_pressure_grad, dim=(nx + 1, ny + 1),
                  inputs=[self.u, self.v, self.p, self.marker, dx, dt, self.rho], device=self.device)
        # solid BCs again
        wp.launch(k_enforce_solid_bc, dim=(nx + 1, ny + 1),
                  inputs=[self.u, self.v, self.marker, nx, ny], device=self.device)
        # G2P + advect
        # update u_saved/v_saved deltas already captured pre-pressure — but FLIP needs delta = post-pressure - pre-pressure.
        # Our u_saved holds pre-pressure values. After pressure step current u is post-pressure. Good.
        wp.launch(k_g2p_and_advect, dim=self.n_particles,
                  inputs=[self.pos, self.vel, self.u, self.v, self.u_saved, self.v_saved,
                          dx, dt, nx, ny, self.flip_blend, self.domain_size],
                  device=self.device)
        wp.synchronize()

    def get_particles(self):
        return self.pos.numpy(), self.vel.numpy()
