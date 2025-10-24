import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import beta as beta_dist, norm, uniform
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple
import ast

st.set_page_config(page_title="2D Opinion ABM (Streamlit)", layout="wide")

# ---------------------------
# Data model
# ---------------------------

@dataclass
class ModelState:
    N: int
    K: int
    gamma: float
    R: int
    w1: float
    w2: float
    w3: float
    GSweight: Tuple[float, float]
    fixprop: float
    fixaut2: float
    sumage: Tuple[float, float]
    UL: float
    LL: float
    int1: np.ndarray
    int2: np.ndarray
    choice1: np.ndarray
    choice2: np.ndarray
    agealpha1: np.ndarray
    agebeta1: np.ndarray
    agealpha2: np.ndarray
    agebeta2: np.ndarray
    seed: int
    fix_idx: np.ndarray  # persistent stubborn-agent index (MATLAB's randperm equivalent)


# ---------------------------
# Utilities (ported from MATLAB)
# ---------------------------

def fit_agents(int_vals: np.ndarray, sumage: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each intrinsic value v in int_vals, find alpha in (1.001, sumage-1.001)
    minimizing |betainv(0.5, alpha, sumage-alpha) - v|.
    Returns vectors alphaVec, betaVec.
    """
    n = int_vals.size
    alpha_vec = np.zeros(n, dtype=float)
    beta_vec = np.zeros(n, dtype=float)

    for t, v in enumerate(int_vals):
        def obj(a):
            med = beta_dist.ppf(0.5, a, sumage - a)
            return abs(med - v)

        res = minimize_scalar(
            obj,
            bounds=(1.001, max(1.002, sumage - 1.001)),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 200}
        )
        aopt = float(res.x)
        alpha_vec[t] = aopt
        beta_vec[t] = sumage - aopt

    return alpha_vec, beta_vec


def bounded_fit(data: np.ndarray, UL: float, LL: float) -> Tuple[float, float]:
    """
    Moment-based Beta fit from mean+sd, then clamp (a+b) into [LL, UL] by scaling.
    Fallback to Beta(2,2) if insufficient data or degenerate variance.
    """
    data = np.asarray(data)
    if data.size >= 2:
        mu = float(np.mean(data))
        sd = float(np.std(data, ddof=0))
        mu = min(max(mu, 1e-6), 1 - 1e-6)  # avoid boundary
        if sd < 1e-10:
            aH, bH = 2.0, 2.0
        else:
            aH = ((1 - mu) * mu**2) / (sd**2) - mu
            bH = ((1 - mu) / mu) * aH
            if not np.isfinite(aH) or not np.isfinite(bH) or aH <= 0 or bH <= 0:
                aH, bH = 2.0, 2.0
    else:
        aH, bH = 2.0, 2.0

    aH = max(1.001, float(aH))
    bH = max(1.001, float(bH))

    s = aH + bH
    if s > UL:
        aH = max(1.001, UL * (aH / s))
        bH = max(1.001, UL * (bH / s))
    if s < LL:
        aH = max(1.001, LL * (aH / s))
        bH = max(1.001, LL * (bH / s))

    return float(aH), float(bH)


def _grid_x():
    # Equivalent to MATLAB 0:0.001:1 (1001 points)
    return np.linspace(0.0, 1.0, 1001, dtype=float)


def update_choice(i: int,
                  choice: np.ndarray,
                  in_idx: np.ndarray,
                  out_idx: np.ndarray,
                  a_vec: np.ndarray,
                  b_vec: np.ndarray,
                  gamma: float, UL: float, LL: float,
                  w1: float, w2: float, w3: float) -> float:
    """
    Discretize x in [0,1] and minimize Y = w1*y0 + w2*yi - w3*yo,
    where y(.) = exp(|gamma*(betacdf(.) - 0.5)|).
    """
    x = _grid_x()

    # Own authenticity
    y0 = np.exp(np.abs(gamma * (beta_dist.cdf(x, a_vec[i], b_vec[i]) - 0.5)))

    # In-group
    if in_idx.size > 0:
        ai, bi = bounded_fit(choice[in_idx], UL, LL)
    else:
        ai, bi = (2.0, 2.0)
    yi = np.exp(np.abs(gamma * (beta_dist.cdf(x, ai, bi) - 0.5)))

    # Out-group
    if out_idx.size > 0:
        ao, bo = bounded_fit(choice[out_idx], UL, LL)
        yo = np.exp(np.abs(gamma * (beta_dist.cdf(x, ao, bo) - 0.5)))
    else:
        yo = np.zeros_like(x)

    Y = w1 * y0 + w2 * yi - w3 * yo
    idx = int(np.argmin(Y))
    return float(x[idx])


def sample_groups(i: int, g1: np.ndarray, g2: np.ndarray, K: int, GSweight: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Replicates MATLAB logic:
    own1/other1 by group 1; own2/other2 by group 2; sample k1 = ceil(GS1*K), k2 = K-k1.
    Return unique indices for in-group and out-group.
    """
    own1 = np.where(g1 == g1[i])[0]
    other1 = np.where(g1 != g1[i])[0]
    own2 = np.where(g2 == g2[i])[0]
    other2 = np.where(g2 != g2[i])[0]

    k1 = int(np.ceil(GSweight[0] * K))
    k2 = int(K - k1)

    rng = np.random.default_rng()

    def _sample(arr, k):
        if arr.size == 0:
            return np.array([], dtype=int)
        if arr.size >= k:
            return rng.choice(arr, size=k, replace=False)
        else:
            return arr

    s1 = _sample(own1, k1)
    s2 = _sample(own2, k2)
    in_idx = np.unique(np.concatenate([s1, s2]))

    o1 = _sample(other1, k1)
    o2 = _sample(other2, k2)
    out_idx = np.unique(np.concatenate([o1, o2]))

    return in_idx, out_idx


# ---------------------------
# Plotting: single composite figure (top + right + center)
# ---------------------------

def plot_all(int1, int2, choice1, choice2, step: int = 0):
    """
    Single figure with three subplots arranged like MATLAB GUI:
      - Top: stacked bar for choice1 distribution by groups (left/right)
      - Right: horizontal stacked bar for choice2 distribution by groups
      - Center: scatter (choice1 vs choice2)
    """
    left_idx  = choice1 < 0.5
    right_idx = ~left_idx

    rng_bins = np.arange(0, 1.0001, 0.05)
    total = choice1.size

    # Histogram counts normalized to probability (like MATLAB 'Normalization','probability')
    yy1l, _ = np.histogram(choice1[left_idx],  bins=rng_bins)
    yy1r, _ = np.histogram(choice1[right_idx], bins=rng_bins)
    yy2l, _ = np.histogram(choice2[left_idx],  bins=rng_bins)
    yy2r, _ = np.histogram(choice2[right_idx], bins=rng_bins)

    yy1l = yy1l / max(1, total)
    yy1r = yy1r / max(1, total)
    yy2l = yy2l / max(1, total)
    yy2r = yy2r / max(1, total)

    # Figure & GridSpec (2x2 grid; use L-shape: [top, .; scatter, right])
    fig = plt.figure(figsize=(8, 7), layout="constrained")
    gs = GridSpec(
        nrows=2, ncols=2, figure=fig,
        height_ratios=[1.0, 4.0],
        width_ratios=[4.0, 1.0]
    )

    ax_top     = fig.add_subplot(gs[0, 0])          # top bar
    ax_scatter = fig.add_subplot(gs[1, 0])          # center scatter
    ax_right   = fig.add_subplot(gs[1, 1])          # right barh (do NOT share y to keep y-ticks on scatter)

    # Top stacked bar for choice1
    ax_top.bar(rng_bins[:-1], yy1l, width=0.05, align="edge")
    ax_top.bar(rng_bins[:-1], yy1r, width=0.05, align="edge", bottom=yy1l)
    ax_top.set_ylim(0, 0.5)
    ax_top.set_xlim(0, 1)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # Right horizontal stacked bar for choice2
    ax_right.barh(rng_bins[:-1], yy2l, height=0.05, align="edge")
    ax_right.barh(rng_bins[:-1], yy2r, height=0.05, align="edge", left=yy2l)
    ax_right.set_xlim(0, 0.5)
    ax_right.set_ylim(0, 1)       # align range with scatter y
    # Do NOT clear y-ticks on the scatter; only hide ticks on the right subplot itself:
    ax_right.tick_params(axis='y', left=False, labelleft=False)
    ax_right.set_xticks([])
    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # Center scatter
    ax_scatter.scatter(choice1[left_idx],  choice2[left_idx],  s=12)
    ax_scatter.scatter(choice1[right_idx], choice2[right_idx], s=12)
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_xlabel("Choice Dim1")
    ax_scatter.set_ylabel("Choice Dim2")
    if step == 0:
        ax_scatter.set_title("Initial System")
    else:
        ax_scatter.set_title(f"Expressed Value Distribution, Step = {step}")

    # Tighten gaps
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    return fig


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("2D Opinion ABM — Streamlit")

with st.sidebar:
    st.header("Parameters")

    N = st.number_input("# Agent (N)", min_value=100, max_value=5000, value=1000, step=50)
    K = st.selectbox("# Sample (K)", options=[10, 20, 30, 40], index=0)
    gamma = st.number_input("Sensitivity (γ)", min_value=1.0, value=10.0, step=1.0, format="%.3f")
    R = st.number_input("# Iterations (R)", min_value=1, value=20000, step=100)
    w1 = st.number_input("Weight_Aut (w1)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    rho = st.number_input("Out-group (rho)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    gs1 = st.number_input("Dim1 priority (m1)", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    fixprop = st.number_input("Stubborn prop", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    fixaut2 = st.number_input("Stubborn pref", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

    sumage_str = st.text_input("Certainty_AUT [a1, a2]", value="[10, 10]")
    norm_str = st.text_input("Certainty_Norm [UL, LL]", value="[20, 3]")

    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=12345, step=1)

    # Visualization controls
    redraw_every = st.slider("Frames: redraw every N steps", min_value=1, max_value=2000, value=max(1, int(R) // 200))
    max_steps_run = st.slider("Steps per click", min_value=100, max_value=int(R), value=min(int(R), 10000), step=100)

# Placeholder for the composite figure and progress
plot_placeholder = st.empty()
progress_placeholder = st.empty()

# ---------------------------
# Initialization & simulation
# ---------------------------

def initialize_model() -> ModelState | None:
    """Parse inputs, build initial state, and persist stubborn-agent index."""
    try:
        sumage_val = ast.literal_eval(sumage_str)  # safer than eval
        UL_LL = ast.literal_eval(norm_str)
    except Exception:
        st.error("Invalid input format. Please use e.g., [10, 10] and [20, 3].")
        return None

    if not (isinstance(sumage_val, (list, tuple)) and len(sumage_val) == 2):
        st.error("Certainty_AUT must be a length-2 vector, e.g., [10, 10].")
        return None
    if not (isinstance(UL_LL, (list, tuple)) and len(UL_LL) == 2):
        st.error("Certainty_Norm must be a length-2 vector, e.g., [20, 3].")
        return None

    UL, LL = float(UL_LL[0]), float(UL_LL[1])
    gs_weight = (float(gs1), 1.0 - float(gs1))
    w2 = (1 - w1) * (1 - rho)
    w3 = (1 - w1) * rho

    # Seeded RNG for reproducibility
    rng = np.random.default_rng(int(seed))

    # lim1/lim2 via Beta(1.001, sumage[0]-1.001) median symmetry around 0.5
    lim1 = beta_dist.ppf(0.5, 1.001, float(sumage_val[0]) - 1.001)
    lim2 = 1.0 - lim1

    # Bivariate normal -> probit -> uniform -> truncated uniform [lim1, lim2]
    Z = rng.normal(size=(int(N), 2))
    Uprob = norm.cdf(Z)
    X1 = uniform.ppf(Uprob[:, 0], loc=lim1, scale=(lim2 - lim1))
    X2 = uniform.ppf(Uprob[:, 1], loc=lim1, scale=(lim2 - lim1))

    int1 = X1.copy()
    int2 = X2.copy()

    # Stubborn agents on dimension 2: persist their indices
    n_fix = int(round(float(fixprop) * int(N)))
    if n_fix > 0:
        fix_idx = rng.choice(np.arange(int(N)), size=n_fix, replace=False)
        int2[fix_idx] = float(fixaut2)
    else:
        fix_idx = np.array([], dtype=int)

    # Per-agent authenticity betas
    agealpha1, agebeta1 = fit_agents(int1, float(sumage_val[0]))
    agealpha2, agebeta2 = fit_agents(int2, float(sumage_val[1]))

    ms = ModelState(
        N=int(N), K=int(K), gamma=float(gamma), R=int(R),
        w1=float(w1), w2=float(w2), w3=float(w3),
        GSweight=gs_weight, fixprop=float(fixprop), fixaut2=float(fixaut2),
        sumage=(float(sumage_val[0]), float(sumage_val[1])),
        UL=UL, LL=LL,
        int1=int1, int2=int2,
        choice1=int1.copy(), choice2=int2.copy(),
        agealpha1=agealpha1, agebeta1=agebeta1,
        agealpha2=agealpha2, agebeta2=agebeta2,
        seed=int(seed),
        fix_idx=fix_idx
    )
    return ms


def run_sim(ms: ModelState, steps_to_run: int, redraw_every: int):
    """Run simulation for a given number of steps with periodic redraws."""
    choice1 = ms.choice1
    choice2 = ms.choice2

    rng = np.random.default_rng(ms.seed)  # deterministic per session

    for step in range(1, steps_to_run + 1):
        i = int(rng.integers(ms.N))

        grp1 = choice1 < 0.5
        grp2 = choice2 < 0.5
        in_idx, out_idx = sample_groups(i, grp1, grp2, ms.K, ms.GSweight)

        # Update dimension 1
        choice1[i] = update_choice(i, choice1, in_idx, out_idx,
                                   ms.agealpha1, ms.agebeta1, ms.gamma, ms.UL, ms.LL, ms.w1, ms.w2, ms.w3)

        # Update dimension 2 (stubborn agents stay fixed)
        if i in ms.fix_idx:
            choice2[i] = ms.fixaut2
        else:
            choice2[i] = update_choice(i, choice2, in_idx, out_idx,
                                       ms.agealpha2, ms.agebeta2, ms.gamma, ms.UL, ms.LL, ms.w1, ms.w2, ms.w3)

        # Periodic redraw
        if (step % max(1, redraw_every) == 0) or (step == steps_to_run):
            fig = plot_all(ms.int1, ms.int2, choice1, choice2, step)
            plot_placeholder.pyplot(fig, clear_figure=True)
            progress_placeholder.progress(step / steps_to_run)

    # Persist changes
    ms.choice1 = choice1
    ms.choice2 = choice2
    return ms


# ---------------------------
# Buttons: fix the "double click needed" issue
# ---------------------------

col1, col2 = st.columns([1, 1])

# 1) Render & handle "Generate" first
init_btn = col1.button("Generate System", key="btn_gen")
if init_btn:
    ms = initialize_model()
    if ms is not None:
        st.session_state["model_state"] = ms
        st.session_state["initialized"] = True
        fig0 = plot_all(ms.int1, ms.int2, ms.choice1, ms.choice2, step=0)
        plot_placeholder.pyplot(fig0, clear_figure=True)
        progress_placeholder.empty()

# Ensure boolean flag is defined
initialized = bool(st.session_state.get("initialized", False))

# 2) Then render "Run" based on the latest state (enabled on first click already)
run_btn = col2.button("Run Simulation", key="btn_run", disabled=not initialized)

# 3) Handle "Run"
if run_btn and initialized:
    ms = st.session_state["model_state"]
    ms = run_sim(ms, steps_to_run=int(max_steps_run), redraw_every=int(redraw_every))
    st.session_state["model_state"] = ms

# ---------------------------
# Footer instructions
# ---------------------------

st.markdown(
    """
    ---
    **Instructions**  
    1. Set parameters in the sidebar and click **Generate System** to create the initial state.  
    2. Click **Run Simulation** to iterate and update the plots.  
    3. To keep iterating from the current state, click **Run Simulation** again.  
    """
)