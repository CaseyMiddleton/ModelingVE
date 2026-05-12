import numpy as np
import math
from random import sample
from scipy.integrate import solve_ivp
import warnings
from scipy.optimize import root_scalar, fsolve
from scipy.stats import beta as beta_dist, norm as norm_dist
from scipy.stats import chi2

def initialize_SIR(N,init_inf=1):
    # Initialize SIR model with init_inf infected unvaccinated individuals
    return {"S": N-init_inf, "I": init_inf, "R": 0}

def get_herd_immunity_threshold(R0,VE):
    return (1 - (1/R0))/VE

def get_herd_immunity_VE(R0,vax_rate):
    return (1 - (1/R0))/vax_rate

def get_herd_immunity_R0(vax_rate,VE):
    return 1 / (1-vax_rate*VE)

# Functions to properly parameterize leaky model to produce VE (see Shim & Galvani (2012))
def equations(vars, beta, tau, f, sigma):
    '''
    Parameters:
        beta : per-contact transmission hazard
        tau : infectious duration
        f : vaccinated fraction
        sigma : vaccine effect parameter
    '''
    Omega_u, Omega_v = vars
    eq1 = Omega_u - (1 - np.exp(-beta * tau * ((1-f)*Omega_u + f*Omega_v)))
    eq2 = Omega_v - (1 - np.exp(-beta * (1 - sigma) * tau * ((1-f)*Omega_u + f*Omega_v)))
    return [eq1, eq2]

def final_size_guess(R0, f, sigma, a=0.5):

    if f >= get_herd_immunity_threshold(R0,sigma):
        return 0.0, 0.0

    Re = R0 * (1 - f * sigma)
    Ztot = 1 - np.exp(-a * (1-1/Re))

    AR_u = 1 - np.exp(-R0 * Ztot)
    AR_v = 1 - np.exp(-(1 - sigma) * R0 * Ztot)

    return AR_u, AR_v

def solve_attack_rates(beta, tau, f, sigma):
    """ Solve per-group final size equations for attack rates. 
    
    Returns:
        omega_u: total proportion of the unvaccinated that are infected
        omega_v: total proportion of the vaccinated that are infected
    """
    R0 = beta*tau
    if sigma == 1 or f == 1:
        a_guess = R0
    else:
        a_guess = (1-f)*(1-sigma)*R0*50
    guess = final_size_guess(R0,f,sigma,a=a_guess)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The iteration is not making good progress')
        sol = fsolve(equations, guess, args=(beta, tau, f, sigma), full_output=True)

    Omega_u, Omega_v = sol[0]
    
    # Ensure non-negative
    Omega_u = max(0, Omega_u)
    Omega_v = max(0, Omega_v)

    return Omega_u, Omega_v

def compute_VE(Omega_u, Omega_v):
    if Omega_u >= 1e-10:
        return 1 - (Omega_v / Omega_u)
    else:
        # Divide by 0 error: sigma solver requires this to be 1
        return 1
    
def solve_sigma_for_target_VE(beta, tau, f, target_VE):
    """
    Find the value of sigma that yields the target vaccine effectiveness (VE).

    Parameters
    ----------
    beta : float
        Transmission rate
    tau : float
        Recovery time
    f : float
        Vaccination coverage (proportion vaccinated)
    target_VE : float
        Goal VE at end of simulation

    Returns
    -------
    float
        Sigma value that produces target_VE, constrained such that sigma >= target_VE

    Notes
    -----
    Uses root_scalar to find sigma such that VE(sigma) - target_VE = 0.
    The result is constrained to sigma >= target_VE to avoid counterintuitive
    results in herd immunity regimes where indirect effects dominate.
    """
    def VE_difference(sigma):
        Omega_u, Omega_v = solve_attack_rates(beta, tau, f, sigma)
        VE = compute_VE(Omega_u, Omega_v)
        return VE - target_VE

    # Constrain lower bound to be at least target_VE
    lower_bound = max(0, target_VE)
    # Constrain upper bound within herd immunity threshold for a given vax rate
    HIT_VE = get_herd_immunity_VE(beta*tau,f)
    if HIT_VE <= target_VE:
        print("Herd immunity effects prevent target VE from being achieved when R0={} and f={}.\n Returning VE={}".format(beta*tau,f,target_VE))
        return target_VE
    upper_bound = min(1, HIT_VE)

    # Root finding for sigma in constrained bounds
    result = root_scalar(VE_difference, bracket=(lower_bound, upper_bound), method='bisect')

    if not result.converged:
        raise RuntimeError("Sigma root finding did not converge.")

    # Get solution
    sigma_star = result.root
    # Ensure solution is greater than target_VE
    sigma_star = max(sigma_star, target_VE)

    return sigma_star


def SIRV_model(t, y, beta, gamma, vax_type, VE, vax_prop, leaky_correction):
    """
    Define SIR ODE with vaccination

    Parameters
    ----------
    t : scalar
        Time.
    y : array
        Current values of S, I, R, VS, VI, VR, V.
    beta : scalar
        Transmission rate.
    gamma : scalar
        Recovery rate. 
    vax_type : str
        Either 'leaky' or 'all-or-nothing'.
    VE : str
        Vaccine effectiveness (i.e. reduction in infection probability due to vaccination)
    vax_prop : float
        Proportion of population vaccinated

    Returns
    -------
    results : array
        Array with updated value for all compartments.
    """

    S, I, R, VS, VI, VR, V = np.maximum(y, 0) # enforce non-negativity to avoid numerical integration errors
    N = S + I + R + VS + VI + VR + V

    # Infection force
    lambda_inf = beta * (I + VI) / N

    # Differential equations
    dS  = -lambda_inf * S
    dI  = lambda_inf * S - gamma * I
    dR  = gamma * I

    if vax_type == "leaky":
        if leaky_correction:
            alpha = solve_sigma_for_target_VE(beta, 1/gamma, vax_prop, VE)
        else:
            alpha = VE
        # All vaccinated have reduced susceptibility
        dVS = -(1 - alpha) * lambda_inf * VS
        dVI = (1 - alpha) * lambda_inf * VS - gamma * VI
        dVR = gamma * VI
        dV  = 0  # fully protected compartment stays constant
    elif vax_type == "all-or-nothing":
        # Vaccinated split between fully protected and unprotected
        dVS = -lambda_inf * VS
        dVI = lambda_inf * VS - gamma * VI
        dVR = gamma * VI
        dV  = 0  # fully protected remain constant
    else:
        raise ValueError("vax_type must be 'leaky' or 'all-or-nothing'")

    return [dS, dI, dR, dVS, dVI, dVR, dV]


def run_SIRV(
    tmax=160,
    dt=0.1,
    inits=None,
    params=None,
    vax_type="leaky",
    leaky_correction=True
):
    """
    Runs an SIR model with vaccination (leaky or all-or-nothing).

    Parameters
    ----------
    tmax : float
        Maximum simulation time.
    dt : float
        Time step for output resolution (e.g. 0.1 days).
    inits : dict
        Initial fractions for unvaccinated S, I, R.
    params : dict
        Model parameters including:
            beta      : transmission rate
            gamma     : recovery rate
            vax_prop  : proportion vaccinated at t=0
            VE        : vaccine efficacy
    vax_type : str
        Either 'leaky' or 'all-or-nothing'.

    Returns
    -------
    results : dict
        Dictionary with arrays for time and all compartments.
    """
    if inits is None:
        inits = {"S": 0.9999, "I": 0.0001, "R": 0.0}
    if params is None:
        params = {"beta": 0.3, "gamma": 1/10, "vax_prop": 0.6, "VE": 0.8}

    beta, gamma, vax_prop, VE = (
        params["beta"],
        params["gamma"],
        params["vax_prop"],
        params["VE"],
    )

    # Initial conditions based on vaccination type
    if vax_type == "leaky":
        S0 = (1 - vax_prop) * inits["S"]
        VS0 = vax_prop * inits["S"]
        V0 = 0.0
    elif vax_type == "all-or-nothing":
        S0 = (1 - vax_prop) * inits["S"]
        VS0 = (1 - VE) * vax_prop * inits["S"]
        V0 = VE * vax_prop * inits["S"]
    else:
        raise ValueError("vax_type must be 'leaky' or 'all-or-nothing'")

    I0 = inits["I"]
    R0 = inits["R"]
    VI0 = VR0 = 0.0

    y0 = [S0, I0, R0, VS0, VI0, VR0, V0]

    # Create time grid based on dt
    t_eval = np.arange(0, tmax + dt, dt)

    # Solve ODE
    sol = solve_ivp(
        SIRV_model,
        [0, tmax],
        y0,
        args=(beta, gamma, vax_type, VE, vax_prop,leaky_correction),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
        max_step=1.0,
        vectorized=False
    )

    # Return solution as dict
    results = {
        "t": sol.t,
        "S": sol.y[0],
        "I": sol.y[1],
        "R": sol.y[2],
        "VS": sol.y[3],
        "VI": sol.y[4],
        "VR": sol.y[5],
        "V": sol.y[6],
    }

    # Print informational message if epidemic not run to completeness
    if results["I"][-1] > 10 or results["VI"][-1] > 10:
        print("Epidemic not yet complete. {} infections remaining.".format(results["I"][-1] + results["VI"][-1]))

    return results


#############################################################
##### Extending to heterogeneous model ##############
#############################################################

def generate_vaccine_responder_classes(
    n_classes,
    mean_VE=None,
    concentration=10,
    skew=0
):
    """
    Generate vaccine responder classes by slicing a Beta distribution.

    You can parameterize this in two ways:
    1. Specify mean_VE and concentration (recommended)
    2. Specify a and b directly

    Parameters
    ----------
    n_classes : int
        Number of responder classes.
    mean_VE : float, optional
        Mean vaccine effectiveness (0 to 1). If specified, overrides a and b.
        Higher mean = more effective vaccine on average.
    concentration : float, default=10
        Concentration parameter controlling the spread around mean_VE.
        Higher = more concentrated (less heterogeneity).
        Lower = more spread out (more heterogeneity).
        Only used if mean_VE is specified.
    skew : float, default=0
        Skewness adjustment (-1 to 1). Positive = right skew, negative = left skew.
        Only used if mean_VE is specified.

    Returns
    -------
    proportions : np.ndarray
        Length-n_classes array giving the proportion of vaccinated individuals
        in each responder class. Sums to 1.
    bin_midpoints : np.ndarray
        VE level for each responder class.

    Notes
    -----
    When using mean_VE parameterization:
    - For symmetric distribution: a = b = concentration * mean_VE
    - Skewness shifts the distribution left or right
    For leaky vaccine, ensure K = 51 or 101

    Examples
    --------
    >>> # Normal-like distribution centered at 0.7
    >>> props, VEs = generate_vaccine_responder_classes(10, mean_VE=0.7)

    >>> # More heterogeneity (lower concentration)
    >>> props, VEs = generate_vaccine_responder_classes(10, mean_VE=0.7, concentration=5)

    >>> # Right-skewed distribution
    >>> props, VEs = generate_vaccine_responder_classes(10, mean_VE=0.7, skew=0.5)
    """

    # Determine Beta parameters
    if mean_VE is not None:
        # Parameterize by mean and concentration
        if not 0 <= mean_VE <= 1:
            raise ValueError("mean_VE must be between 0 and 1")

        # Add skew adjustment
        # Positive skew increases a relative to b (right tail)
        # Negative skew increases b relative to a (left tail)
        skew_factor = 1 + skew

        a = mean_VE * concentration * skew_factor
        b = (1 - mean_VE) * concentration / skew_factor

    else:
        # Default to symmetric, moderately concentrated
        a = 5
        b = 5

    # Create bins with 0 and 1 as explicit endpoints
    # First bin: [0, 0.025] with midpoint at 0
    # Last bin: [0.975, 1] with midpoint at 1
    # Middle bins: equally spaced between
    edge_bin_width = min(0.025,1/n_classes)  # Width of edge bins

    # Bin edges: 0, 0.025, ..., 0.975, 1
    # First edge at 0, then edge_bin_width, then equally spaced to 1-edge_bin_width, then 1
    if n_classes == 2:
        # Special case: just 0 and 1
        bin_edges = np.array([0, 1])
        bin_midpoints = np.array([0, 1])
    else:
        # Create interior edges equally spaced between edge_bin_width and 1-edge_bin_width
        interior_edges = np.linspace(edge_bin_width, 1 - edge_bin_width, n_classes - 1)
        bin_edges = np.concatenate([[0], interior_edges, [1]])

        # Midpoints: 0 for first bin, 1 for last bin, actual midpoints for interior
        bin_midpoints = np.zeros(n_classes)
        bin_midpoints[0] = 0  # First bin midpoint at 0
        bin_midpoints[-1] = 1  # Last bin midpoint at 1
        # Interior midpoints
        for i in range(1, n_classes - 1):
            bin_midpoints[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

    # Evaluate CDF at bin edges to get cumulative probabilities.
    c_total = a + b
    actual_mean = a / c_total
    actual_sigma = np.sqrt(a * b / (c_total**2 * (c_total + 1)))
    approx_bin_width = bin_edges[1] - bin_edges[0]

    if actual_sigma < approx_bin_width * 0.01:
        # Effectively a point mass: assign all probability to the single bin
        # containing actual_mean. If actual_mean falls exactly on an edge,
        # assign to the bin to the right (half-open [left, right) convention).
        proportions = np.zeros(n_classes)
        bin_idx = int(np.searchsorted(bin_edges[1:], actual_mean, side='right'))
        bin_idx = min(bin_idx, n_classes - 1)
        proportions[bin_idx] = 1.0
    elif c_total <= 0.01:
        # Bernoulli approximation: Beta(a,b) with a+b<<1 degenerates to a two-point
        # distribution with mass (1-mean_VE) at 0 and mean_VE at 1.
        # Avoids scipy precision failures (_beta_cdf divide-by-zero) for tiny a, b.
        proportions = np.zeros(n_classes)
        proportions[0] = 1.0 - actual_mean   # mass at VE=0 bin
        proportions[-1] = actual_mean         # mass at VE=1 bin
    elif c_total >= 10000:
        # Point mass at actual_mean: snap the nearest bin midpoint to actual_mean
        # and assign all probability there.
        proportions = np.zeros(n_classes)
        nearest_idx = int(np.argmin(np.abs(bin_midpoints - actual_mean)))
        bin_midpoints[nearest_idx] = actual_mean
        proportions[nearest_idx] = 1.0
    else:
        cdf_at_edges = beta_dist.cdf(bin_edges, a, b)
        proportions = np.diff(cdf_at_edges)

    # Numerical safety: renormalize
    proportions = proportions / proportions.sum()

    return proportions, bin_midpoints

def get_mean_VE_heterogeneous(proportions,VE_levels):
    return np.average(VE_levels,weights=proportions)

def equations_heterogeneous(vars, beta, tau, v_dist_total, sigma_levels):
    '''
    Final size equations for heterogeneous model with K susceptibility classes.

    Parameters
    ----------
    vars : array
        [Omega_u, Omega_v_0, Omega_v_1, ..., Omega_v_{K-1}]
        Attack rates for unvaccinated and each vaccinated class.
    beta : float
        Per-contact transmission hazard.
    tau : float
        Infectious duration.
    v_dist_total : array of length K
        Proportion of total population in each vaccinated class.
        Sum of v_dist_total equals total vaccination coverage f.
    sigma_levels : array of length K
        VE (sigma) for each vaccinated class.

    Returns
    -------
    list
        Residuals for each equation.
    '''
    K = len(sigma_levels)
    Omega_u = vars[0]
    Omega_v = vars[1:1+K]  # Attack rates for each vaccinated class

    # Total vaccination coverage
    f_total = np.sum(v_dist_total)

    # Weighted average of all infected: unvaccinated + sum of all vaccinated classes
    # Each class contributes proportional to its population fraction
    weighted_infections = (1 - f_total) * Omega_u + np.sum(v_dist_total * Omega_v)

    # Equation for unvaccinated
    eq_u = Omega_u - (1 - np.exp(-beta * tau * weighted_infections))

    # Equations for each vaccinated class
    eq_v = []
    for i in range(K):
        eq_v_i = Omega_v[i] - (1 - np.exp(-beta * (1 - sigma_levels[i]) * tau * weighted_infections))
        eq_v.append(eq_v_i)

    return [eq_u] + eq_v


def final_size_guess_heterogeneous(R0, v_dist_total, sigma_levels, a=0.5):
    '''
    Compute an approximate initial guess for the heterogeneous final size equations,
    mirroring the leaky model's final_size_guess.

    Parameters
    ----------
    R0 : float
        Basic reproduction number.
    v_dist_total : array of length K
        Proportion of total population in each vaccinated class.
    sigma_levels : array of length K
        VE (sigma) for each vaccinated class.
    a : float
        Scaling parameter for the total attack rate approximation.

    Returns
    -------
    AR_u : float
        Approximate unvaccinated attack rate.
    AR_v : array of length K
        Approximate attack rates for each vaccinated class.
    '''
    K = len(sigma_levels)
    f_total = np.sum(v_dist_total)

    # Weighted average sigma
    if f_total > 0:
        v_weights = v_dist_total / f_total
        sigma_avg = np.sum(v_weights * sigma_levels)
    else:
        sigma_avg = 0

    # Check HIT
    if f_total >= get_herd_immunity_threshold(R0, sigma_avg):
        return 0.0, np.zeros(K)

    # Effective reproduction number
    Re = R0 * (1 - f_total * sigma_avg)
    Ztot = 1 - np.exp(-a * (1 - 1/Re))

    # Per-group attack rates
    AR_u = 1 - np.exp(-R0 * Ztot)
    AR_v = np.array([1 - np.exp(-(1 - sigma_levels[i]) * R0 * Ztot) for i in range(K)])

    return AR_u, AR_v


def solve_attack_rates_heterogeneous(beta, tau, v_dist_total, sigma_levels):
    '''
    Solve for attack rates in heterogeneous model with K susceptibility classes.

    Parameters
    ----------
    beta : float
        Transmission rate.
    tau : float
        Infectious duration.
    v_dist_total : array of length K
        Proportion of total population in each vaccinated class.
    sigma_levels : array of length K
        VE (sigma) for each vaccinated class.

    Returns
    -------
    Omega_u : float
        Attack rate for unvaccinated.
    Omega_v : array of length K
        Attack rates for each vaccinated class.
    '''
    K = len(sigma_levels)
    f_total = np.sum(v_dist_total)
    R0 = beta * tau

    # Weighted average sigma for guess scaling
    if f_total > 0:
        v_weights = v_dist_total / f_total
        sigma_avg = np.sum(v_weights * sigma_levels)
    else:
        sigma_avg = 0

    if sigma_avg == 1 or f_total == 1:
        a_guess = R0
    else:
        a_guess = (1 - f_total) * (1 - sigma_avg) * R0 * 10

    AR_u_guess, AR_v_guess = final_size_guess_heterogeneous(R0, v_dist_total, sigma_levels, a=a_guess)
    guess = [AR_u_guess] + list(AR_v_guess)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The iteration is not making good progress')
        sol = fsolve(equations_heterogeneous, guess, args=(beta, tau, v_dist_total, sigma_levels))

    Omega_u = max(0, sol[0])
    Omega_v = np.maximum(0, sol[1:1+K])

    return Omega_u, Omega_v


def compute_VE_heterogeneous(Omega_u, Omega_v, v_dist_total):
    '''
    Compute overall VE from attack rates in heterogeneous model.

    Parameters
    ----------
    Omega_u : float
        Attack rate for unvaccinated.
    Omega_v : array of length K
        Attack rates for each vaccinated class.
    v_dist_total : array of length K
        Proportion of total population in each vaccinated class.

    Returns
    -------
    float
        Overall vaccine effectiveness.
    '''
    if Omega_u > 1e-4:
        # VE = Weighted average of VE_i across all vaccinated classes
        # Get total vaccine coverage
        f_total = np.sum(v_dist_total)

        # For VE calculation, we need the attack rate among vaccinated,
        # which is weighted by the fraction of vaccinated in each class
        # v_dist_total is fraction of total pop, so v_dist_total/f_total = v_dist (fraction of vaccinated pop)
        v_dist = v_dist_total / f_total
        Omega_v_avg = np.sum(v_dist * Omega_v)

        return 1 - (Omega_v_avg / Omega_u)
    else:
        return 1


def solve_distribution_params_for_target_VE_heterogeneous(beta, tau, vax_prop, K, concentration, target_VE,
                                                     mean_VE_bounds=(0.01, 0.99)):
    """
    Find mean_VE parameter that yields target VE in heterogeneous model for a given concentration.

    This function solves for the mean_VE parameter of generate_vaccine_responder_classes
    such that the overall observed VE equals target_VE, while keeping concentration fixed.

    Parameters
    ----------
    beta : float
        Transmission rate.
    tau : float
        Recovery time (1/gamma).
    vax_prop : float
        Proportion of population vaccinated.
    K : int
        Number of vaccine responder classes.
    concentration : float
        Concentration parameter for the Beta distribution (fixed).
    target_VE : float
        Goal VE at end of simulation.
    mean_VE_bounds : tuple
        Bounds for mean_VE parameter (default: (0.01, 0.99)).

    Returns
    -------
    mean_VE : float
        Mean VE parameter that produces target_VE, constrained such that mean_VE >= target_VE.
    VS0_dist : array of length K
        Distribution of vaccinated individuals across classes.
    VE_levels : array of length K
        VE levels for each class.

    Notes
    -----
    The result is constrained to mean_VE >= target_VE to avoid counterintuitive
    results in herd immunity regimes where indirect effects dominate.
    """
    def VE_difference(mean_VE):
        # Generate vaccine responder classes with this mean_VE and concentration
        VS0_dist, VE_levels = generate_vaccine_responder_classes(
            n_classes=K,
            mean_VE=mean_VE,
            concentration=concentration
        )

        # Calculate v_dist_total (proportion of total population in each vaccinated class)
        v_dist_total = VS0_dist * vax_prop

        # Solve for attack rates
        Omega_u, Omega_v = solve_attack_rates_heterogeneous(beta, tau, v_dist_total, VE_levels)

        # Compute overall VE
        VE = compute_VE_heterogeneous(Omega_u, Omega_v, v_dist_total)

        return VE - target_VE

    # Constrain lower bound to be at least target_VE
    lower_bound = max(mean_VE_bounds[0], target_VE)
    upper_bound = mean_VE_bounds[1]

    # Check if constrained bounds are valid
    if lower_bound >= upper_bound:
        warnings.warn(f"Cannot satisfy mean_VE >= target_VE={target_VE:.3f} with upper bound {upper_bound}. "
                     f"Returning target_VE.")
        VS0_dist, VE_levels = generate_vaccine_responder_classes(
            n_classes=K, mean_VE=target_VE, concentration=concentration
        )
        return target_VE, VS0_dist, VE_levels

    # Check if a solution exists in the constrained range
    VE_at_lower = VE_difference(lower_bound)
    VE_at_upper = VE_difference(upper_bound)

    # If target_VE is achievable at mean_VE = target_VE, return it directly
    if abs(VE_at_lower) < 1e-6:
        VS0_dist, VE_levels = generate_vaccine_responder_classes(
            n_classes=K, mean_VE=lower_bound, concentration=concentration
        )
        return lower_bound, VS0_dist, VE_levels

    # Check if root exists in bracket
    if VE_at_lower * VE_at_upper > 0:
        # No root in bracket - return the bound that gives closest VE
        if abs(VE_at_lower) < abs(VE_at_upper):
            warnings.warn(f"No solution found in constrained range [{lower_bound:.3f}, {upper_bound:.3f}]. "
                         f"Returning mean_VE={lower_bound:.3f} (VE≈{VE_at_lower + target_VE:.3f})")
            mean_VE_solution = lower_bound
        else:
            warnings.warn(f"No solution found in constrained range [{lower_bound:.3f}, {upper_bound:.3f}]. "
                         f"Returning mean_VE={upper_bound:.3f} (VE≈{VE_at_upper + target_VE:.3f})")
            mean_VE_solution = upper_bound

        VS0_dist, VE_levels = generate_vaccine_responder_classes(
            n_classes=K, mean_VE=mean_VE_solution, concentration=concentration
        )
        return mean_VE_solution, VS0_dist, VE_levels

    # Root finding for mean_VE in constrained bounds
    try:
        result = root_scalar(VE_difference, bracket=(lower_bound, upper_bound), method='bisect')

        if not result.converged:
            raise RuntimeError("mean_VE root finding did not converge for heterogeneous model.")

        mean_VE_solution = result.root
    except ValueError as e:
        # If bracket doesn't contain a root, return target_VE
        raise RuntimeError(f"Could not find mean_VE that produces target_VE={target_VE}. "
                          f"Try adjusting concentration or target_VE. Error: {e}")

    # Double-check constraint
    mean_VE_solution = max(mean_VE_solution, target_VE)

    # Generate final distribution with solved mean_VE
    VS0_dist, VE_levels = generate_vaccine_responder_classes(
        n_classes=K,
        mean_VE=mean_VE_solution,
        concentration=concentration
    )

    return mean_VE_solution, VS0_dist, VE_levels


def SIRV_heterogeneous_model(t, y, beta, gamma, sigma_levels):
    """
    SIR model with K vaccinated susceptibility classes.
    Each class k has its own VS_k, VI_k, VR_k.

    State vector:
      y = [S, I, R,
           VS_0,...,VS_{K-1},
           VI_0,...,VI_{K-1},
           VR_0,...,VR_{K-1}]

    Parameters
    ----------
    t : float
        Time.
    y : array
        State vector.
    beta : float
        Transmission rate.
    gamma : float
        Recovery rate.
    sigma_levels : array of length K
        VE levels for each class (either uncorrected or corrected).
    """
    K = len(sigma_levels)

    # Extract compartments
    S = max(y[0], 0.0)
    I = max(y[1], 0.0)
    R = max(y[2], 0.0)

    VS = np.maximum(y[3:3+K], 0.0)
    VI = np.maximum(y[3+K:3+2*K], 0.0)
    VR = np.maximum(y[3+2*K:3+3*K], 0.0)

    # Total population
    N = S + I + R + VS.sum() + VI.sum() + VR.sum()

    # Force of infection
    lambda_inf = beta * (I + VI.sum()) / N

    # Unvaccinated dynamics
    dS  = -lambda_inf * S
    dI  = lambda_inf * S - gamma * I
    dR  = gamma * I

    # Vaccinated susceptibles (class-specific)
    infections_per_class = (1 - sigma_levels) * lambda_inf * VS
    dVS = -infections_per_class

    # Vaccinated infected
    dVI = infections_per_class - gamma * VI

    # Vaccinated recovered
    dVR = gamma * VI

    # Pack derivatives back into vector
    return np.concatenate([[dS, dI, dR], dVS, dVI, dVR])

def run_SIRV_heterogeneous(
    tmax=160,
    dt=0.1,
    inits=None,
    params=None,
    K=10,
    mean_VE=0.5,
    concentration=0.001, 
    sigma_correction=True
):
    """
    Run full K-class SIRV heterogeneous model with separate VI_k and VR_k compartments.

    Parameters
    ----------
    tmax : float
        Maximum simulation time.
    dt : float
        Time step for output resolution.
    inits : dict
        Initial fractions for unvaccinated S, I, R.
    params : dict
        Model parameters including beta, gamma, vax_prop.
    K : int
        Number of susceptibility strata.
    mean_VE : float, optional
        Mean of Beta distribution. If sigma_correction = True, used to inform target_VE.
    concentration : float, optional
        Concentration parameter for Beta distribution (default: all-or-nothing).
    sigma_correction : bool
        Whether to apply sigma correction to achieve target_VE (default: True).

    Returns
    -------
    dict
        Dictionary with time series for all compartments.
    """

    # Defaults
    if inits is None:
        inits = {"S": 0.9999, "I": 0.0001, "R": 0.0}
    if params is None:
        params = {"beta": 0.5, "gamma": 1/4, "vax_prop": 0.6}

    beta = params["beta"]
    gamma = params["gamma"]
    vax_prop = params["vax_prop"]
    
    # Apply alpha correction if requested
    if sigma_correction:
        mean_VE_solution, props, sigma_levels = \
        solve_distribution_params_for_target_VE_heterogeneous(beta, 
                                                        1/gamma, 
                                                        vax_prop, 
                                                        K, 
                                                        concentration, 
                                                        mean_VE)

   
    # Otherwise, create uncorrected susceptibility distribution
    else:
        props, sigma_levels = generate_vaccine_responder_classes(
            n_classes=K,
            mean_VE=mean_VE,
            concentration=concentration
        )

    # Calculate v_dist_total for initial conditions 
        # v_dist_total = proportion of total pop. in each vax compartment 
    v_dist_total = props * vax_prop

    # Initial susceptible, infected, recovered
    S0 = (1 - vax_prop) * inits["S"]
    I0 = inits["I"]
    R0 = inits["R"]

    # Vaccinated susceptibles across classes
    VS0 = v_dist_total * inits["S"]

    # Each vaccinated class starts with 0 infected and 0 recovered
    VI0 = np.zeros(K)
    VR0 = np.zeros(K)

    # Initial state vector
    y0 = np.concatenate([[S0, I0, R0], VS0, VI0, VR0])

    # Time grid
    t_eval = np.arange(0, tmax + dt, dt)

    sol = solve_ivp(
        SIRV_heterogeneous_model,
        [0, tmax],
        y0,
        args=(beta, gamma, sigma_levels),
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
        max_step=1.0
    )

    # Extract solution
    out = {
        "t": sol.t,
        "S": sol.y[0],
        "I": sol.y[1],
        "R": sol.y[2],
        "VS": sol.y[3:3+K],          # (K x T)
        "VI": sol.y[3+K:3+2*K],      # (K x T)
        "VR": sol.y[3+2*K:3+3*K],    # (K x T)
        "VE_levels": sigma_levels,
        "target_VE": mean_VE
    }

    return out

# -----------------------------------------------------------
# Other models of VE ---------------------------
# -----------------------------------------------------------

def estimate_VE_cox_tvi(res, time_categories=None, stratified=False):
    """
    Estimate vaccine effectiveness using Cox proportional hazards model with
    time-varying covariates (time category-vaccine interaction).

    Parameters
    ----------
    res : dict
        Model outputs from run_SIRV or run_SIRV_heterogeneous. Expected keys:
        - "t": time array
        - "S", "I", "R": unvaccinated compartments
        - "VS", "VI", "VR": vaccinated compartments (or shape (K, T) for heterogeneous)
        - "V": fully protected compartment (for all-or-nothing)
    time_categories : array-like, optional
        Time points defining category boundaries. If None, uses quartiles.
    stratified : bool
        If True and heterogeneous model, return VE per stratum.

    Returns
    -------
    VE_estimates : dict
        Dictionary with:
        - "overall": Overall VE estimate
        - "by_time_category": VE estimates for each time period
        - "hazard_ratio": Hazard ratios (1 - VE)
        - "time_categories": Time category boundaries used

    Notes
    -----
    This function constructs individual-level survival data from compartmental
    model outputs and fits a Cox model with vaccine status and time category
    interaction terms.
    """

    # Extract time array
    t = res["t"]
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    # Define time categories if not provided
    if time_categories is None:
        # Use quartiles of infection times
        infection_times = []
        for i in range(len(t)):
            if i > 0:
                new_inf_u = max(0, (res["I"][i] - res["I"][i-1]) / dt)
                new_inf_v = max(0, (res["VI"][i] - res["VI"][i-1]) / dt) if res["VI"].ndim == 1 else \
                           max(0, (res["VI"][:, i].sum() - res["VI"][:, i-1].sum()) / dt)
                infection_times.extend([t[i]] * int(new_inf_u + new_inf_v))

        if len(infection_times) > 0:
            time_categories = np.percentile(infection_times, [0, 25, 50, 75, 100])
        else:
            time_categories = np.linspace(t[0], t[-1], 5)

    time_categories = np.array(time_categories)
    n_categories = len(time_categories) - 1

    # Initialize arrays to store hazard rates by time category
    hazard_unvax = np.zeros(n_categories)
    hazard_vax = np.zeros(n_categories)
    person_time_unvax = np.zeros(n_categories)
    person_time_vax = np.zeros(n_categories)
    events_unvax = np.zeros(n_categories)
    events_vax = np.zeros(n_categories)

    # Handle heterogeneous model (VS, VI, VR are shape (K, T))
    is_heterogeneous = res["VS"].ndim == 2 if "VS" in res else False

    # Calculate hazard rates for each time category
    for cat_idx in range(n_categories):
        # Find time indices in this category
        t_start = time_categories[cat_idx]
        t_end = time_categories[cat_idx + 1]
        time_mask = (t >= t_start) & (t < t_end)

        if not np.any(time_mask):
            continue

        # Get time indices in this category
        t_indices = np.where(time_mask)[0]

        for i in t_indices:
            if i == 0:
                continue

            # Unvaccinated population at risk
            S_atrisk = (res["S"][i-1] + res["S"][i]) / 2

            # Vaccinated population at risk
            if is_heterogeneous:
                VS_atrisk = (res["VS"][:, i-1].sum() + res["VS"][:, i].sum()) / 2
            # all-or-nothing
            elif res["V"][0] != 0:
                VS_atrisk = (res["VS"][i-1] + res["VS"][i]) / 2 + (res["V"][i-1] + res["V"][i]) / 2
            # leaky
            else:
                VS_atrisk = (res["VS"][i-1] + res["VS"][i]) / 2

            # New infections in this time step
            dI_u = max(0, res["S"][i-1] - res["S"][i])

            if is_heterogeneous:
                dI_v = max(0, res["VS"][:, i-1].sum() - res["VS"][:, i].sum())
            else:
                dI_v = max(0, res["VS"][i-1] - res["VS"][i])

            # Accumulate events and person-time
            events_unvax[cat_idx] += dI_u
            events_vax[cat_idx] += dI_v
            person_time_unvax[cat_idx] += S_atrisk * dt
            person_time_vax[cat_idx] += VS_atrisk * dt

    # Calculate hazard rates (events / person-time)
    hazard_unvax = np.where(person_time_unvax > 0,
                            events_unvax / person_time_unvax, 0)
    hazard_vax = np.where(person_time_vax > 0,
                         events_vax / person_time_vax, 0)

    # Calculate hazard ratios and VE by time category
    hazard_ratios = np.zeros(n_categories)
    VE_by_category = np.zeros(n_categories)

    for cat_idx in range(n_categories):
        if hazard_unvax[cat_idx] > 0:
            hazard_ratios[cat_idx] = hazard_vax[cat_idx] / hazard_unvax[cat_idx]
            VE_by_category[cat_idx] = 1 - hazard_ratios[cat_idx]
        else:
            hazard_ratios[cat_idx] = np.nan
            VE_by_category[cat_idx] = np.nan

    # Calculate overall VE (weighted by events)
    total_events = events_unvax.sum() + events_vax.sum()
    if total_events > 0:
        # Mantel-Haenszel estimate pooled across strata
        numerator = np.sum(events_vax * person_time_unvax / (person_time_unvax + person_time_vax))
        denominator = np.sum(events_unvax * person_time_vax / (person_time_unvax + person_time_vax))

        if denominator > 0:
            overall_HR = numerator / denominator
            overall_VE = 1 - overall_HR
        else:
            overall_VE = np.nan
            overall_HR = np.nan
    else:
        overall_VE = np.nan
        overall_HR = np.nan

    # Return results
    return {
        "overall_VE": overall_VE,
        "VE_by_time_category": VE_by_category,
        "hazard_ratio_by_category": hazard_ratios,
        "overall_hazard_ratio": overall_HR,
        "time_categories": time_categories,
        "events_unvax": events_unvax,
        "events_vax": events_vax,
        "person_time_unvax": person_time_unvax,
        "person_time_vax": person_time_vax,
        "hazard_unvax": hazard_unvax,
        "hazard_vax": hazard_vax
    }

def get_VE_binomial(num_exposures,pr_inf_per_exposure,alpha,mech="leaky",c=None):
    '''
    Calculate VE from simple Bernoulli trial model

    Parameters:
        num_exposures: number of exposure events
        pr_inf_per_exposure: probability of infection per exposure for fully susceptible person
        alpha: reduction in per-exposure infection probability for a vaccinacted person
        mech: vaccine mechanism ("all-or-nothing","leaky","heterogeneous")
        c: shape parameter (if vaccine mechanism is heterogeneous, this is required)
    
    Calculates VE as 1 - Pr(infected | vaxed) / Pr(infected | unvaxed)
    '''
    if mech == "all-or-nothing":
        VE = alpha

    elif mech == "leaky":
        VE = 1 - (1 - (1 - (1-alpha)*pr_inf_per_exposure)**num_exposures) / (1 - (1 - pr_inf_per_exposure)**num_exposures)

    elif mech == "heterogeneous":
        props, alpha_levels = generate_vaccine_responder_classes(
            n_classes=50,
            mean_VE=alpha,
            concentration=c
        )
        P_vax = 0 
        for pr_alpha, alpha in zip(props, alpha_levels):
            P_vax += pr_alpha * (1 - (1 - (1-alpha)*pr_inf_per_exposure)**num_exposures)
            VE = 1 - P_vax / (1 - (1 - pr_inf_per_exposure)**num_exposures)
    
    return VE

# -----------------------------------------------------------
# Observed VE from model outputs ---------------------------
# -----------------------------------------------------------

def get_observed_VE_direct(res,time_varying=False):
    N_unvax = res["S"][0] + res["I"][0] + res["R"][0]
    N_vax = res["VS"][0] + res["VI"][0] + res["VR"][0] + res["V"][0]
    if time_varying:
        VE = 1 - (res["VR"]/N_vax)/(res["R"]/N_unvax)
    else:
        try: 
            VE = 1 - (res["VR"][-1]/N_vax)/(res["R"][-1]/N_unvax)
        except:
            VE = 0
    return VE

def get_observed_VE_direct_heterogeneous(res, time_varying=False, stratified=False):
    """ 
    Calculate observed direct VE for the heterogeneous model with per-stratum VI_k, VR_k.

    Parameters
    ----------
    res : dict
        Model outputs from run_SIRV_heterogeneous_full. Expected keys:
            "S","I","R"         : arrays length T
            "VS","VI","VR"      : arrays shape (K, T)
    time_varying : bool
        If True return VE(t) arrays; if False return endpoint scalar(s).
    stratified : bool
        If True return VE per stratum (dict); else return pooled VE.

    Returns
    -------
    If stratified=False:
        VE (float) or VE_time (np.ndarray length T)
    If stratified=True:
        dict {k: VE_k} where VE_k is float or np.ndarray
    """

    # ---- unpack and validate ----
    S = np.asarray(res["S"])
    I = np.asarray(res["I"])
    R = np.asarray(res["R"])

    VS = np.asarray(res["VS"])   # shape (K, T)
    VI = np.asarray(res["VI"])   # shape (K, T)
    VR = np.asarray(res["VR"])   # shape (K, T)

    # ensure shapes: (K, T)
    if VS.ndim != 2 or VI.ndim != 2 or VR.ndim != 2:
        raise ValueError("VS, VI, VR must be 2D arrays with shape (K, T)")

    K, T = VS.shape
    # time axis length check
    if not (I.size == T and R.size == T and S.size == T and VI.shape[1] == T and VR.shape[1] == T):
        raise ValueError("Time dimension mismatch between compartments")

    # ---- population sizes at t=0 (initial populations) ----
    N_unvax = S[0] + I[0] + R[0]

    N_vax_k = (VS[:, 0] + VI[:, 0] + VR[:, 0])  # length K
    N_vax_total = N_vax_k.sum()

    # avoid zero denominators later
    # ---- attack rates ----
    if time_varying:
        # AR_unvax(t) = R(t) / N_unvax
        with np.errstate(divide='ignore', invalid='ignore'):
            AR_unvax_t = R / N_unvax  # length T

        # per-stratum vaccinated AR_k(t) = VR_k(t) / N_vax_k
        AR_vax_k_t = np.empty((K, T), dtype=float)
        for k in range(K):
            if N_vax_k[k] <= 0:
                AR_vax_k_t[k, :] = np.nan
            else:
                AR_vax_k_t[k, :] = VR[k, :] / N_vax_k[k]

        # pooled vaccinated AR(t) = sum_k VR_k(t) / N_vax_total
        if N_vax_total <= 0:
            AR_vax_t = np.full(T, np.nan)
        else:
            AR_vax_t = VR.sum(axis=0) / N_vax_total

        # ---- compute VE(t) ----
        # VE = 1 - (AR_vax / AR_unvax)
        with np.errstate(divide='ignore', invalid='ignore'):
            if stratified:
                VE_dict = {}
                for k in range(K):
                    VE_k_t = 1 - (AR_vax_k_t[k, :] / AR_unvax_t)
                    VE_dict[k] = VE_k_t
                return VE_dict
            else:
                VE_t = 1 - (AR_vax_t / AR_unvax_t)
                return VE_t

    else:
        # endpoint (final time)
        R_final = R[-1]
        if N_unvax == 0:
            AR_unvax = np.nan
        else:
            AR_unvax = R_final / N_unvax

        AR_vax_k = np.empty(K, dtype=float)
        for k in range(K):
            if N_vax_k[k] <= 0:
                AR_vax_k[k] = np.nan
            else:
                AR_vax_k[k] = VR[k, -1] / N_vax_k[k]

        if N_vax_total <= 0:
            AR_vax = np.nan
        else:
            AR_vax = VR[:, -1].sum() / N_vax_total

        with np.errstate(divide='ignore', invalid='ignore'):
            if stratified:
                VE_dict = {k: (1 - (AR_vax_k[k] / AR_unvax)) for k in range(K)}
                return VE_dict
            else:
                VE = 1 - (AR_vax / AR_unvax)
                return VE