from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class Candidate:
    name: str
    utility: float
    preprocessing_cost: float
    detector_cost: float

    def __post_init__(self):
        if not np.isfinite([self.utility, self.preprocessing_cost, self.detector_cost]).all() or not 0 <= self.utility <= 1 or min(self.preprocessing_cost, self.detector_cost) < 0:
            raise ValueError('Candidates require accuracy in [0, 1] and finite nonnegative costs')

    @property
    def cost(self) -> float:
        return self.preprocessing_cost + self.detector_cost


def image_formation(radiance, depth, attenuation, backscatter) -> np.ndarray:
    clear = np.asarray(radiance, dtype=float)
    distance = np.asarray(depth, dtype=float)
    beta = np.asarray(attenuation, dtype=float)
    background = np.asarray(backscatter, dtype=float)
    if not all(np.isfinite(x).all() for x in (clear, distance, beta, background)) or np.any(distance < 0) or np.any(beta < 0):
        raise ValueError('Formation inputs must be finite; depth and attenuation must be nonnegative')
    if distance.ndim == clear.ndim - 1:
        distance = distance[..., None]
    transmission = np.exp(-distance * beta)
    return clear * transmission + background * (1.0 - transmission)


def lagrangian(utility: Callable, p: float, n: float, budget: float, multiplier: float, mu_p: float = 0.0, mu_n: float = 0.0) -> float:
    if min(multiplier, mu_p, mu_n) < 0:
        raise ValueError('Dual multipliers must be nonnegative')
    return float(utility(p, n) - multiplier * (p + n - budget) + mu_p * p + mu_n * n)


def kkt_residuals(gradient, p: float, n: float, budget: float, multiplier: float, mu_p: float, mu_n: float) -> dict:
    fp, fn = np.asarray(gradient, dtype=float)
    return {'stationarity': [float(fp - multiplier + mu_p), float(fn - multiplier + mu_n)], 'complementarity': [multiplier * (p + n - budget), mu_p * p, mu_n * n], 'primal_violation': max(0.0, -p, -n, p + n - budget), 'dual_violation': max(0.0, -multiplier, -mu_p, -mu_n)}


def budget_derivatives(gradient, hessian) -> tuple[float, float]:
    g, h = np.asarray(gradient, dtype=float), np.asarray(hessian, dtype=float)
    direction = np.array([1.0, -1.0])
    return float(g @ direction), float(direction @ h @ direction)


def continuous_allocation(utility: Callable, gradient: Callable, budget: float, tolerance: float = 1e-9) -> dict:
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError('Budget must be finite and positive')
    starts = ((0.0, 0.0), (budget, 0.0), (0.0, budget), (budget / 2, budget / 2), (budget / 3, budget / 3))
    solutions = [minimize(lambda z: -utility(*z), start, jac=lambda z: -np.asarray(gradient(*z), dtype=float), bounds=((0.0, budget), (0.0, budget)), constraints={'type': 'ineq', 'fun': lambda z: budget - z.sum(), 'jac': lambda z: np.array([-1.0, -1.0])}, method='SLSQP', options={'ftol': tolerance, 'maxiter': 2000}) for start in starts]
    valid = [s for s in solutions if s.success and s.x.sum() <= budget + tolerance * 10]
    if not valid:
        raise RuntimeError('No converged feasible allocation: ' + '; '.join(s.message for s in solutions))
    optimum = min(valid, key=lambda s: s.fun)
    p, n = map(float, optimum.x)
    g = np.asarray(gradient(p, n), dtype=float)
    active = abs(p + n - budget) <= tolerance * 100
    positive = np.array([p, n]) > tolerance * 100
    multiplier = max(0.0, float(g[positive].mean())) if active and positive.any() else 0.0
    mus = np.where(positive, 0.0, multiplier - g)
    return {'preprocessing': p, 'detector': n, 'utility': float(-optimum.fun), 'multiplier': multiplier, 'kkt': kkt_residuals(g, p, n, budget, multiplier, *mus), 'global_optimality_requires': 'Concavity of the supplied utility and satisfaction of KKT conditions.'}


def finite_allocation(candidates: Sequence[Candidate], budget: float) -> dict:
    if not candidates or not np.isfinite(budget) or budget < 0:
        raise ValueError('Supply candidates and a finite nonnegative budget')
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError('Candidate names must be unique')
    feasible = [c for c in candidates if c.cost <= budget]
    if not feasible:
        raise ValueError('No candidate satisfies the budget')
    optimum = max(c.utility for c in feasible)
    selected = [c.name for c in feasible if c.utility == optimum]
    retained = [c.name for c in candidates if not any(other.cost <= c.cost and other.utility > c.utility for other in candidates)]
    costs = np.array([c.cost for c in candidates])
    utilities = np.array([c.utility for c in candidates])
    dual = linprog([budget, 1.0], A_ub=np.column_stack((-costs, -np.ones(len(costs)))), b_ub=-utilities, bounds=[(0.0, None), (None, None)], method='highs')
    if not dual.success:
        raise RuntimeError(dual.message)
    multiplier = float(dual.x[0])
    return {'budget': budget, 'selected': selected, 'utility': optimum, 'retained': retained, 'dual_multiplier': multiplier, 'dual_bound': float(dual.fun), 'duality_gap': max(0.0, float(dual.fun - optimum)), 'scores': {c.name: c.utility - multiplier * c.cost for c in candidates}}


def score_difference(a: Candidate, b: Candidate, multiplier: float) -> float:
    if multiplier < 0:
        raise ValueError('Multiplier must be nonnegative')
    return a.utility - b.utility - multiplier * (a.cost - b.cost)


def substitution_gain(surface: Callable, capacity: float, p0: float, p1: float) -> float:
    if p1 <= p0:
        raise ValueError('Require p1 > p0')
    return float(surface(capacity, p1) - surface(capacity, p0))


def substitution_derivative(cross_partial: Callable, capacity: float, p0: float, p1: float) -> tuple[float, float]:
    if p1 <= p0:
        raise ValueError('Require p1 > p0')
    return quad(lambda s: cross_partial(capacity, s), p0, p1)


def decreasing_difference(cross_partial: Callable, k0: float, k1: float, p0: float, p1: float) -> dict:
    if k1 <= k0 or p1 <= p0:
        raise ValueError('Require increasing capacity and preprocessing intervals')
    value, error = quad(lambda k: quad(lambda p: cross_partial(k, p), p0, p1)[0], k0, k1)
    return {'gain_difference': value, 'outer_quadrature_error': error}


def trajectory_mean_variance(covariance) -> float:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise ValueError('Supply a finite symmetric covariance matrix from independent repeated experiments')
    if np.linalg.eigvalsh(matrix).min() < -1e-10:
        raise ValueError('Covariance must be positive semidefinite')
    return float(matrix.sum() / matrix.shape[0] ** 2)


def selection_optimism(expected_scores, zero_mean_error_draws) -> dict:
    truth = np.asarray(expected_scores, dtype=float)
    errors = np.asarray(zero_mean_error_draws, dtype=float)
    if truth.ndim != 1 or errors.ndim != 2 or errors.shape[1] != len(truth) or not errors.size or not np.isfinite(truth).all() or not np.isfinite(errors).all():
        raise ValueError('Supply candidate expectations and repeated finite error draws')
    selected = np.max(truth[None, :] + errors, axis=1)
    return {'mean_selected_score': float(selected.mean()), 'best_expected_score': float(truth.max()), 'monte_carlo_optimism': float(selected.mean() - truth.max()), 'sample_error_means': errors.mean(axis=0).tolist()}
