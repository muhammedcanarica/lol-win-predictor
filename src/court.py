"""Kronolojik OOF olasılıklarıyla eğitilen üst modeller ve kanıt değerlendirmesi."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONTEXT = ("golddiffat10", "xpdiffat10", "csdiffat10", "kill_diff", "elo_diff")
METHODS = ("mean", "weighted", "stacking", "gating", "evidence_only", "judge_context", "judge_stacking")
# Sadece seçim katlarında, eşit log-loss durumunda listedeki basit yöntem tercih edilir.
SELECTABLE = ("mean", "weighted", "stacking", "gating", "judge_context", "judge_stacking")


def ridge():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


def context_row(record):
    base = record["base"]
    return [base["golddiffat10"], base["xpdiffat10"], base["csdiffat10"],
            base["killsat10"] - base["deathsat10"], record["elo"]["elo_diff"]]


def gate_inputs(p, context):
    return np.column_stack((context, np.abs(context[:, 0]), p.std(axis=1)))


def evidence_inputs(p, context, reasons, expert):
    votes = p >= 0.5
    direction = np.where(votes[:, expert], 1, -1)
    supporting_others = (votes == votes[:, [expert]]).sum(axis=1) - 1
    return np.column_stack((np.abs(p[:, expert] - 0.5), direction, supporting_others,
                            context * direction[:, None], reasons[:, expert] * direction[:, None]))


@dataclass
class Court:
    weights: np.ndarray
    stack: object
    gate_scaler: StandardScaler
    gate_coef: np.ndarray
    reliability: list
    context_judge: object

    def evaluate(self, p, context, reasons):
        gates = self.gate_scaler.transform(gate_inputs(p, context))
        dynamic_weights = softmax(np.column_stack((np.ones(len(p)), gates)) @ self.gate_coef, axis=1)
        reliability = np.column_stack([
            model.predict_proba(evidence_inputs(p, context, reasons, i))[:, 1]
            for i, model in enumerate(self.reliability)
        ])
        votes = p >= 0.5
        strength = reliability * np.abs(2 * p - 1)
        blue = (strength * votes).sum(axis=1)
        red = (strength * ~votes).sum(axis=1)
        evidence = 0.5 + 0.5 * (blue - red) / reliability.sum(axis=1)
        stacking = self.stack.predict_proba(p)[:, 1]
        independent = self.context_judge.predict_proba(context)[:, 1]
        return {
            "mean": p.mean(axis=1), "weighted": p @ self.weights, "stacking": stacking,
            "gating": (p * dynamic_weights).sum(axis=1), "evidence_only": evidence,
            "judge_context": (evidence + independent) / 2, "judge_stacking": (evidence + stacking) / 2,
        }, {"reliability": reliability, "blue_evidence": blue, "red_evidence": red,
            "context_judge": independent, "gating_weights": dynamic_weights}


def fit_weights(p, y):
    n_experts = p.shape[1]
    uniform = np.full(n_experts, 1 / n_experts)

    def weighted_loss(w):
        error = p @ w - y
        shrinkage = w - uniform
        return (np.mean(error ** 2) + 0.001 * np.sum(shrinkage ** 2),
                2 * (p.T @ error) / len(y) + 0.002 * shrinkage)

    fixed = minimize(weighted_loss, uniform, bounds=[(0.0, 1.0)] * n_experts, method="SLSQP", jac=True,
                     constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda _: np.ones(n_experts)},
                     options={"ftol": 1e-10})
    if not fixed.success:
        raise RuntimeError(f"Sabit ağırlık optimizasyonu başarısız: {fixed.message}")
    return fixed.x


def fit_court(p, context, reasons, y) -> Court:
    n_experts = p.shape[1]
    weights = fit_weights(p, y)

    scaler = StandardScaler().fit(gate_inputs(p, context))
    design = np.column_stack((np.ones(len(p)), scaler.transform(gate_inputs(p, context))))
    shape = (design.shape[1], n_experts)

    def gate_loss(flat):
        coef = flat.reshape(shape)
        weights = softmax(design @ coef, axis=1)
        prediction = (weights * p).sum(axis=1, keepdims=True)
        error = prediction - y[:, None]
        penalty = 0.02
        loss = np.mean(error ** 2) + penalty * np.sum(coef ** 2)
        gradient = 2 * (design.T @ (error * weights * (p - prediction))) / len(y) + 2 * penalty * coef
        return loss, gradient.ravel()

    gate = minimize(gate_loss, np.zeros(np.prod(shape)), jac=True, method="L-BFGS-B")
    if not gate.success:
        raise RuntimeError(f"Koşullu ağırlık optimizasyonu başarısız: {gate.message}")
    votes = p >= 0.5
    reliability = [ridge().fit(evidence_inputs(p, context, reasons, i), votes[:, i] == y) for i in range(n_experts)]
    return Court(weights, ridge().fit(p, y), scaler, gate.x.reshape(shape), reliability, ridge().fit(context, y))
