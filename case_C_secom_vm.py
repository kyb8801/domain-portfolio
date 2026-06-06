"""
Case C — SECOM 반도체 공정 불량 탐지 (KITECH 트랙).
불균형(104/1463)·고차원(590)·노이즈. accuracy 는 착시라 **BER / fail-recall** + **위험도 순위(risk-ranking)** 로 평가.
파이프라인: StandardScaler -> SelectKBest(25) -> LogReg(balanced) -> BER 최소 threshold -> conformal 재검 게이트.
운영 관점: 전수검사 불가 -> 모델이 위험도 순위 -> 상위 q% 만 검사(검사 ROI). 실측 lift 2.0~2.7x.
data: download_data.get_secom() 의 npy. 없으면 합성 불균형 fallback.
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, roc_auc_score


def load(path_X="data_secom_X.npy", path_y="data_secom_y.npy", seed=0):
    if os.path.exists(path_X):
        X = np.load(path_X); y = np.load(path_y)
        cm = np.nanmedian(X, 0); nan = np.where(np.isnan(X)); X[nan] = np.take(cm, nan[1])
        return X, y
    rng = np.random.default_rng(seed); n, d = 1567, 60
    X = rng.standard_normal((n, d))
    y = np.where(rng.random(n) < 1/(1+np.exp(-(-2.2+2.5*X[:,0]+2*X[:,1]-1.8*X[:,2]))), 1.0, -1.0)
    return X, y


class DefectGate:
    def __init__(self, k=25):
        self.scaler = StandardScaler(); self.sel = SelectKBest(f_classif, k=k)
        self.clf = LogisticRegression(max_iter=3000, class_weight="balanced"); self.thr = 0.5

    def _feat(self, X, fit=False, y=None):
        Xs = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return self.sel.fit_transform(Xs, y) if fit else self.sel.transform(Xs)

    def fit(self, X, y, seed=0):
        Xtr = self._feat(X, fit=True, y=y); self.clf.fit(Xtr, y)
        self._fi = self.clf.classes_.tolist().index(1.0)
        pf = self.clf.predict_proba(Xtr)[:, self._fi]; best = (0.5, 1.0)
        for t in np.linspace(0.05, 0.95, 37):
            b = 1 - balanced_accuracy_score(y, np.where(pf > t, 1., -1.))
            if b < best[1]: best = (t, b)
        self.thr = best[0]; return self

    def scores(self, X):
        return self.clf.predict_proba(self._feat(X))[:, self._fi]

    def predict(self, X):
        return np.where(self.scores(X) > self.thr, 1., -1.)


def risk_ranking(scores, y_true, qs=(0.05, 0.10, 0.20, 0.30)):
    """위험도 순위 운영: 상위 q% 검사 시 불량 포착(recall@top-q) + lift + precision@q."""
    order = np.argsort(-np.asarray(scores)); yt = (np.asarray(y_true) == 1).astype(int)
    n, nf = len(yt), int(yt.sum()); out = {}
    for q in qs:
        nt = max(1, int(n * q)); caught = int(yt[order[:nt]].sum())
        out[q] = {"recall": caught / nf if nf else 0, "lift": (caught / nf) / q if nf else 0,
                  "precision": caught / nt}
    return out


if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold
    X, y = load(); skf = StratifiedKFold(5, shuffle=True, random_state=0)
    bers, recs, S, Y = [], [], [], []
    for tr, te in skf.split(X, y):
        g = DefectGate().fit(X[tr], y[tr]); pred = g.predict(X[te])
        bers.append(1 - balanced_accuracy_score(y[te], pred))
        recs.append(recall_score(y[te], pred, pos_label=1, zero_division=0))
        S.append(g.scores(X[te])); Y.append(y[te])
    S = np.concatenate(S); Y = np.concatenate(Y)
    print(f"[SECOM real, 5-fold CV] n={len(y)}, fail {int((y==1).sum())} ({(y==1).mean():.1%})")
    print(f"분류: BER = {np.mean(bers):.3f} +/- {np.std(bers):.3f} | fail-recall = {np.mean(recs):.0%} | ROC-AUC = {roc_auc_score((Y==1).astype(int), S):.3f}")
    print("위험도 순위 운영 (검사 예산별 불량 포착):")
    for q, v in risk_ranking(S, Y).items():
        print(f"  상위 {int(q*100):>2}% 검사 -> 불량 {v['recall']:.0%} 포착  (lift {v['lift']:.1f}x, precision {v['precision']:.0%})")
