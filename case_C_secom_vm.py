"""
Case C — SECOM 반도체 공정 불량 탐지 + conformal 분류 게이트 (KITECH 트랙).
SECOM 은 분류(pass=-1 / fail=1, 불균형 104/1463)라 회귀 VM 대신 conformal 분류 게이트를 쓴다.
게이트: 확실 정상 / 확실 불량 / 불확실(재검) — 불확실만 사람이 보면 되어 검사 부하가 준다.
data: download_data.get_secom() 가 저장한 data_secom_X.npy / data_secom_y.npy. 없으면 합성 불균형 fallback.
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def load(path_X="data_secom_X.npy", path_y="data_secom_y.npy", seed=0):
    if os.path.exists(path_X):
        X = np.load(path_X); y = np.load(path_y)
        col_med = np.nanmedian(X, axis=0)               # NaN -> 열 중앙값
        nan = np.where(np.isnan(X)); X[nan] = np.take(col_med, nan[1])
        return X, y
    rng = np.random.default_rng(seed)                   # 합성 불균형 fallback
    n, d = 1567, 60
    X = rng.standard_normal((n, d))
    logit = -2.7 + 1.3*X[:, 0] + 0.9*X[:, 1] - 0.8*X[:, 2]
    y = np.where(rng.random(n) < 1/(1+np.exp(-logit)), 1.0, -1.0)
    return X, y


class ConformalGate:
    """RF 분류 + split conformal(1 - P_true) → 예측집합. 집합 크기로 게이트."""
    def __init__(self, n_estimators=300):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=0)
        self.qhat = None; self.classes_ = None

    def fit(self, X, y, alpha=0.1, calib_frac=0.25, seed=0):
        idx = np.random.default_rng(seed).permutation(len(X))
        nc = max(1, int(len(X)*calib_frac)); cal, tr = idx[:nc], idx[nc:]
        self.clf.fit(X[tr], y[tr]); self.classes_ = list(self.clf.classes_)
        proba = self.clf.predict_proba(X[cal])
        ti = [self.classes_.index(t) for t in y[cal]]
        scores = 1 - proba[np.arange(len(cal)), ti]
        self.qhat = float(np.quantile(scores, 1 - alpha))
        return self

    def gate(self, X):
        keep = self.clf.predict_proba(X) >= (1 - self.qhat)   # 집합 멤버십
        return keep, keep.sum(1)                               # size1=확실,2=불확실,0=이상


if __name__ == "__main__":
    X, y = load()
    ntr = int(len(X)*0.7)
    g = ConformalGate().fit(X[:ntr], y[:ntr], alpha=0.1)
    keep, size = g.gate(X[ntr:]); yt = y[ntr:]; cls = g.classes_
    ti = [cls.index(t) for t in yt]
    covered = keep[np.arange(len(yt)), ti].mean()
    print(f"[SECOM conformal gate] target coverage 90% vs empirical {covered:.0%}")
    print(f"certain(자동) {(size==1).mean():.0%} | uncertain(재검) {(size==2).mean():.0%} | 이상 {(size==0).mean():.0%}")
    sure = size == 1
    if sure.sum():
        pred = np.array([cls[k.argmax()] for k in keep[sure]])
        print(f"확실판정 정확도 {(pred==yt[sure]).mean():.0%}  (불확실 {(size==2).mean():.0%}만 사람이 재검)")
