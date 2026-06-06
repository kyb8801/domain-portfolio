"""
Case C scaffold — Virtual Metrology + conformal gate (KITECH 트랙).
실 SECOM(UCI)은 data/ 에 받고(load_secom(path=...)), 여기선 구조 + 합성으로 게이트 coverage 검증.
shared 레이어(ml_inverse, measurement_gate) 재사용.
"""
import numpy as np
from ml_inverse import MLInverse
from measurement_gate import gate_report, PASS


def load_secom(path=None, seed=0):
    if path:
        import pandas as pd
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].fillna(df.median(numeric_only=True)).values
        y = df.iloc[:, -1].values
        return X, y
    rng = np.random.default_rng(seed)
    n, d = 2000, 40
    X = rng.standard_normal((n, d))
    y = 100 + 3*X[:, 0] + 2*X[:, 1] - 1.5*X[:, 2] + 1.0*X[:, 3] + rng.standard_normal(n)*1.0
    return X, y


if __name__ == "__main__":
    X, y = load_secom()
    ntr = 1400
    mdl = MLInverse().fit(X[:ntr], y[:ntr])
    o = mdl.predict(X[ntr:]); yt = y[ntr:]
    SPEC_LO, SPEC_HI = 96.0, 104.0
    rep = gate_report(o["pred"], o["conformal_halfwidth"], SPEC_LO, SPEC_HI, o["conformal_level"])
    c = rep["counts"]
    print(f"[VM gate] spec=[{SPEC_LO:.0f},{SPEC_HI:.0f}] level={rep['conformal_level']:.0%} halfwidth={rep['halfwidth']:.2f}")
    print(f"PASS/FAIL/RETEST = {c['PASS']}/{c['FAIL']}/{c['RETEST']} (retest {rep['retest_frac']:.0%})")
    cover = (np.abs(o['pred']-yt) <= o['conformal_halfwidth']).mean()
    print(f"conformal coverage: target {o['conformal_level']:.0%} vs empirical {cover:.0%}")
    g = rep['decisions']; true_in = (yt >= SPEC_LO) & (yt <= SPEC_HI); pm = g == PASS
    if pm.sum():
        print(f"PASS precision = {(true_in & pm).sum()/pm.sum():.0%}")
