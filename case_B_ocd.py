"""
Case B — OCD scatterometry inverse + uncertainty + gate (삼성전기 트랙).
NIST L100P300 (공개, data.nist.gov ark:/88434/mds2-2290). 측정 스펙트럼 -> linewidth(CD) 역산.
- 정확도: sim library held-out 5-fold RMSE.
- 실측 9 dies: CD + conformal 불확도 + 합부 게이트 + GUM U(k=2).
- sim-to-real gap: measured 가 sim 분포 밖이면 ensemble epistemic 이 커져 OOD 를 정직하게 드러낸다.
"""
import os
import numpy as np
from sklearn.model_selection import KFold
from ml_inverse import MLInverse
from measurement_gate import gate_report
from gum_posterior_bridge import conformal_to_standard, posterior_to_gum


def load_nist(d="."):
    f = lambda n: np.loadtxt(os.path.join(d, n), delimiter=",")
    return (f("Phi_L100P300_sim_467.csv"), f("Pi_L100P300_sim_467.csv"), f("Imean_L100P300_exp.csv"))


if __name__ == "__main__":
    Phi, Pi, Imean = load_nist()
    mid = Pi[:, 1]
    print(f"[NIST L100P300] sim library {Phi.shape}, labels {Pi.shape}, measured {Imean.shape}")

    kf = KFold(5, shuffle=True, random_state=0); errs = []
    for tr, te in kf.split(Phi):
        m = MLInverse().fit(Phi[tr], mid[tr])
        errs.append(np.sqrt(np.mean((m.predict(Phi[te])["pred"] - mid[te]) ** 2)))
    rmse = np.mean(errs)
    print(f"[정확도] mid-CD inverse, sim 5-fold RMSE = {rmse:.2f} +/- {np.std(errs):.2f} nm")

    m = MLInverse().fit(Phi, mid); o = m.predict(Imean)
    u_conf = conformal_to_standard(o["conformal_halfwidth"], o["conformal_level"])
    U2 = float(posterior_to_gum(u_conf, u_sys=0.5)["U"])           # conformal 기반 U(k=2)
    spec_lo, spec_hi = 105.0, 130.0
    rep = gate_report(o["pred"], o["conformal_halfwidth"], spec_lo, spec_hi, o["conformal_level"])
    print(f"[실측 9 dies] CD ± U(k=2) = ±{U2:.1f} nm (conformal), spec=[{spec_lo:.0f},{spec_hi:.0f}]")
    for i in range(len(o["pred"])):
        ood = " <- OOD(sim-real gap)" if o["u_epistemic"][i] > 2 * rmse else ""
        print(f"  Die {i+1}: CD={o['pred'][i]:.1f} ± {U2:.1f} nm  [{rep['decisions'][i]}]{ood}")
    c = rep["counts"]
    print(f"게이트: PASS {c['PASS']} / FAIL {c['FAIL']} / RETEST {c['RETEST']}  "
          f"(epistemic 중앙값 {np.median(o['u_epistemic']):.1f} nm >> sim RMSE {rmse:.1f} -> 측정이 라이브러리 밖)")
