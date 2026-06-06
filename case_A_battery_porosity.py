"""
Case A scaffold — battery cathode 미세구조 정량 + 불확도 (LG·SDI·포스코 트랙).
실 NREL Battery Microstructure Library tif 는 data/ 에 받고(tifffile.imread), 여기선
구조 + 합성 segmented 볼륨으로 검증. shared/uncertainty 재사용 — 정량에 U(k=2) 부착이 차별점.
"""
import numpy as np
from uncertainty import expand


def phase_fractions(seg, labels=(0, 1, 2)):
    return {int(l): float((seg == l).mean()) for l in labels}


def porosity_with_uncertainty(seg, n_subvol=8, pore_label=0):
    subs = np.array_split(seg, n_subvol, axis=0)          # 서브볼륨 분산 → type A
    p = np.array([(s == pore_label).mean() for s in subs])
    u_a = p.std(ddof=1) / np.sqrt(len(p))
    return float(p.mean()), float(u_a)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 합성 segmented 볼륨 (0=pore, 1=active NMC, 2=carbon-binder)
    seg = rng.choice([0, 1, 2], size=(80, 128, 128), p=[0.38, 0.50, 0.12])
    print(f"phase fractions: {phase_fractions(seg)}")
    p, u = porosity_with_uncertainty(seg)
    print(f"porosity = {p:.4f} +/- U(k=2) = {expand(u):.4f}  (target 0.380)")
    print(f"  -> 계측사 언어: porosity {p*100:.1f}% +/- {expand(u)*100:.2f}% (k=2)")
