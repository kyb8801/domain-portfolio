# Domain Portfolio — Uncertainty-Aware Metrology on Public Data

계측·품질 방법을 **타깃 도메인 공개 데이터**에 적용한 케이스 모음.
공통 무기 = 모든 결과에 **불확도**를 붙인다 — 점추정이 아니라 *"이 측정을 얼마나 믿을 수 있나"* 까지.

> "그 도메인 안 해봤잖아"에 **작동하는 결과물**로 답하면서, 동시에 **불확도·보장**까지 붙인다. 남들이 segmentation/분류만 할 때.

---

## Cases

| | 도메인 (타깃) | 공개 데이터 | 핵심 결과 |
|---|---|---|---|
| **B ⭐** | 반도체 박막·**OCD 계측** (삼성전기) | NIST L100P300 | mid-CD inverse **RMSE 3.69 nm**; 불확도가 **sim-to-real OOD** 를 포착해 과신 대신 RETEST → [자세히](case_B_README.md) |
| **C** | 가상계측·**이상탐지** (KITECH) | UCI SECOM | 불량탐지 **BER 0.329 / ROC-AUC 0.70**; **위험도 순위** 상위 20% 검사 → 불량 47% (lift 2.4×) → [자세히](case_C_README.md) |
| A | 배터리 양극재 정량 (LG·SDI·포스코) | NREL (예정) | porosity + per-metric 불확도 (합성 검증 완료, 실 tif 연결 예정) |

**B(OCD)** 가 하이라이트 — 반도체 계측 본업에서, 상용 점추정 도구와 달리 *측정이 라이브러리 밖이면 불확도가 커져 "재측정"을 요구하는* calibrated 계측 AI 를 보인다.

---

## Core — 재사용 불확도 레이어 (sandbox 검증 완료)

| 파일 | 역할 |
|---|---|
| `uncertainty.py` | GUM 불확도 (combine / expand / budget / Monte Carlo) |
| `ml_inverse.py` | ensemble inverse + conformal 불확도 |
| `measurement_gate.py` | PASS / FAIL / RETEST 합부 게이트 (분포무관 보장) |
| `gum_posterior_bridge.py` | ML 불확도 → GUM U(k=2) 번역 |

---

## 정직 원칙 (전 케이스 공통)

- 공개 **대리 데이터** — 특정 기업 데이터 아님.
- **accuracy 착시 거부** — 불균형은 BER/recall, 측정은 불확도로 평가.
- **안 되는 것도 기록** — SMOTE·mutual_info·cost-sensitive 등 무효 시도를 숨기지 않음. 숫자를 부풀리지 않는다.

## 재현

```bash
pip install -r requirements.txt
python download_data.py      # SECOM (ucimlrepo / 직접 다운로드)
python case_B_ocd.py         # OCD inverse (NIST CSV 포함, 공개)
python case_C_secom_vm.py    # SECOM 불량탐지 + 위험도 순위
```
