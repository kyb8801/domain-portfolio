# Domain Portfolio — Uncertainty-Aware Metrology on Public Data

계측·MetroAI 방법을 **타깃 도메인 공개데이터**에 적용한 케이스 모음.
공통 무기 = 모든 결과에 **불확도**를 붙인다(남들이 segmentation/분류만 할 때).

## Core (재사용 불확도 레이어 — sandbox 검증 완료)
| 파일 | 역할 |
|---|---|
| `uncertainty.py` | GUM (combine / expand / budget / Monte Carlo) |
| `ml_inverse.py` | ensemble inverse + conformal 불확도 |
| `measurement_gate.py` | PASS / FAIL / RETEST 합부 게이트 (분포무관 보장) |
| `gum_posterior_bridge.py` | ML 불확도 → GUM U(k=2) 번역 |

## Cases
| | 도메인 (타깃) | 공개 데이터 | 상태 |
|---|---|---|---|
| B | 반도체 박막·OCD 계측 (삼성전기) | NIST L100P300 (보유) | `measurement_gate` + 기존 inverse |
| A | 배터리 양극재 정량 (LG·SDI·포스코) | NREL Battery Microstructure Library | `case_A_battery_porosity.py` (합성 검증, 실 tif 연결 예정) |
| C | 가상계측·이상탐지 (KITECH) | SECOM (UCI, CC BY 4.0) | `case_C_secom_vm.py` (합성 검증, 실데이터 연결 예정) |

## 데이터 받기 (네 PC — sandbox 는 UCI/NREL 접근 차단)
```bash
pip install -r requirements.txt
python download_data.py            # SECOM 자동 (ucimlrepo)
# 배터리: NREL 페이지에서 tif 다운로드 -> data/
```

## 정직 고지 (반드시 유지)
- 공개 **대리 데이터** — 특정 기업 데이터 아님.
- 차별점 = **불확도/보장**. segmentation·분류 자체는 레드오션.
- 현재 스캐폴드는 **합성 데이터로 검증**됨. 실데이터 연결은 진행 중.
- SECOM 은 분류(pass/fail) 데이터라, case_C 게이트를 분류용으로 조정 필요(다음 단계).
