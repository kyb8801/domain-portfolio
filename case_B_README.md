# Case B — OCD Scatterometry Inverse + Uncertainty (NIST L100P300)

광학 산란 스펙트럼에서 **선폭(CD)을 역산**하고, 측정 신뢰도를 불확도·합부 게이트로 평가한다. (삼성전기·반도체 박막 트랙)

## 데이터
- **NIST L100P300** (공개, data.nist.gov ark:/88434/mds2-2290). sim library 467×84, labels 467×3(bottom/middle/top nm), 실측 9 dies 9×84.

## 결과 (실데이터)
- **정확도**: sim library 5-fold CV, mid-CD inverse **RMSE 3.69 ± 0.40 nm**
- **실측 9 dies**: CD 108–112 nm, **U(k=2) = ±7.8 nm** (conformal 기반)
- **sim-to-real gap 포착**: 실측 epistemic ≈10 nm ≫ sim RMSE 3.7 nm → 측정이 라이브러리 분포 밖(OOD). 게이트가 9개 중 8개를 **RETEST** 로 분류.

## 핵심 — 불확도 레이어의 가치
점추정 OCD(상용 도구)는 라이브러리 매칭 CD 를 그대로 내놓는다. 여기서는 **measured 가 sim 분포 밖일 때 불확도가 커져 게이트가 RETEST(재측정/라이브러리 확장)를 요구**한다 — 과신해서 틀린 CD 를 출력하는 대신 "이 측정은 신뢰 못 함"을 신호한다. 이것이 traceable·calibrated 계측 AI 의 차별점.

## 정직 노트
- sim 정확도(3.69 nm)는 좋지만, 실측은 라벨이 없어 절대정확도를 잴 수 없고 **OOD 라 신뢰구간이 크다** — 숨기지 않고 드러낸다.
- 개선 방향: 라이브러리를 실측 분포로 확장(domain adaptation), RCWA forward 재보정, 측정 노이즈 모델 편입.

## 재현
```bash
pip install -r ../requirements.txt
python case_B_ocd.py     # NIST CSV 3개가 같은 폴더에 있어야 함 (repo 포함, 공개 데이터)
```
