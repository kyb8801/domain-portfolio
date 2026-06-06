# Case C — SECOM Semiconductor Defect Detection (Risk-Ranking)

반도체 공정 센서 데이터(UCI SECOM)로 **불량을 탐지**하고, 전수검사가 불가능한 현실에서 **위험도 순위로 검사 예산을 배분**한다. (KITECH·공정 모니터링 트랙)

## 데이터 (정직)
- **UCI SECOM** (CC BY 4.0). 1567 × 590 센서, 불량 6.6%(104/1463), 결측 41,951셀.
- **공개 대리 데이터** — 특정 기업 데이터 아님.

## 방법
`StandardScaler → SelectKBest(25, f_classif) → LogisticRegression(balanced) → BER 최소 threshold → 위험도 순위(risk-ranking)`

## 결과 (5-fold 교차검증, 실데이터)
**분류**: BER **0.329 ± 0.042** | fail-recall **61%** | ROC-AUC **0.698**  (UCI baseline ~0.33)

**위험도 순위 운영** — 전수검사 대신 상위 q% 만 검사:
| 검사 예산 | 불량 포착 | 효율 (lift) |
|---|---|---|
| 상위 5% | 16% | **3.3×** |
| 상위 10% | 22% | 2.2× |
| **상위 20%** | **47%** | **2.4×** |
| 상위 30% | 61% | 2.0× |

→ *검사 예산 20%로 무작위 대비 2.4배 불량을 잡는다(fab 검사 ROI).*

## 정직 노트 (이게 핵심)
- **"정확도 94%"는 착시** — pass 가 94%라 전부 pass 로 찍어도 94%. 불균형은 **BER/recall** 로 평가. 첫 결과 BER 0.50(불량 0개)을 잡아낸 뒤 고쳤다.
- **모델**: 선형 LogReg 적합 (RF/GBT 는 BER 0.45 실패).
- **feature**: 590 → 25 (노이즈가 대부분, k=100+ 는 악화).
- **안 되는 것도 확인**: SMOTE·mutual_info·cost-sensitive threshold·결측-as-feature 모두 무효/역효과. 숫자를 부풀리지 않았다.
- **recall ~61% 가 분류 천장**(신호 한계) → **위험도 순위(운영 ROI)로 우회**. "합격/불합격"이 아니라 "검사 예산을 어디 쓰나"로 질문을 바꾼 것이 기여.

## 재현
```bash
pip install -r ../requirements.txt
python ../download_data.py     # SECOM -> data_secom_X/y.npy
python case_C_secom_vm.py      # 5-fold CV BER/recall/ROC-AUC + 위험도 순위
```
