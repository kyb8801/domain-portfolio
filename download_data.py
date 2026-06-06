"""
공개 데이터 헬퍼 — 각 케이스. (sandbox 는 네트워크 차단, 네 PC 에서 실행)
SECOM(UCI, CC BY 4.0) 은 ucimlrepo 한 줄로. 배터리/OCD 는 주석 링크 참고.
"""
def get_secom():
    from ucimlrepo import fetch_ucirepo
    s = fetch_ucirepo(id=179)              # SECOM: 1567 x 590, pass(-1)/fail(1)
    return s.data.features, s.data.targets

# Case A 배터리: NREL Battery Microstructure Library (NMC, segmented 3D tif)
#   https://www.nrel.gov/transportation/battery-microstructure-library-data  -> data/ 에 저장 후 tifffile.imread
# Case B OCD: NIST L100P300 (이미 보유)

if __name__ == "__main__":
    X, y = get_secom()
    print("SECOM:", X.shape, y.shape)
