"""
공개 데이터 헬퍼 — 네 PC 에서 실행 (sandbox 는 네트워크 차단).
SECOM(UCI, CC BY 4.0): ucimlrepo 가 환경에 따라 실패하므로 정적 zip 직접 다운로드를 기본으로.
"""
import io
import zipfile
import urllib.request
import numpy as np

SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"


def get_secom(save=True):
    """SECOM 직접 다운로드. returns X (1567,590, NaN 포함), y (1567,) in {-1(pass),1(fail)}."""
    raw = urllib.request.urlopen(SECOM_URL, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    data_f = [n for n in z.namelist() if n.endswith("secom.data")][0]
    lab_f = [n for n in z.namelist() if "labels" in n][0]
    X = np.genfromtxt(z.open(data_f))               # 공백 구분, NaN 포함
    y = np.genfromtxt(z.open(lab_f), usecols=0)      # 첫 컬럼 = -1/1
    if save:
        np.save("data_secom_X.npy", X)
        np.save("data_secom_y.npy", y)
    return X, y


def get_secom_via_ucimlrepo():
    """대안 경로(환경 되면)."""
    from ucimlrepo import fetch_ucirepo
    s = fetch_ucirepo(id=179)
    return s.data.features, s.data.targets


# Case A 배터리: NREL Battery Microstructure Library (NMC, segmented 3D tif)
#   https://www.nrel.gov/transportation/battery-microstructure-library-data  -> data/ 에 저장 후 tifffile.imread
# Case B OCD: NIST L100P300 (이미 보유)

if __name__ == "__main__":
    X, y = get_secom()
    print("SECOM:", X.shape, "labels", y.shape,
          "| fail:", int((y == 1).sum()), "pass:", int((y == -1).sum()),
          "| missing:", int(np.isnan(X).sum()))
    print("saved -> data_secom_X.npy / data_secom_y.npy")
