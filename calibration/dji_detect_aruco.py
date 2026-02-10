#!/usr/bin/env python3
import os
import sys
import json
import glob
import argparse
from typing import Tuple, Optional

'''
 python3 /home/geonhyup/Workspace/dji_connect/detect_aruco.py   --device /dev/video0   --dict DICT_6X6_250   --marker-size 0.04   --calib /home/geonhyup/Workspace/dji_connect/calibration_intrinsics.json   --undistort
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="보정 결과로 ArUco 검출 및 포즈 확인")
    p.add_argument("--device", type=str, default=None, help="비디오 디바이스 (예: /dev/video0). 기본: 첫 /dev/video*")
    p.add_argument("--dict", type=str, default="DICT_6X6_250", help="사전 (예: DICT_6X6_250)")
    p.add_argument("--marker-size", type=float, default=0.20, help="마커 한 변 실제 길이(미터)")
    p.add_argument("--calib", type=str, default="calibration_intrinsics.json", help="보정 결과 JSON 경로")
    p.add_argument("--undistort", action="store_true", help="왜곡 보정된 프레임을 사용/표시")
    return p.parse_args()


def find_first_video() -> str:
    vids = sorted(glob.glob("/dev/video*"))
    if not vids:
        raise SystemExit("비디오 장치를 찾지 못했습니다.")
    return vids[0]


def load_calibration(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"보정 파일을 찾지 못했습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    import numpy as np  # type: ignore
    K = np.array(data["K"], dtype="float64")
    dist = np.array(data["dist"], dtype="float64").reshape(-1, 1)
    width = int(data["image_size"]["width"])
    height = int(data["image_size"]["height"])
    return K, dist, (width, height)


def main() -> int:
    args = parse_args()
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        aruco = cv2.aruco  # type: ignore
    except Exception as e:
        print("ArUco 검출을 위해 opencv-contrib-python 패키지가 필요합니다.")
        print("설치: python3 -m pip install opencv-contrib-python numpy")
        print(e)
        return 2

    device = args.device or find_first_video()
    K, dist, image_size = load_calibration(args.calib)

    # 사전
    dict_name = args.dict.upper()
    if not hasattr(aruco, dict_name):
        print(f"알 수 없는 사전: {args.dict}")
        return 3
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    params = aruco.DetectorParameters_create()

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다: {device}")
        return 4
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

    win = f"ArUco Detect - {device} (q: 종료)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # 첫 프레임으로 현재 스트림 해상도 확인 후 K 스케일
    # 보정은 image_size 기준이므로, 해상도 불일치 시 K를 스케일링
    ok_probe, probe = cap.read()
    if not ok_probe or probe is None:
        print("첫 프레임을 획득하지 못했습니다.")
        return 5
    stream_h, stream_w = probe.shape[:2]
    calib_w, calib_h = image_size
    sx = float(stream_w) / float(calib_w)
    sy = float(stream_h) / float(calib_h)
    K_scaled = K.copy()
    # fx, cx는 가로 스케일, fy, cy는 세로 스케일
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy

    # undistort용 맵 생성(현재 스트림 해상도 기준)
    map1 = map2 = None
    if args.undistort:
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_scaled, (stream_w, stream_h), cv2.CV_16SC2)

    # 창 크기를 크게 조정(기본 1.5배, 최대 1920x1080)
    scale = 1
    new_w = min(int(stream_w * scale), 1920)
    new_h = min(int(stream_h * scale), 1080)
    try:
        cv2.resizeWindow(win, new_w, new_h)
    except Exception:
        pass

    print(f"[INFO] dict={args.dict}, marker_size={args.marker_size} m, calib={args.calib}")
    print(f"[INFO] calib_size={calib_w}x{calib_h}, stream_size={stream_w}x{stream_h}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            view = frame
            if args.undistort and map1 is not None:
                view = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

            gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)

            if ids is not None and len(ids) > 0:
                aruco.drawDetectedMarkers(view, corners, ids)
                # 현재 해상도에 맞춘 K로 포즈 추정
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, args.marker_size, K_scaled, dist)
                for rvec, tvec in zip(rvecs, tvecs):
                    aruco.drawAxis(view, K_scaled, dist, rvec, tvec, args.marker_size * 0.5)
                # 첫 마커 포즈를 콘솔에 간단히 출력
                r0 = rvecs[0].reshape(-1)
                t0 = tvecs[0].reshape(-1)
                print(f"[POSE] id={int(ids[0])} rvec={r0.round(3).tolist()} tvec(m)={t0.round(3).tolist()}")
            cv2.imshow(win, view)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())


