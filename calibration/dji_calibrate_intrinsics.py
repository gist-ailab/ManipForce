#!/usr/bin/env python3
import os
import sys
import glob
import json
import argparse
from typing import List, Tuple


'''
python3 calibration/dji_calibrate_intrinsics.py --images "/home/ailab-2204/Workspace/gail-umi/test2/episode_1/images/*.jpg" --rows 6 --cols 9 --square 0.024 --show
'''



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="체스보드 이미지로 카메라 내적 파라미터 보정")
    p.add_argument("--images", type=str, default="test/episode_1/images/*.jpg", help="이미지 글롭 경로 (예: data/*.jpg)")
    p.add_argument("--rows", type=int, default=6, help="체스보드 내부 코너 세로 개수")
    p.add_argument("--cols", type=int, default=9, help="체스보드 내부 코너 가로 개수")
    p.add_argument("--square", type=float, default=0.024, help="한 칸(정사각형) 실제 한 변 길이(미터 등 단위)")
    p.add_argument("--out", type=str, default="calibration_intrinsics.json", help="결과 저장 파일(JSON)")
    p.add_argument("--show", action="store_true", help="감지 결과 시각화 표시")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths: List[str] = sorted(glob.glob(args.images))
    if not paths:
        print(f"이미지를 찾지 못했습니다: {args.images}")
        return 2
    print(f"[INFO] 이미지 {len(paths)}장 로드")

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        print("필요 패키지: pip install opencv-python numpy")
        print(e)
        return 3

    pattern_size: Tuple[int, int] = (args.cols, args.rows)
    # 월드 좌표계 상 코너 3D 좌표(체스보드를 z=0 평면으로 가정)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= float(args.square)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    vis_image = None
    good = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        # 서브픽셀 정제
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        good += 1
        if args.show:
            vis = cv2.drawChessboardCorners(img.copy(), pattern_size, corners2, True)
            vis_image = vis

    if good < 3:
        print(f"유효한 체스보드가 감지된 이미지가 너무 적습니다: {good}")
        return 4
    print(f"[INFO] 사용 이미지(체스보드 감지 성공): {good}장")

    image_size = (gray.shape[1], gray.shape[0])  # (width, height)

    # 카메라 보정
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
        flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_ZERO_TANGENT_DIST
    )
    print(f"[RESULT] 평균 재투영 오차: {ret:.4f}")
    print("[RESULT] K (내적행렬):")
    print(K)
    print("[RESULT] 왜곡계수 (k1,k2,p1,p2,k3,k4,k5,k6 ...):")
    print(dist.ravel())

    # 품질 확인을 위한 per-view reprojection error
    per_view_errors: List[float] = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        e = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        per_view_errors.append(float(e))
    print(f"[RESULT] 뷰별 재투영 오차 평균: {sum(per_view_errors)/len(per_view_errors):.4f}")

    # 저장
    result = {
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "board": {"rows": args.rows, "cols": args.cols, "square": float(args.square)},
        "reprojection_error": float(ret),
        "per_view_errors": per_view_errors,
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {args.out}")

    if args.show and vis_image is not None:
        cv2.imshow("Last corners", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())


