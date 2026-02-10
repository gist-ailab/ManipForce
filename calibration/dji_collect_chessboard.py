#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import time
from typing import Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="체스보드 자동 수집기: 코너 감지 시만 저장")
    p.add_argument("--device", type=str, default=None, help="비디오 디바이스 (예: /dev/video0). 기본: 첫 /dev/video*")
    p.add_argument("--rows", type=int, default=6, help="체스보드 내부 코너 세로 개수")
    p.add_argument("--cols", type=int, default=9, help="체스보드 내부 코너 가로 개수")
    p.add_argument("--out", type=str, default="test2/episode_1/images", help="저장 디렉토리")
    p.add_argument("--interval", type=float, default=0.25, help="저장 최소 간격(초). 너무 자주 저장 방지")
    p.add_argument("--max", type=int, default=500, help="최대 저장 장수")
    p.add_argument("--draw", action="store_true", help="코너 감지 결과를 그려서 표시")
    return p.parse_args()


def find_first_video() -> str:
    vids = sorted(glob.glob("/dev/video*"))
    if not vids:
        raise SystemExit("비디오 장치를 찾지 못했습니다.")
    return vids[0]


def main() -> int:
    args = parse_args()
    device = args.device or find_first_video()
    rows, cols = args.rows, args.cols
    os.makedirs(args.out, exist_ok=True)

    try:
        import cv2  # type: ignore
    except Exception as e:
        print("OpenCV 필요: pip install opencv-python")
        print(e)
        return 2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다: {device}")
        return 3
    # 지연 최소화: 내부 버퍼 축소, 불필요한 RGB 변환 비활성화
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # 단일 채널 프레임로 들어와 변환 오류가 날 수 있으므로 RGB 변환은 활성화
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

    win = f"Collect Chessboard - {device} (q: 종료)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_save_t = 0.0
    saved = 0
    pattern_size: Tuple[int, int] = (cols, rows)
    # 빠른 탐지 플래그
    chess_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

    print(f"[INFO] 수집 시작: {device}, 패턴 {pattern_size}, 저장경로 {args.out}")
    print("[HINT] 체스보드를 다양한 거리/각도로 천천히 움직여 주세요.")
    try:
        while True:
            # 백로그 프레임 스킵으로 최신 프레임만 처리
            for _ in range(2):
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                continue
            # 프레임 채널 수에 따라 안전하게 그레이스케일 생성
            if frame is None:
                continue
            if len(frame.shape) == 2:
                gray = frame
            else:
                # shape: (H, W, C)
                c = frame.shape[2] if len(frame.shape) == 3 else 1
                if c == 1:
                    gray = frame[:, :, 0]
                elif c == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif c == 4:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                else:
                    # 알 수 없는 포맷은 안전하게 첫 채널 사용
                    gray = frame[..., 0]

            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=chess_flags)
            vis = frame
            if found:
                # 서브픽셀 정제
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                if args.draw:
                    vis = cv2.drawChessboardCorners(frame.copy(), pattern_size, corners, found)
                now = time.monotonic()
                if (now - last_save_t) >= args.interval and saved < args.max:
                    ts = int(time.time() * 1e3)
                    fname = f"chess_{saved:05d}_{ts}.jpg"
                    path = os.path.join(args.out, fname)
                    okw = cv2.imwrite(path, frame)
                    if okw:
                        saved += 1
                        last_save_t = now
                        if args.draw:
                            cv2.displayStatusBar(win, f"Saved {saved} -> {path}", 2000)
                        print(f"[SAVE] {path}")
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if saved >= args.max:
                print(f"[INFO] 최대 저장 장수({args.max}) 도달, 종료")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"[DONE] 총 {saved}장 저장")
    return 0


if __name__ == "__main__":
    sys.exit(main())


