#!/usr/bin/env python3
import glob
import os
import sys
import subprocess
from typing import List, Optional


def find_video_devices() -> List[str]:
    devices = sorted(glob.glob("/dev/video*"))
    return devices


def which(cmd: str) -> Optional[str]:
    from shutil import which as _which
    return _which(cmd)


def run_ffplay(device: str) -> int:
    cmd = [
        "ffplay",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        "-vf", "setpts=PTS-STARTPTS",
        "-f", "v4l2",
        "-i", device,
    ]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def run_opencv(device: str) -> int:
    try:
        import cv2  # type: ignore
    except Exception:
        print("OpenCV가 설치되어 있지 않습니다. 다음 중 하나를 사용하세요:")
        print("- sudo apt install ffmpeg (ffplay 사용)")
        print("- python3 -m pip install opencv-python (OpenCV 미리보기 사용)")
        return 1

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    # 빠른 프레임 처리를 위해 설정 (가능한 경우)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 90)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다: {device}")
        return 2
    win = f"Preview - {device} (q: 종료, s: 저장 시작, x: 저장 중지)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # 저장 관련 상태
    saving = False
    saved_count = 0
    # 저장 경로: 리포지토리 내부 고정 경로
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "test", "episode_1", "images")
    os.makedirs(save_dir, exist_ok=True)
    # 주파수 측정 변수
    frame_count = 0
    last_report_time = __import__("time").time()
    monitor_interval = 5.0  # seconds

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if not saving:
                    saving = True
                    print(f"[SAVE] 시작: {save_dir}")
            elif key == ord('x'):
                if saving:
                    saving = False
                    print(f"[SAVE] 중지: 총 {saved_count}장 저장")
            # 저장 동작
            if saving:
                # 고유 파일명: 단조 증가 카운트 + ns 타임스탬프
                ts_ns = str(int(__import__("time").time() * 1e9)).split('.')[0]
                filename = f"frame_{saved_count:06d}_{ts_ns}.jpg"
                filepath = os.path.join(save_dir, filename)
                # JPG로 저장(기본 품질)
                try:
                    __import__("cv2").imwrite(filepath, frame)
                    saved_count += 1
                except Exception as e:
                    print(f"[SAVE] 실패: {e}")

            # 프레임 주파수 모니터링
            frame_count += 1
            now = __import__("time").time()
            if now - last_report_time >= monitor_interval:
                fps = frame_count / (now - last_report_time)
                print(f"[FPS] 최근 {monitor_interval:.1f}s 평균: {fps:.2f} Hz (총 {frame_count}프레임)")
                frame_count = 0
                last_report_time = now
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    devices = find_video_devices()
    if not devices:
        print("비디오 장치가 없습니다. 카메라를 UVC(웹캠) 모드로 전환했는지 확인하세요.")
        return 3
    preferred = "/dev/video2"
    device = os.environ.get("DJI_DEVICE", preferred if preferred in devices else devices[0])
    print(f"사용 장치: {device}")

    # 키 입력 및 저장 기능을 위해 OpenCV 경로를 기본으로 사용
    return run_opencv(device)


if __name__ == "__main__":
    sys.exit(main())


