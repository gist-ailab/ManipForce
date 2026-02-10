import cv2
import os
import sys
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="인터랙티브 카메라 동기화 뷰어")
    parser.add_argument("--episode_path", type=str, default="data/260209/episode_1", help="에피소드 경로")
    parser.add_argument("--synced", action="store_true", help="정렬된(synced) 폴더 사용 여부")
    parser.add_argument("--add_cam", action="store_true", help="추가 카메라(additional_cam) 포함 여부")
    args = parser.parse_args()

    # 경로 설정
    if args.synced:
        h_dir = os.path.join(args.episode_path, "images", "handeye")
        p_dir = os.path.join(args.episode_path, "images", "pose_tracking")
        a_dir = os.path.join(args.episode_path, "images", "additional_cam")
    else:
        h_dir = os.path.join(args.episode_path, "images", "raw_backup", "handeye")
        p_dir = os.path.join(args.episode_path, "images", "raw_backup", "pose_tracking")
        a_dir = os.path.join(args.episode_path, "images", "raw_backup", "additional_cam")

    # 기본 필수 경로 확인
    if not os.path.exists(h_dir) or not os.path.exists(p_dir):
        print(f"[ERROR] 필수 경로를 찾을 수 없습니다: {h_dir} 또는 {p_dir}")
        return

    # 공통 파일 찾기 (타임스탬프 기준)
    h_files = set(os.listdir(h_dir))
    p_files = set(os.listdir(p_dir))
    common_files = h_files.intersection(p_files)
    
    if args.add_cam:
        if os.path.exists(a_dir):
            a_files = set(os.listdir(a_dir))
            common_files = common_files.intersection(a_files)
        else:
            print(f"[WARNING] --add_cam이 요청되었으나 경로가 없습니다: {a_dir}")
            args.add_cam = False

    common_files = sorted(list(common_files))

    if not common_files:
        print("[ERROR] 공통된 타임스탬프 파일을 찾을 수 없습니다.")
        return

    print(f"[INFO] 총 {len(common_files)}개의 동기화된 프레임 발견")
    print("[Controls] A/Left: 이전, D/Right: 다음, Q: 종료")

    idx = 0
    win_name = "Sync Player (A/D to navigate, Q to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        fname = common_files[idx]
        img1 = cv2.imread(os.path.join(h_dir, fname))
        img2 = cv2.imread(os.path.join(p_dir, fname))
        img3 = cv2.imread(os.path.join(a_dir, fname)) if args.add_cam else None

        if img1 is None or img2 is None:
            print(f"이미지 로딩 실패: {fname}")
            idx = (idx + 1) % len(common_files)
            continue

        # 시각화를 위한 리사이즈 (높이 480 기준)
        target_h = 480
        def resize_h(img):
            h, w = img.shape[:2]
            return cv2.resize(img, (int(w * (target_h / h)), target_h))

        disp1 = resize_h(img1)
        disp2 = resize_h(img2)
        
        if args.add_cam and img3 is not None:
            disp3 = resize_h(img3)
            combined = np.hstack([disp1, disp2, disp3])
            footer = "Handeye | Pose Tracking | Additional"
        else:
            combined = np.hstack([disp1, disp2])
            footer = "Handeye | Pose Tracking"
        
        # 텍스트 추가
        info_text = f"[{idx+1}/{len(common_files)}] {fname}"
        cv2.putText(combined, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, footer, (20, target_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(win_name, combined)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27: break
        elif key == ord('d') or key == 83: idx = (idx + 1) % len(common_files)
        elif key == ord('a') or key == 81: idx = (idx - 1) % len(common_files)
        # Linux arrow keys
        if key == 83: idx = (idx + 1) % len(common_files)
        elif key == 81: idx = (idx - 1) % len(common_files)

    cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
