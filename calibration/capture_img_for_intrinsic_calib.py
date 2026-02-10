import cv2
import os
from datetime import datetime

# 설정된 디렉토리 경로
save_path = 'calib_imgs'
image_count = 0

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

# USB 카메라 열기 (기본적으로 0번 카메라 사용)

cap = cv2.VideoCapture(5)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("s를 눌러 이미지를 저장하세요, 종료하려면 q를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # 카메라 프레임 출력
    cv2.imshow("Camera", frame)

    # 키 입력
    key = cv2.waitKey(1) & 0xFF

    # s를 눌러 이미지 저장
    if key == ord('s'):
        image_count += 1
        image_filename = os.path.join(save_path, f'calib_{image_count:03d}.png')
        cv2.imwrite(image_filename, frame)
        print(f"이미지 저장: {image_filename}")

    # q를 눌러 종료
    elif key == ord('q'):
        print("프로그램 종료")
        break

# 종료 시 모든 윈도우 닫기
cap.release()
cv2.destroyAllWindows()
