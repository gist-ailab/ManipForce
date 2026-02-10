# import cv2
# import cv2.aruco as aruco
# import numpy as np

# # A4 용지 크기 (픽셀 단위, 300 DPI 기준)
# A4_WIDTH, A4_HEIGHT = 2480, 3508

# # AprilTag 생성에 사용할 dictionary
# dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

# # 마커 이미지 크기와 간격 설정
# marker_size = 236   # 각 마커의 크기 (픽셀 단위, 약 2cm)
# margin = 80  

# # A4 용지 흰색 배경 생성
# a4_image = np.ones((A4_HEIGHT, A4_WIDTH), dtype=np.uint8) * 255

# # 4x4 마커 배열 생성
# marker_id = 0
# y_offset = margin
# for row in range(4):  # 4행
#     x_offset = margin
#     for col in range(4):  # 4열
#         # 마커 생성
#         marker_image = aruco.drawMarker(dictionary, marker_id, marker_size)

#         # A4 이미지에 마커 배치
#         a4_image[y_offset:y_offset + marker_size, x_offset:x_offset + marker_size] = marker_image

#         # 다음 마커 위치로 이동
#         x_offset += marker_size + margin
#         marker_id += 1

#     y_offset += marker_size + margin

# # 이미지 저장
# cv2.imwrite("board_img/A4_with_16_apriltag_markers.png", a4_image)
# print("3x3 AprilTag 마커가 배치된 A4 사이즈 마커 이미지가 생성되었습니다.")



# Make cube

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import cv2.aruco as aruco
import numpy as np
import os

# A4 용지 크기 설정 (픽셀 단위, 300 DPI)
A4_WIDTH, A4_HEIGHT = 2480, 3508  # 210mm x 297mm @ 300DPI

# 마커 설정
# border_size_mm = 58  # 검은 테두리 4.5cm
# marker_size_mm = 52  # 마커는 테두리보다 살짝 작게

border_size_mm = 59  # 검은 테두리 4.5cm
marker_size_mm = 53  # 마커는 테두리보다 살짝 작게

border_thickness_mm = 0.2  # 테두리 두께 2mm
dpi = 300  # 해상도
marker_size_px = int(marker_size_mm * dpi / 25.4)  # 마커 픽셀 크기
border_size_px = int(border_size_mm * dpi / 25.4)  # 테두리 픽셀 크기
border_thickness_px = int(border_thickness_mm * dpi / 25.4)  # 테두리 두께 픽셀

# A4 이미지 생성 (흰색 배경)
a4_image = np.ones((A4_HEIGHT, A4_WIDTH), dtype=np.uint8) * 255

# ArUco 마커 딕셔너리 선택
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# 마커 9개를 생성하고 A4 용지에 배치
num_markers = 5  # 총 마커 개수
margin = 80  # 마커 간 간격 (픽셀)

# 마커 배치 시작 위치 (좌측 상단 여백)
x_offset = (A4_WIDTH - (3 * border_size_px + 2 * margin)) // 2
y_offset = (A4_HEIGHT - (3 * border_size_px + 2 * margin)) // 2

# 마커 그리기
marker_id = 0
for row in range(3):  # 3행
    for col in range(3):  # 3열
        # 현재 위치 계산
        x = x_offset + col * (border_size_px + margin)
        y = y_offset + row * (border_size_px + margin)
        
        # 검은색 테두리를 그림 (-1 대신 border_thickness_px 사용)
        cv2.rectangle(a4_image, (x, y), (x + border_size_px, y + border_size_px), 0, border_thickness_px)
        
        # ArUco 마커 생성
        marker_image = aruco.drawMarker(dictionary, marker_id, marker_size_px)
        
        # 테두리 중앙에 마커를 배치하기 위한 오프셋 계산
        marker_offset = (border_size_px - marker_size_px) // 2
        
        # 마커를 테두리 중앙에 배치
        marker_x = x + marker_offset
        marker_y = y + marker_offset
        a4_image[marker_y:marker_y + marker_size_px, marker_x:marker_x + marker_size_px] = marker_image

        marker_id += 1
        print(marker_id)

# 출력 폴더 생성 및 이미지 저장
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "A4_with_6_aruco_markers.png")
cv2.imwrite(output_path, a4_image)

# 결과 확인
print(f"9개의 ArUco 마커가 A4 용지에 배치되어 저장되었습니다: {output_path}")

# 이미지 보기
cv2.imshow("A4 ArUco Markers", a4_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
