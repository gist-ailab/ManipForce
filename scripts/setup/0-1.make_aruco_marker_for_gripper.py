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

# 마커 설정 (small)
border_size_mm = 13  # 외부 테두리 21mm (실제 출력 시 21mm가 되도록 약간 크게 설정)
marker_size_mm = int(round(border_size_mm * (24/28)))  # 기존 비율 유지 (24/28)
border_thickness_mm = 0.2
dpi = 300  # 해상도

marker_size_px = int(marker_size_mm * dpi / 25.4)
border_size_px = int(border_size_mm * dpi / 25.4)
border_thickness_px = int(border_thickness_mm * dpi / 25.4)

# A4 이미지 생성 (흰색 배경)
a4_image = np.ones((A4_HEIGHT, A4_WIDTH), dtype=np.uint8) * 255

# ArUco 마커 딕셔너리 선택
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# 마커 간격 설정
margin = 100  # 마커 간 간격 (픽셀)

# 마커 배치 시작 위치 (왼쪽 상단 모서리)
x_offset = 100  # 왼쪽 여백
y_offset = 100  # 상단 여백

# 6세트 마커 그리기 (각 세트는 ID 6, 7)
set_count = 6
marker_ids = [6, 7]  # 한 세트당 마커 ID

for set_idx in range(set_count):
    # 현재 세트의 y 위치 계산 (2행 3열로 배치)
    row = set_idx // 3  # 0, 1 (2행)
    col = set_idx % 3   # 0, 1, 2 (3열)
    
    for i, marker_id in enumerate(marker_ids):
        # 현재 위치 계산
        x = x_offset + col * (2 * border_size_px + margin) + i * (border_size_px + 50)  # 세트 내 마커 간격 50px
        y = y_offset + row * (border_size_px + margin)
        
        # 검은색 테두리
        cv2.rectangle(a4_image, (x, y), (x + border_size_px, y + border_size_px), 0, border_thickness_px)
        
        # ArUco 마커 생성
        marker_image = aruco.drawMarker(dictionary, marker_id, marker_size_px)
        
        # 테두리 중앙에 마커를 배치
        marker_offset = (border_size_px - marker_size_px) // 2
        marker_x = x + marker_offset
        marker_y = y + marker_offset
        a4_image[marker_y:marker_y + marker_size_px, marker_x:marker_x + marker_size_px] = marker_image

# 출력 폴더 생성 및 이미지 저장
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "A4_with_2_aruco_markers.png")
cv2.imwrite(output_path, a4_image)

# 결과 확인
print(f"6세트(총 12개)의 13mm ArUco 마커가 A4 용지에 배치되어 저장되었습니다: {output_path}")

# 이미지 보기
cv2.imshow("A4 ArUco Markers", a4_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
