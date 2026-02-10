import cv2
import numpy as np
import glob

# 체스보드 크기 설정 (행 x 열, 내부 코너 개수)
chessboard_size = (9, 6)

# 체스보드 각 코너의 실제 3D 좌표 준비 (Z는 0으로 설정)
square_size = 0.0233  # 체스보드 한 칸의 크기 (단위: 미터)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 3D 점들과 2D 점들을 저장할 리스트
objpoints = []  # 실제 3D 점
imgpoints = []  # 이미지에 투영된 2D 점

# 체스보드 이미지들이 저장된 디렉토리 (본인 경로로 변경)
images = glob.glob('calib_imgs/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너 그리기 및 시각화
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Calibration Image', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(distortion_coeffs)

    # 결과 저장
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coeffs": distortion_coeffs.tolist()
    }
    with open('camera_calibration_brio.json', 'w') as f:
        import json
        json.dump(calibration_data, f, indent=2)

    print("Calibration data saved to 'camera_calibration.json'")
else:
    print("Calibration failed!")
