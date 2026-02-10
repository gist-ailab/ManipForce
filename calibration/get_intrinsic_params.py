import pyrealsense2 as rs
import json

# 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 스트리밍 시작
pipeline.start(config)

try:
    # 카메라에서 프레임을 가져와서 intrinsic matrix 추출
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # intrinsic 매트릭스를 JSON 형식으로 변환
    intrinsic_data = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "model": str(intrinsics.model),
        "coeffs": intrinsics.coeffs
    }

    # JSON 파일로 저장
    with open("intrinsic_matrix_d415.json", "w") as json_file:
        json.dump(intrinsic_data, json_file, indent=4)

    print("intrinsic_matrix.json 파일이 생성되었습니다.")

finally:
    # 파이프라인 종료
    pipeline.stop()
