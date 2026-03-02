import socket, struct
import numpy as np
import cv2

def recvall(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

HOST, PORT = "0.0.0.0", 5001

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"listening {HOST}:{PORT} ...")
    conn, addr = s.accept()
    print("connected:", addr)

    with conn:
        while True:
            hdr = recvall(conn, 12)  # [len u32][w u16][h u16][frameId u32]
            if hdr is None:
                print("disconnected")
                break

            length, w, h, frame_id = struct.unpack("!IHHI", hdr)
            data = recvall(conn, length)
            if data is None:
                print("disconnected")
                break

            # 1. 바이트 데이터를 NumPy 배열로 변환
            img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))

            # 2. RGBA를 BGR(OpenCV 기본 포맷)로 변환
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

            # 3. [추가된 부분] 상하 반전 (Y-Flip)
            # cv2.flip의 두 번째 인자가 0이면 상하 반전, 1이면 좌우 반전, -1이면 상하좌우 반전을 의미합니다.
            bgr = cv2.flip(bgr, 0)

            # 4. 화면 출력
            cv2.imshow("Quest PCA Stream", bgr)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
                break