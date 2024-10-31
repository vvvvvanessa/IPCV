import cv2

# 初始化视频
cap = cv2.VideoCapture('output_video_stablized.mp4')

# 存储点击坐标
coordinates = []

def click_event(event, x, y, flags, param):
    """
    鼠标事件回调函数，用于捕获点击坐标
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键点击记录坐标
        print(f"Clicked at: ({x}, {y})")
        coordinates.append((x, y))
        # 显示点击位置
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Frame", frame)

# 设置窗口
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_event)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 显示帧
    cv2.imshow("Frame", frame)
    
    # 按键控制
    key = cv2.waitKey(0)  # 按任意键切换到下一帧
    if key == ord('q'):  # 按 'q' 键退出
        break

cap.release()
cv2.destroyAllWindows()

# 输出记录的坐标
print("Coordinates:", coordinates)
