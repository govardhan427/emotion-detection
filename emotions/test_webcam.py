import cv2

for i in range(5):  # Try indexes 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Integrated webcam detected at index {i}")
        cap.release()
        break
else:
    print("❌ No integrated webcam found!")
