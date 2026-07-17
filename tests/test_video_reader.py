import cv2

video_path = "data/windy/videos/Tab-7-9-2026_14_10.webm"

video = cv2.VideoCapture(video_path)

print("Opened:", video.isOpened())

count = 0

while True:

    success, frame = video.read()

    if not success:
        break

    count += 1

print("Frames actually read:", count)

video.release()