import cv2
from deepface import DeepFace

# Try opening the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("📸 Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame.")
        break

    try:
        # Analyze the frame for emotion detection
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        # Overlay emotion text on the frame
        cv2.putText(frame,
                    f'Emotion: {emotion}',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)
    except Exception as e:
        print("⚠️ Error analyzing frame:", e)

    # Show the frame
    cv2.imshow("Facial Expression Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
