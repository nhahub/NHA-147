from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  

# Open webcam
cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Run detection
    results = model.predict(source=frame, conf=0.25, verbose=False)

    # Display predictions on frame
    annotated_frame = results[0].plot()  # this adds bounding boxes
    cv2.imshow("Waste Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()