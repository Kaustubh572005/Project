import cv2
from ultralytics import YOLO

# --------------------------
# FACE DETECTOR
# --------------------------

face_detector=cv2.CascadeClassifier(
'haarcascade_frontalface_default.xml'
)

# --------------------------
# YOLO OBJECT DETECTION
# --------------------------

model=YOLO("yolov8n.pt")


# --------------------------
# FINGERPRINT FUNCTION
# --------------------------

def fingerprint_match():

    img1=cv2.imread(
    "fingerprint/fp1.jpg",0
    )

    img2=cv2.imread(
    "fingerprint/fp2.jpg",0
    )

    orb=cv2.ORB_create()

    kp1,des1=orb.detectAndCompute(
    img1,None
    )

    kp2,des2=orb.detectAndCompute(
    img2,None
    )

    bf=cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches=bf.match(
        des1,
        des2
    )

    print(
    "Fingerprint Match Points:",
    len(matches)
    )

    output=cv2.drawMatches(
        img1,kp1,
        img2,kp2,
        matches[:30],
        None
    )

    cv2.imshow(
    "Fingerprint Matching",
    output
    )

    cv2.waitKey(0)



# --------------------------
# CAMERA
# --------------------------

cap=cv2.VideoCapture(0)

mode="face"

print("Press F Face Detection")
print("Press O Object/Fruit Detection")
print("Press P Fingerprint Match")
print("Press Q Quit")


while True:

    ret,frame=cap.read()

    key=cv2.waitKey(1)&0xFF

    if key==ord('f'):
        mode="face"

    elif key==ord('o'):
        mode="object"

    elif key==ord('p'):
        fingerprint_match()

    elif key==ord('q'):
        break


# -------------------
# FACE MODE
# -------------------

    if mode=="face":

        gray=cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces=face_detector.detectMultiScale(
            gray,
            1.3,
            5
        )

        for (x,y,w,h) in faces:

            cv2.rectangle(
                frame,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                3
            )

            cv2.putText(
                frame,
                "Face Detected",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )


# --------------------
# OBJECT MODE
# --------------------

    elif mode=="object":

        results=model(frame)

        frame=results[0].plot()


    cv2.imshow(
        "Smart Recognition System",
        frame
    )


cap.release()
cv2.destroyAllWindows()