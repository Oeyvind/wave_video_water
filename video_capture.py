import cv2

def get_frame(cap, loop=False):
    ret, frame = cap.read()
    if not ret:
        if loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None
        else:
            return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray
