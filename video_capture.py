import cv2


def get_frame(cap, loop=False, target_size=None):
    ret, frame = cap.read()
    if not ret:
        if loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None
        else:
            return None
    if target_size is not None:
        target_width, target_height = target_size
        if target_width > 0 and target_height > 0:
            current_height, current_width = frame.shape[:2]
            if current_width != target_width or current_height != target_height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray
