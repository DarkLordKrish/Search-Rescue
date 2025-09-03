import cv2
import numpy as np
import math


SHAPE_PRIO = {"star": 3, "triangle": 2, "square": 1}
EMERG_PRIO = {"red": 3, "yellow": 2, "green": 1}

# --- HSV thresholds for land/ocean ---
HSV_BLUE   = (np.array([90, 50, 50]), np.array([130, 255, 255]))  # ocean
HSV_LAND   = (np.array([20, 40, 40]), np.array([85, 255, 255]))   # land

# --- Casualty color thresholds ---
HSV_RED_1  = (np.array([0, 80, 80]), np.array([10, 255, 255]))
HSV_RED_2  = (np.array([170, 80, 80]), np.array([180, 255, 255]))
HSV_YELLOW = (np.array([20, 100, 100]), np.array([35, 255, 255]))
HSV_GREEN  = (np.array([45, 50, 50]), np.array([85, 255, 255]))

def segment_land_ocean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_ocean = cv2.inRange(hsv, HSV_BLUE[0], HSV_BLUE[1])
    mask_land  = cv2.inRange(hsv, HSV_LAND[0], HSV_LAND[1])
    # Overlay result
    overlay = img.copy()
    overlay[mask_ocean > 0] = (255, 0, 0)   # blue
    overlay[mask_land  > 0] = (0, 255, 0)   # green
    return mask_land, mask_ocean, overlay

def detect_casualties(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masks = {
        "red":    cv2.inRange(hsv, HSV_RED_1[0], HSV_RED_1[1]) | cv2.inRange(hsv, HSV_RED_2[0], HSV_RED_2[1]),
        "yellow": cv2.inRange(hsv, HSV_YELLOW[0], HSV_YELLOW[1]),
        "green":  cv2.inRange(hsv, HSV_GREEN[0], HSV_GREEN[1])
    }
    casualties = []
    for cname, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:   # skip noise
                continue
            # Approx shape
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03*peri, True)
            v = len(approx)
            if v == 3: shape = "triangle"
            elif v == 4: shape = "square"
            else: shape = "star"
            # Get center
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            casualties.append({
                "center": (cx, cy),
                "shape": shape,
                "emergency": cname,
                "shape_p": SHAPE_PRIO[shape],
                "emerg_p": EMERG_PRIO[cname]
            })
    return casualties

def detect_pads(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=10, maxRadius=0)
    pads = []
    if circles is not None:
        circles = np.uint16(np.around(circles[0,:]))
        for (x,y,r) in circles:
            pads.append({"center": (x,y), "r": r})
    return pads


if __name__ == "__main__":
    img = cv2.imread("1.png")


    land, ocean, seg = segment_land_ocean(img)
    cv2.imwrite("segmented.png", seg)

    casualties = detect_casualties(img)
    print("Detected casualties:")
    for c in casualties:
        print(c)


    pads = detect_pads(img)
    print("Detected rescue pads:", pads)


    vis = img.copy()
    for c in casualties:
        cv2.circle(vis, c["center"], 5, (0,0,255), -1)  # mark casualty
    for p in pads:
        cv2.circle(vis, p["center"], p["r"], (255,255,0), 2)  # mark pad
    cv2.imwrite("detections.png", vis)
