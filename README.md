# Search-Rescue
Search and rescue project for UAS software dept.
# UAS-DTU Round 2 – Software Task (Search & Rescue)

Problem Statement:
The UAV collects aerial images of a shipwreck. The task is to:
- Segment land vs ocean.
- Detect casualties (shapes represent age groups, colors represent medical condition).
- Detect rescue pads (circles).
- Assign casualties to pads while maximizing rescue priority ratio.

Casualty Priorities:
- Shape:  Star = 3, Triangle = 2, ◼Square = 1  
- Emergency: Severe = 3, Mild = 2,  Safe = 1  

Rescue Pads
- Blue Pad = 4 capacity  
- Pink Pad = 3 capacity  
- Grey Pad = 2 capacity  

---

Current Progress
I worked on this over 2 extra days (was not in delhi during the 5-day period).  
Here’s what I have completed so far:

Done:
- Segmentation of ocean (blue) vs land (green/brown) using HSV thresholding.  
- Detection of casualties (shapes + emergency color).  
- Detection of rescue pads (circles via Hough Transform).  
- Visualization of detections.  

Pending:
- Automatic assignment of casualties to pads (with capacities).  
- Distance matrix computation.  
- Calculation of rescue ratio **Pr**.  

---

Sample Outputs
- `segmented.png` → segmented land vs ocean.  
- `detections.png` → casualties + pads marked.  

---
 Tech Stack
- Python 3.13
- OpenCV (image processing, shape detection)
- NumPy (array operations, distances)
