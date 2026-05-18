
* Test contour method across all videos, 
  - we wante contours detected where we can visually see them in the image
  - some videos, like Nidelv_2025_05_2.mp4 would not detect any contours

* What is act and speed of the flow analyses, do they both show in the arrows
  - perhaps act could be shown as line thickness?
  - is the speed shown with arrow length?



* Slomo mode 4 frames pr sec, update temporal analysis


# Contour analysis
  - adapt threshold and parms when turning tdiff and temporal filt on or off
  - if tdiff off and threshold on, shouldn't we have clear contours?
    - contours does not trig here

# Optimization
  - put render in its own process
  
# Mask
  - make quadrants relative to mask