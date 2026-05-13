

* Reduce from 5 bands to 3 bands pyramid
  - yellow with some bleed, blue and purple
  - would this significantly reduce cpu cost?

* What is act and speed of the flow analyses, do they both show in the arrows
  - perhaps act could be shown as line thickness?
  - is the speed shown with arrow length?

* Explain what the gabor and lbp analyses show

* Test and verify flow direction in several videos

* Optimize the pyramid analysis to fewer bands:
  fine, mid, coarse
  three first current bands are in the fine

* Slomo mode 4 frames pr sec, update temporal analysis


# Contour analysis
  - adapt threshold and parms when turning tdiff and temporal filt on or off
  - if tdiff off and threshold on, shouldn't we have clear contours?
    - contours does not trig here

# Optimization
  - put render in its own process
  
# Mask
  - make quadrants relative to mask