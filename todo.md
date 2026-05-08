* Separate contours based on movement
  - z and x switches seems to do nothing
  - static contrours should be colored yellow


# Spectral analysis
  - investigate and debug, verify it is correct and doing what I assume
  - what do we need to extract?
    - low/mid/hi band frequency
    - single most prominent frequency
    - centroid, based on bandpass 1Hz to 8Hz
    - centroid based on lo/mid/hi bands

# Temporal filter
  - suppress static elements, line reflections
    - suppress the line but keep the wave displacements

# Contour analysis
  - adapt threshold and parms when turning tdiff and temporal filt on or off
  - if tdiff off and threshold on, shouldn't we have clear contours?
    - contours does not trig here

# Optimization
  - put render in its own process
  
# Mask
  - make quadrants relative to mask