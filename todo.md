# Video files
  - rename new video files

# Spectral analysis
  - in slits display: show the amplitudes as intensity
    - for spatial frequency: use 4 slits, 2 horiz, 2 vertical
    - for temporal freq: use 4 points, in each quadrant
  - simplify display
  - what do we need to extract?
    - low/mid/hi band frequency
    - single most prominent frequency
    - centroid, based on bandpass 1Hz to 8Hz
    - centroid based on lo/mid/hi bands
  - make 4 spectral analyses, display for each quadrant
    - spatial freq quadrants still analyze the whole?
  - spatial freq does not show in graph
  * Spectral analysis is not based on multiple slits
    - display these slits if we use this method
    - OR make a 2D FFT (better, not much more expensive?)

# Temporal filter
  - simplify
  - suppress static elements, line reflections
    - suppress the line but keep the wave displacements

# Flow analysis
  - new resolution: 4 quadrants

# Contour analysis
  - adapt threshold and parms when turning tdiff and temporal filt on or off
  - if tdiff off and threshold on, shouldn't we have clear contours?
    - contours does not trig here

# Optimization
  - put render in its own process
  - tdiff and screeen uses a lot of CPU

# Mask
  - make quadrants relative to mask