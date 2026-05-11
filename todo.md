* Make signal flow graph and simplify
  - serial signal patch with taps for the different stages/analyses
* Chack slow flow
roi_gray = source AND mask
preprocess_display = blur AND mask (are these simply addint two signals, or some binary processing?)
Otsu threshold on blur
edges masked, vs thresh masked

* For the spectral analysis and flow direction, I am really only interested in capturing different wave sizes and their movement. On a water surface, we see patterns of bumps. This creates almost like a texture, with fine grained patterns owhen we have ripples on the water, and larger and smoother pieces in the texture when we have larger waves. The whole spectral analysis section can be replaced if we find a method to identify wave size. The optical flow seems to be ale to capture flow direction, but it seems to depend on searching for flow within a specific freqiuency band or pattern size. If we have a separate pattern size detector, we can perhaps also adjust the optical flow to be sensitive to the prevalent wave size.  


* Spectral plots:
  - make 3 bands, display vertical bar for each, red/green/blue
    - temporal freq 1, 2.5, 6 Hz (Nyquist at 12 Hz)
    - spatial freq 4, 10, 20
    - centroid from these three

* Contour sensitivity
  - seems we can have clear thresholded regions without blob detection triggering


* Flow analysis, a slow moving wave might cross faster movements. The slow moving wave can be perceptually very visible. It does not showe on the flow arrow direction. Also, does the arow scale with flo amplitude/amount?

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