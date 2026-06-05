* Audio gen idea
  - 3 bands of noise controlled by the temporal band activity
  - rhythmic pulse (granular like), so
    - pulse tempo
    - pulse length (and shape)
    - bandpass center freq (x3, separately for each generator)


* Flow for the quadrants
  - debug: print the x and y values for the globals next to the Hz label

* Contour:
  - skip contours with area > 10% of image, as preprocess before everything else

* Readme for key control switches etc

* Waveform transfer
  - downsample (how much) ... to 512 samples?
  - transfer values via 16 sample chunks over OSC
    - make 8 of these OSC "channels" (= 128 samples)
    - assemble waveform in Csound (4x transfer to make complete waveform)
    - then crossfade from old to new



* We are not using contours for anything are we?
  - how could they be useful?

* Make quadrants relate to mask
  - keep positions of gui displays, but mask the image array and divide it into quadrants

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

# todo audio
 
# multiple instances
Use the same instr definition for the 4 quadrants, just instantiate the instrument with a fractional instr number to keep the quadrant instances apart. Use a p-field to distinguish between quadrants (0,1,2,3,4) when calling the instrument. Then you can also use sprintf to create the chn names to receive the corresponding parameters. Like for example "spatial_low_1" (for upper left quadrant) can be made with 
Sspatial_low sprintf ""spatial_low_%i", p4
kspatial_low chnget Sspatial_low

