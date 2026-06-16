* S and T frequencies can be investigated, see chat log for CST


* Wave transfer
  - add waveshape mode
  - dc block for both modes
  - lowpass

* Noisegrain:
  - num voices, rdev for wavfreq and grainrate
    - rdev has amp on rslider
    - low and hi freq on nsliders above rslider
  - source select wave
  - keep all in 1, or make 3 modules?
    - gainmask and chanmask can be nslider unmodulated
    - source select can be combobox, skip noiselevel
    - waveform wave needs samplepos
    - wave freq: different scale (on rslider display) for tone, noise, wave

* Audio gen idea
  - 3 bands of noise controlled by the temporal band activity
  - rhythmic pulse (granular like), so
    - pulse tempo
    - pulse length (and shape)
    - bandpass center freq (x3, separately for each generator)

* NoiseGrains refine:
  - Amp in dB (for all oscillators)
  - make playable and fun, haha
  - grain masking, sequence of 1,2,3,4,5 for gain and channel
  - bandpass cutoff and bandwidth, auto compensate level on bandwidth change
  - stereo width
  - master pan 
    - hard pan left right 
    - reduce width when pan is > 0.5 or < 0.5

* MAke 3x Noise grains

* Audio processing
  - divide 3 bands, set cutoff 1 and 2
  - separate pitch mod per band

* Waveform transfer
  - downsample (how much) ... to 512 samples?
  - transfer values via 16 sample chunks over OSC
    - make 8 of these OSC "channels" (= 128 samples)
    - assemble waveform in Csound (4x transfer to make complete waveform)
    - then crossfade from old to new

* Midi synth idea
  - Zyne harmonics, control pitch dev with wave param
    - same parm, per quadrant?
    - several parm globally

* Make modulator expansion as radio button, only one open at a time

* Contour:
  - skip contours with area > 10% of image, as preprocess before everything else

* Readme for key control switches etc




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

