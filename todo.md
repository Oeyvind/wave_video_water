* Spatial and temporal centroid: renormalize it so that we get a full 0 to 1 range. What I mean here is that the centroid of the 3 bands will necessarily never go to zero and also never to 1, as there will be some activity in all bands. Can you suggest a renormalisation formula? Analyze the video example clips in C:\Projects\efx_experiments\wave_video_files to find reasonable clip valies for the renormalization. Rescale to fill the full 0 to 1 range. Use this value for the python visual display, and for the OSC sent centroid value. Do we use the current raw centroid to control the flow analysis cutoff between slow and fast flow?  If so, you will probably need to maintain the current centroid for that purpose, and then make a new variable for the renormalized centroid.
Do this operation for the quadrant centroids, and for the global one.




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
## Modulation mapping
For each synthesizer parameter, we want to be able to modulate them with parameters from the wave analysis. Set up a routing control system for each parameter, allowing 2 modulators to influence the synth parameter. Use a basic gui control to set the offset (or base value if you like), then modulate this value with the modulators we set up. Create a modulation mapping module containing a combobox to select the source parameter. Make a clip and rescale module with the parameters minimum and maximum clip value, and a small button to allow rescaling so the clipped parameter is stretched to 0-1 range (or -1 to 1 range if it was initially bipolar). Make a shape module where we raise the value to some exponent (set by a gui nslider). Then a gain control, allowing to set howe much this modulator will influence the synth parameter. This whole modulation module should be relatively compact in the gui. I'm hoping it can fit within a 100x100 pix area. Then we can use 2 such mod mapping modules for each of the synth parameters. The general idea for a layout would be to have a synth parameter on top, and then two mod modules underneath it. Create a dual rslider to set the base value f0r the parameter, where we can use the outer rslider to set the value, and the inner rslider (overlaid) to show the value after it has been affected by modulators. The csd file C:\Cabbage_VST\CabbageEfx\amp_switcher\centroid_splitter.csd has an example of such a dual rslider on line 4 and 5. 

# multiple instances
Use the same instr definition for the 4 quadrants, just instantiate the instrument with a fractional instr number to keep the quadrant instances apart. Use a p-field to distinguish between quadrants (0,1,2,3,4) when calling the instrument. Then you can also use sprintf to create the chn names to receive the corresponding parameters. Like for example "spatial_low_1" (for upper left quadrant) can be made with 
Sspatial_low sprintf ""spatial_low_%i", p4
kspatial_low chnget Sspatial_low

# GUI
Make a button to turn on/off the audio generator in instr 31 for each quadrant. A volume control in dB for each audio generator. 
Also look at the description above for additional parameters described there.