Now, make a Csound csd with instruments to receive the  parameters from the water surface analysis. 

# OSC receive
Collect all OSC receives in one instrument. Use instrument 10 for this, since we might need other instruments before it. The most interesting parameters are: 
- Spatial frequencies (lo, mid, high) from pyramid analysis
- Spatial centroid
- Temporal frequencies (lo mid high) from pyramid analysis
- Temporal centroid
- Wavelenght (WL), using the rescaling we also use for the display
- Flow direction (fast and slow), reshaped from angle to "up-ness" and "right-ness". Here, I mean that if the arrow is pointing upwards from the center, up-ness is positive, and increasing towards the arrow pointing straight up. Up-ness is negative when the arrow is pointing downwards. Right-ness is similar, with positive values when the arrow is pointing to the right, increasing so that right-ness is at max when the arrow is pointing straight right. When the arrow is pointing lefwards from the center, right-ness is negative. 
- Flow magnitude (fast and slow)
- For the lbp analysis, we want the compond measure smoothness (0 to 1 range), and chaos/order with a range of -1 to +1. -1 means full chaos, +1 means full order.
Receive all the OSC variables and write them to chn channels for consumption by the audio generation instruments. To distinguish between the global parameters and the 4 quadrants, use a naming convention: parameter_name_0 is for the global parameter, parameter_name_1 is for the upper left quadrant, then 2 is upper right, 3 is lower left, 4 is lower right.
Mix the 2 upper quadrant parameters (each of them) to a separate chn channel with the naming convention parameter_name_12 (representing 1 for UL and 2 of UR). Similarly make a sum for the two lower quadrant parameters, and also for the left side quadrants (1 and 3), and the right side quadrants (2 and 4)

# Gui handling
We will need to use instr 1 for gui handling, for example to turn sound generating instruments on or off and so on. Look at C:\Cabbage_VST\CabbageEfx\midiplugs\domen_ai\Rope\rope_midi.csd for how I like to do this. Instr 1 is on line 405 pp in that file. Here, we also use a UDO (ButtonEvent) for the gui-button-to-instr-event mechanism. 

# Audio synthesis and audio processing instruments
We will develop many variants of audio synthesis and live audio processing that use the wave analysis parameters. Let's put the sound generating instruments starting on instr 31 onwards, with one separate instrument per audio generation technique. The first instrument to explore will be relatively simple, and perhaps mostly for debugging and signal flow inspection. Here, we will use 3 sine tone oscillators initially tuned to tuned to the notes A (110Hz), E (fifth above) and C# (an octave and a major third above the A). Include gui nsøiders to allow changing these frequencies in the range 50 Hz to 1000Hz. Add three noise generators with a bandpass filter tuned to the same frequencies as the sine the generators. The width of the bandpass should be adjustable in the range 0.02 to 0.5 octaves. Find an amplitude compensation formula that allows the noise bandpass output to compensate for the varying filter width, so the output of the bandpassed noise is comparable in amplitude to the output of the sine tone generators. Then add a parameter that adjusts the sine/noise balance. 
So the instr has 3x these gui controllable parameters:
- Pitch 
- Fine tune pitch (range 3 semitones)
- Noise/Tone balance
- Noise bandwidth
- Stereo pan
- Amp
Make such and instrument for each quadrant. Use the same initial frequencies for the lower left and lower right quadrant synthesizer, but hard pan them left and right in the stereo image. Use the same pitches an octave higher for the upper two quadrants. 

## Modulation mapping
For each synthesizer parameter, we want to be able to modulate them with parameters from the wave analysis. Set up a routing control system for each parameter, allowing 2 modulators to influence the synth parameter. Use a basic gui control to set the offset (or base value if you like), then modulate this value with the modulators we set up. Create a modulation mapping module containing a combobox to select the source parameter. Make a clip and rescale module with the parameters minimum and maximum clip value, and a small button to allow rescaling so the clipped parameter is stretched to 0-1 range (or -1 to 1 range if it was initially bipolar). Make a shape module where we raise the value to some exponent (set by a gui nslider). Then a gain control, allowing to set howe much this modulator will influence the synth parameter. This whole modulation module should be relatively compact in the gui. I'm hoping it can fit within a 100x100 pix area. Then we can use 2 such mod mapping modules for each of the synth parameters. The general idea for a layout would be to have a synth parameter on top, and then two mod modules underneath it. Create a dual rslider to set the base value f0r the parameter, where we can use the outer rslider to set the value, and the inner rslider (overlaid) to show the value after it has been affected by modulators. The csd file C:\Cabbage_VST\CabbageEfx\amp_switcher\centroid_splitter.csd has an example of such a dual rslider on line 4 and 5. 

# multiple instances
Use the same instr definition for the 4 quadrants, just instantiate the instrument with a fractional instr number to keep the quadrant instances apart. Use a p-field to distinguish between quadrants (0,1,2,3,4) when calling the instrument. Then you can also use sprintf to create the chn names to receive the corresponding parameters. Like for example "spatial_low_1" (for upper left quadrant) can be made with 
Sspatial_low sprintf ""spatial_low_%i", p4
kspatial_low chnget Sspatial_low

# GUI
Make a button to turn on/off the audio generator in instr 31 for each quadrant. A volume control in dB for each audio generator. 
Also look at the description above for additional parameters described there.