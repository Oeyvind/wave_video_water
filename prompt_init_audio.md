Now, make a Csound csd with instruments to receive the  parameters from the water surface analysis. 

# OSC receive
Collect all OSC receives in one instrument. Use instrument 10 for this, since we might need other instruments before it. Include these parameters: 
- Spatial frequencies (lo, mid, high) from pyramid analysis
- Spatial centroid
- Temporal frequencies (lo mid high) from pyramid analysis
- Temporal centroid
- Wavelenght (WL), using the rescaling we also use for the display
- Flow direction (fast and slow), reshaped from polar to cartesian coordinates (angle converted to x and y)
- Flow magnitude (fast and slow)
- For the lbp analysis, we want the compond measure smoothness (0 to 1 range), and chaos/order with a range of -1 to +1. -1 means full chaos, +1 means full order.

Receive all the OSC variables and write them to chn channels for consumption by the audio generation instruments. To distinguish between the global parameters and the 4 quadrants, use a naming convention: parameter_name_0 is for the global parameter, parameter_name_1 is for the upper left quadrant, then 2 is upper right, 3 is lower left, 4 is lower right.
Mix the 2 upper quadrant parameters (each of them) to a separate chn channel with the naming convention parameter_name_12 (representing 1 for UL and 2 of UR). Similarly make a sum for the two lower quadrant parameters, and also for the left side quadrants (1 and 3), and the right side quadrants (2 and 4)

For OSC send, try to pack several floats into one OSC message whenever possible. The OSClisten in Csound can reliably receive 16 floats. 

# Gui handling
We will need to use instr 1 for gui handling, for example to turn sound generating instruments on or off and so on. Look at C:\Cabbage_VST\CabbageEfx\midiplugs\domen_ai\Rope\rope_midi.csd for how I like to do this. Instr 1 is on line 405 pp in that file. Here, we also use a UDO (ButtonEvent) for the gui-button-to-instr-event mechanism. 

# Audio synthesis and audio processing instruments
We will develop many variants of audio synthesis and live audio processing that use the wave analysis parameters. Let's put the sound generating instruments starting on instr 31 onwards, with one separate instrument per audio generation technique. The first instrument to explore will be relatively simple, and perhaps mostly for debugging and signal flow inspection. Here, we will use 3 sine tone oscillators. Include gui rsliders to allow changing these frequencies in the range 50 Hz to 1000Hz. Add three noise generators with a bandpass filter tuned to the same frequencies as the sine the generators. The width of the bandpass should be adjustable in the range 0.02 to 0.5 octaves. Find an amplitude compensation formula that allows the noise bandpass output to compensate for the varying filter width, so the output of the bandpassed noise is comparable in amplitude to the output of the sine tone generators. Then add a parameter that adjusts the sine/noise balance. 
So the instr has 3x these gui controllable parameters:
- Pitch 
- Noise/Tone balance
- Noise bandwidth
- Stereo pan
- Amp
Make a groupbox for the gui controls of this instrument, with the rsliders arranged horizontally. Make each rslider of size 70 pix.
Also make a gui button to turn the instrument on/off, patched as described under Gui handling

