<Cabbage>
form caption("Wave Synth") size(1260, 700), guiMode("queue"), pluginId("wSyn"), colour(20, 20, 20)

; --- Instrument 31 controls ---

groupbox bounds(5, 5, 1250, 290), channel("modGroup"), text("Instr31: 3x Sine + Bandpassed Noise + Mod Routing"), colour(58, 58, 58), fontColour(220, 220, 220), outlineColour(92, 92, 92) {

button bounds(3, 50, 36, 24), channel("inst31_on"), text("On"), value(0), colour:0("#3c4652"), colour:1("#2ecc71"), fontColour("white")
button bounds(3, 78, 36, 18), channel("mod_collapse"), text("+","-"), value(1), colour:0(60,30,30), colour:1(30,60,30), fontColour("white")
button bounds(32, 142, 6, 34), channel("master_reset"), value(0), colour:0(100,20,20), colour:1(200,40,40)
button bounds(41, 142, 6, 34), channel("master_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
; Source menu reused on every modulation module.
; 1=None, then incoming OSC channels/mixes.

; Module 1: v1_pitch
rslider channel("v1_pitch_view"), bounds(48, 25, 70, 70), text(""), range(50, 1000, 160, 1, 0.01), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v1_pitch_base"), bounds(48, 29, 70, 70), text("Pitch"), range(50, 1000, 160, 1, 0.01), trackerColour(40,80,200)
combobox bounds(48, 102, 85, 18), channel("m_v1_pitch_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(48, 122, 85, 18), channel("m_v1_pitch_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(48, 142, 50, 16), channel("m_v1_pitch_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(100, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 160, 50, 16), channel("m_v1_pitch_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(100, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(99, 142, 6, 34), channel("m_v1_pitch_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(48, 178, 50, 16), channel("m_v1_pitch_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(100, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 196, 50, 16), channel("m_v1_pitch_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(48, 214, 50, 16), channel("m_v1_pitch_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(100, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(100, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 232, 50, 16), channel("m_v1_pitch_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(100, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(48, 250, 50, 16), channel("m_v1_pitch_mode"), value(1), text("Add","Mul")
label bounds(100, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; Module 2: v1_bal
rslider channel("v1_bal_view"), bounds(140, 25, 70, 70), text(""), range(0, 1, 0.2, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v1_bal_base"), bounds(140, 29, 70, 70), text("Tone/noise"), range(0, 1, 0.2, 1, 0.001), trackerColour(40,80,200)
combobox bounds(140, 102, 85, 18), channel("m_v1_bal_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(140, 122, 85, 18), channel("m_v1_bal_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(140, 142, 50, 16), channel("m_v1_bal_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(140, 160, 50, 16), channel("m_v1_bal_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(191, 142, 6, 34), channel("m_v1_bal_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(140, 178, 50, 16), channel("m_v1_bal_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(140, 196, 50, 16), channel("m_v1_bal_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(140, 214, 50, 16), channel("m_v1_bal_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(140, 232, 50, 16), channel("m_v1_bal_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(140, 250, 50, 16), channel("m_v1_bal_mode"), value(1), text("Add","Mul")

; Module 3: v1_bw
rslider channel("v1_bw_view"), bounds(232, 25, 70, 70), text(""), range(0.02, 0.5, 0.12, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v1_bw_base"), bounds(232, 29, 70, 70), text("BW1"), range(0.02, 0.5, 0.12, 1, 0.001), trackerColour(40,80,200)
combobox bounds(232, 102, 85, 18), channel("m_v1_bw_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(232, 122, 85, 18), channel("m_v1_bw_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(232, 142, 50, 16), channel("m_v1_bw_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(284, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 160, 50, 16), channel("m_v1_bw_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(284, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(283, 142, 6, 34), channel("m_v1_bw_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(232, 178, 50, 16), channel("m_v1_bw_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(284, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 196, 50, 16), channel("m_v1_bw_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(232, 214, 50, 16), channel("m_v1_bw_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(284, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(284, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 232, 50, 16), channel("m_v1_bw_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(284, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(232, 250, 50, 16), channel("m_v1_bw_mode"), value(1), text("Add","Mul")
label bounds(284, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; Module 4: v1_amp
rslider channel("v1_amp_view"), bounds(324, 25, 70, 70), text(""), range(0, 1, 0.25, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v1_amp_base"), bounds(324, 29, 70, 70), text("Amp1"), range(0, 1, 0.25, 1, 0.001), trackerColour(40,80,200)
combobox bounds(324, 102, 85, 18), channel("m_v1_amp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(324, 122, 85, 18), channel("m_v1_amp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(324, 142, 50, 16), channel("m_v1_amp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(324, 160, 50, 16), channel("m_v1_amp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(375, 142, 6, 34), channel("m_v1_amp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(324, 178, 50, 16), channel("m_v1_amp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(324, 196, 50, 16), channel("m_v1_amp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(324, 214, 50, 16), channel("m_v1_amp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(324, 232, 50, 16), channel("m_v1_amp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(324, 250, 50, 16), channel("m_v1_amp_mode"), value(1), text("Add","Mul")

; Module 5: v2_pitch
rslider channel("v2_pitch_view"), bounds(416, 25, 70, 70), text(""), range(50, 1000, 280, 1, 0.01), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v2_pitch_base"), bounds(416, 29, 70, 70), text("Pitch2"), range(50, 1000, 280, 1, 0.01), trackerColour(40,80,200)
combobox bounds(416, 102, 85, 18), channel("m_v2_pitch_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(416, 122, 85, 18), channel("m_v2_pitch_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(416, 142, 50, 16), channel("m_v2_pitch_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(468, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 160, 50, 16), channel("m_v2_pitch_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(468, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(467, 142, 6, 34), channel("m_v2_pitch_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(416, 178, 50, 16), channel("m_v2_pitch_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(468, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 196, 50, 16), channel("m_v2_pitch_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(416, 214, 50, 16), channel("m_v2_pitch_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(468, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(468, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 232, 50, 16), channel("m_v2_pitch_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(468, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(416, 250, 50, 16), channel("m_v2_pitch_mode"), value(1), text("Add","Mul")
label bounds(468, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; Module 6: v2_bal
rslider channel("v2_bal_view"), bounds(508, 25, 70, 70), text(""), range(0, 1, 0.25, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v2_bal_base"), bounds(508, 29, 70, 70), text("Tone/noise2"), range(0, 1, 0.25, 1, 0.001), trackerColour(40,80,200)
combobox bounds(508, 102, 85, 18), channel("m_v2_bal_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(508, 122, 85, 18), channel("m_v2_bal_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(508, 142, 50, 16), channel("m_v2_bal_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(508, 160, 50, 16), channel("m_v2_bal_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(559, 142, 6, 34), channel("m_v2_bal_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(508, 178, 50, 16), channel("m_v2_bal_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(508, 196, 50, 16), channel("m_v2_bal_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(508, 214, 50, 16), channel("m_v2_bal_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(508, 232, 50, 16), channel("m_v2_bal_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(508, 250, 50, 16), channel("m_v2_bal_mode"), value(1), text("Add","Mul")

; Module 7: v2_bw
rslider channel("v2_bw_view"), bounds(600, 25, 70, 70), text(""), range(0.02, 0.5, 0.12, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v2_bw_base"), bounds(600, 29, 70, 70), text("BW2"), range(0.02, 0.5, 0.12, 1, 0.001), trackerColour(40,80,200)
combobox bounds(600, 102, 85, 18), channel("m_v2_bw_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(600, 122, 85, 18), channel("m_v2_bw_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(600, 142, 50, 16), channel("m_v2_bw_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(652, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 160, 50, 16), channel("m_v2_bw_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(652, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(651, 142, 6, 34), channel("m_v2_bw_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(600, 178, 50, 16), channel("m_v2_bw_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(652, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 196, 50, 16), channel("m_v2_bw_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(600, 214, 50, 16), channel("m_v2_bw_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(652, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(652, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 232, 50, 16), channel("m_v2_bw_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(652, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(600, 250, 50, 16), channel("m_v2_bw_mode"), value(1), text("Add","Mul")
label bounds(652, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; Module 8: v2_amp
rslider channel("v2_amp_view"), bounds(692, 25, 70, 70), text(""), range(0, 1, 0.2, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v2_amp_base"), bounds(692, 29, 70, 70), text("Amp2"), range(0, 1, 0.2, 1, 0.001), trackerColour(40,80,200)
combobox bounds(692, 102, 85, 18), channel("m_v2_amp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(692, 122, 85, 18), channel("m_v2_amp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(692, 142, 50, 16), channel("m_v2_amp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(692, 160, 50, 16), channel("m_v2_amp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(743, 142, 6, 34), channel("m_v2_amp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(692, 178, 50, 16), channel("m_v2_amp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(692, 196, 50, 16), channel("m_v2_amp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(692, 214, 50, 16), channel("m_v2_amp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(692, 232, 50, 16), channel("m_v2_amp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(692, 250, 50, 16), channel("m_v2_amp_mode"), value(1), text("Add","Mul")

; Module 9: v3_pitch
rslider channel("v3_pitch_view"), bounds(784, 25, 70, 70), text(""), range(50, 1000, 520, 1, 0.01), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v3_pitch_base"), bounds(784, 29, 70, 70), text("Pitch3"), range(50, 1000, 520, 1, 0.01), trackerColour(40,80,200)
combobox bounds(784, 102, 85, 18), channel("m_v3_pitch_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(784, 122, 85, 18), channel("m_v3_pitch_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(784, 142, 50, 16), channel("m_v3_pitch_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(836, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 160, 50, 16), channel("m_v3_pitch_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(836, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(835, 142, 6, 34), channel("m_v3_pitch_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(784, 178, 50, 16), channel("m_v3_pitch_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(836, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 196, 50, 16), channel("m_v3_pitch_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(784, 214, 50, 16), channel("m_v3_pitch_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(836, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(836, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 232, 50, 16), channel("m_v3_pitch_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(836, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(784, 250, 50, 16), channel("m_v3_pitch_mode"), value(1), text("Add","Mul")
label bounds(836, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; Module 10: v3_bal
rslider channel("v3_bal_view"), bounds(876, 25, 70, 70), text(""), range(0, 1, 0.3, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v3_bal_base"), bounds(876, 29, 70, 70), text("Tone/noise3"), range(0, 1, 0.3, 1, 0.001), trackerColour(40,80,200)
combobox bounds(876, 102, 85, 18), channel("m_v3_bal_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(876, 122, 85, 18), channel("m_v3_bal_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(876, 142, 50, 16), channel("m_v3_bal_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(876, 160, 50, 16), channel("m_v3_bal_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(927, 142, 6, 34), channel("m_v3_bal_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(876, 178, 50, 16), channel("m_v3_bal_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(876, 196, 50, 16), channel("m_v3_bal_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(876, 214, 50, 16), channel("m_v3_bal_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(876, 232, 50, 16), channel("m_v3_bal_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(876, 250, 50, 16), channel("m_v3_bal_mode"), value(1), text("Add","Mul")

; Module 11: v3_bw
rslider channel("v3_bw_view"), bounds(968, 25, 70, 70), text(""), range(0.02, 0.5, 0.12, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v3_bw_base"), bounds(968, 29, 70, 70), text("BW3"), range(0.02, 0.5, 0.12, 1, 0.001), trackerColour(40,80,200)
combobox bounds(968, 102, 85, 18), channel("m_v3_bw_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(968, 122, 85, 18), channel("m_v3_bw_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(968, 142, 50, 16), channel("m_v3_bw_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(1020, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(968, 160, 50, 16), channel("m_v3_bw_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(1020, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(1019, 142, 6, 34), channel("m_v3_bw_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(968, 178, 50, 16), channel("m_v3_bw_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(1020, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(968, 196, 50, 16), channel("m_v3_bw_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(968, 214, 50, 16), channel("m_v3_bw_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(1020, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(1020, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(968, 232, 50, 16), channel("m_v3_bw_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(1020, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(968, 250, 50, 16), channel("m_v3_bw_mode"), value(1), text("Add","Mul")
label bounds(1020, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; Module 12: v3_amp
rslider channel("v3_amp_view"), bounds(1060, 25, 70, 70), text(""), range(0, 1, 0.18, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("v3_amp_base"), bounds(1060, 29, 70, 70), text("Amp3"), range(0, 1, 0.18, 1, 0.001), trackerColour(40,80,200)
combobox bounds(1060, 102, 85, 18), channel("m_v3_amp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(1060, 122, 85, 18), channel("m_v3_amp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(1060, 142, 50, 16), channel("m_v3_amp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(1060, 160, 50, 16), channel("m_v3_amp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(1111, 142, 6, 34), channel("m_v3_amp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(1060, 178, 50, 16), channel("m_v3_amp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(1060, 196, 50, 16), channel("m_v3_amp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(1060, 214, 50, 16), channel("m_v3_amp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(1060, 232, 50, 16), channel("m_v3_amp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(1060, 250, 50, 16), channel("m_v3_amp_mode"), value(1), text("Add","Mul")

; Module 13: master pan
rslider channel("master_pan_view"), bounds(1152, 25, 70, 70), text(""), range(0, 1, 0.5, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("master_pan_base"), bounds(1152, 29, 70, 70), text("Pan"), range(0, 1, 0.5, 1, 0.001), trackerColour(40,80,200)
combobox bounds(1152, 102, 85, 18), channel("m_master_pan_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC")
combobox bounds(1152, 122, 85, 18), channel("m_master_pan_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(1152, 142, 50, 16), channel("m_master_pan_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(1204, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(1152, 160, 50, 16), channel("m_master_pan_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(1204, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(1203, 142, 6, 34), channel("m_master_pan_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(1152, 178, 50, 16), channel("m_master_pan_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(1204, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(1152, 196, 50, 16), channel("m_master_pan_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(1152, 214, 50, 16), channel("m_master_pan_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(1204, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(1204, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(1152, 232, 50, 16), channel("m_master_pan_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(1204, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(1152, 250, 50, 16), channel("m_master_pan_mode"), value(1), text("Add","Mul")
label bounds(1204, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
}

csoundoutput bounds(5, 500, 1250, 200)

</Cabbage>

<CsoundSynthesizer>
<CsOptions>
-n -d
</CsOptions>
<CsInstruments>

sr = 48000
ksmps = 32
nchnls = 2
0dbfs = 1

gihOsc OSCinit 8000

opcode ButtonEvent, 0, Sii
    Schan, iInstr, iMode xin
    kval chnget Schan
    ktrig changed kval
    if (ktrig == 1) then
        if (kval > 0.5) then
            event "i", iInstr, 0, -1
        else
            turnoff2 iInstr, 0, 1
        endif
    endif
endop

opcode UpdateMixSums, 0, S
    Sbase xin
    S1 sprintfk "%s_1", Sbase
    S2 sprintfk "%s_2", Sbase
    S3 sprintfk "%s_3", Sbase
    S4 sprintfk "%s_4", Sbase

    S12 sprintfk "%s_12", Sbase
    S34 sprintfk "%s_34", Sbase
    S13 sprintfk "%s_13", Sbase
    S24 sprintfk "%s_24", Sbase

    k1 chnget S1
    k2 chnget S2
    k3 chnget S3
    k4 chnget S4

    chnset (k1 + k2) * 0.5, S12
    chnset (k3 + k4) * 0.5, S34
    chnset (k1 + k3) * 0.5, S13
    chnset (k2 + k4) * 0.5, S24
endop

opcode GetAreaSuffix, S, k
    karea xin
    Ssuffix init "_0"

    if (karea < 1.5) then
        Ssuffix strcpyk "_0"
    elseif (karea < 2.5) then
        Ssuffix strcpyk "_1"
    elseif (karea < 3.5) then
        Ssuffix strcpyk "_2"
    elseif (karea < 4.5) then
        Ssuffix strcpyk "_3"
    elseif (karea < 5.5) then
        Ssuffix strcpyk "_4"
    elseif (karea < 6.5) then
        Ssuffix strcpyk "_12"
    elseif (karea < 7.5) then
        Ssuffix strcpyk "_34"
    elseif (karea < 8.5) then
        Ssuffix strcpyk "_13"
    else
        Ssuffix strcpyk "_24"
    endif

    xout Ssuffix
endop

opcode GetModSource01, k, kk
    ksel, karea xin
    Ssuffix GetAreaSuffix karea
    Schan init ""
    ksrc init 0

    if (ksel < 1.5) then
        ksrc = 0
    elseif (ksel < 2.5) then
        Schan sprintfk "s_lo%s", Ssuffix
    elseif (ksel < 3.5) then
        Schan sprintfk "s_mid%s", Ssuffix
    elseif (ksel < 4.5) then
        Schan sprintfk "s_high%s", Ssuffix
    elseif (ksel < 5.5) then
        Schan sprintfk "t_lo%s", Ssuffix
    elseif (ksel < 6.5) then
        Schan sprintfk "t_mid%s", Ssuffix
    elseif (ksel < 7.5) then
        Schan sprintfk "t_high%s", Ssuffix
    elseif (ksel < 8.5) then
        Schan sprintfk "s_centr%s", Ssuffix
    elseif (ksel < 9.5) then
        Schan sprintfk "t_centr%s", Ssuffix
    elseif (ksel < 10.5) then
        Schan sprintfk "wl%s", Ssuffix
    elseif (ksel < 11.5) then
        Schan sprintfk "flow_fast_mag%s", Ssuffix
    elseif (ksel < 12.5) then
        Schan sprintfk "flow_slow_mag%s", Ssuffix
    elseif (ksel < 13.5) then
        Schan sprintfk "flow_fast_x%s", Ssuffix
    elseif (ksel < 14.5) then
        Schan sprintfk "flow_fast_y%s", Ssuffix
    elseif (ksel < 15.5) then
        Schan sprintfk "flow_slow_x%s", Ssuffix
    elseif (ksel < 16.5) then
        Schan sprintfk "flow_slow_y%s", Ssuffix
    elseif (ksel < 17.5) then
        Schan sprintfk "lbp_smooth%s", Ssuffix
    else
        Schan sprintfk "lbp_orderchaos%s", Ssuffix
    endif

    if (ksel >= 1.5) then
        ksrc chnget Schan
        if (ksel >= 12.5 && ksel < 16.5) then
            ksrc = (ksrc + 1.0) * 0.5
        elseif (ksel >= 17.5 && ksel < 18.5) then
            ksrc = (ksrc + 1.0) * 0.5
        endif
    endif

    xout ksrc
endop

opcode GetModSourceName, S, kk
    ksel, karea xin
    Ssuffix GetAreaSuffix karea
    Schan init "none"

    if (ksel < 1.5) then
        Schan sprintfk "%s", "none"
    elseif (ksel < 2.5) then
        Schan sprintfk "s_lo%s", Ssuffix
    elseif (ksel < 3.5) then
        Schan sprintfk "s_mid%s", Ssuffix
    elseif (ksel < 4.5) then
        Schan sprintfk "s_high%s", Ssuffix
    elseif (ksel < 5.5) then
        Schan sprintfk "t_lo%s", Ssuffix
    elseif (ksel < 6.5) then
        Schan sprintfk "t_mid%s", Ssuffix
    elseif (ksel < 7.5) then
        Schan sprintfk "t_high%s", Ssuffix
    elseif (ksel < 8.5) then
        Schan sprintfk "s_centr%s", Ssuffix
    elseif (ksel < 9.5) then
        Schan sprintfk "t_centr%s", Ssuffix
    elseif (ksel < 10.5) then
        Schan sprintfk "wl%s", Ssuffix
    elseif (ksel < 11.5) then
        Schan sprintfk "flow_fast_mag%s", Ssuffix
    elseif (ksel < 12.5) then
        Schan sprintfk "flow_slow_mag%s", Ssuffix
    elseif (ksel < 13.5) then
        Schan sprintfk "flow_fast_x%s", Ssuffix
    elseif (ksel < 14.5) then
        Schan sprintfk "flow_fast_y%s", Ssuffix
    elseif (ksel < 15.5) then
        Schan sprintfk "flow_slow_x%s", Ssuffix
    elseif (ksel < 16.5) then
        Schan sprintfk "flow_slow_y%s", Ssuffix
    elseif (ksel < 17.5) then
        Schan sprintfk "lbp_smooth%s", Ssuffix
    else
        Schan sprintfk "lbp_orderchaos%s", Ssuffix
    endif

    xout Schan
endop

opcode ApplyMod, k, Skkkkkkkkkkk
    Sparam, kbase, ksel, karea, kmin, kmax, kexp, koffs, kgain, kmode, kcal, klp xin
    ksrc GetModSource01 ksel, karea

    Smin sprintfk "m_%s_min", Sparam
    Smax sprintfk "m_%s_max", Sparam
    kcalOn trigger kcal, 0.5, 0
    kautoMin init 0
    kautoMax init 1
    if (kcalOn > 0) then
        kautoMin = ksrc
        kautoMax = ksrc
    endif
    if (kcal > 0.5) then
        if (ksrc < kautoMin) then
            kautoMin = ksrc
        endif
        if (ksrc > kautoMax) then
            kautoMax = ksrc
        endif
        cabbageSetValue Smin, kautoMin, 1
        cabbageSetValue Smax, kautoMax, 1
    endif

    kclipMin = (kcal > 0.5 ? kautoMin : limit(kmin, 0, 1))
    kclipMax = (kcal > 0.5 ? kautoMax : limit(kmax, 0, 1))
    if (kclipMax < kclipMin) then
        ktmp = kclipMax
        kclipMax = kclipMin
        kclipMin = ktmp
    endif
    kclip = limit(ksrc, kclipMin, kclipMax)
    kden = max(1e-6, kclipMax - kclipMin)
    knorm = (kclip - kclipMin) / kden
    kshape = pow(knorm, kexp)
    kpost = kshape + koffs
    khtim = 0.1103 / max(klp, 0.001)
    klpout1 portk kpost, khtim
    klpout2 portk klpout1, khtim
    kpost = klpout2
    if (kgain == 0) then
        kout = kbase
    else
        if (kmode < 1.5) then
            kout = kbase + (kpost * kgain)
        else
            kout = kbase * (1.0 + (kpost * kgain))
        endif
    endif

    xout kout
endop

; GUI handling and button -> instrument events
instr 1
    kmod_col  chnget "mod_collapse"
    ktrig_col changed kmod_col
    if ktrig_col == 1 then
        if kmod_col > 0.5 then
            cabbageSet ktrig_col, "modGroup", "bounds(5, 5, 1250, 290)"
        else
            cabbageSet ktrig_col, "modGroup", "bounds(5, 5, 1250, 100)"
        endif
    endif

    ButtonEvent "inst31_on", 31, 0

    kmaster_reset chnget "master_reset"
    ktrig_mr changed kmaster_reset
    if (ktrig_mr > 0 && kmaster_reset > 0.5) then
        cabbageSetValue "m_v1_pitch_min",  0, 1
        cabbageSetValue "m_v1_pitch_max",  1, 1
        cabbageSetValue "m_v1_bal_min",    0, 1
        cabbageSetValue "m_v1_bal_max",    1, 1
        cabbageSetValue "m_v1_bw_min",     0, 1
        cabbageSetValue "m_v1_bw_max",     1, 1
        cabbageSetValue "m_v1_amp_min",    0, 1
        cabbageSetValue "m_v1_amp_max",    1, 1
        cabbageSetValue "m_v2_pitch_min",  0, 1
        cabbageSetValue "m_v2_pitch_max",  1, 1
        cabbageSetValue "m_v2_bal_min",    0, 1
        cabbageSetValue "m_v2_bal_max",    1, 1
        cabbageSetValue "m_v2_bw_min",     0, 1
        cabbageSetValue "m_v2_bw_max",     1, 1
        cabbageSetValue "m_v2_amp_min",    0, 1
        cabbageSetValue "m_v2_amp_max",    1, 1
        cabbageSetValue "m_v3_pitch_min",  0, 1
        cabbageSetValue "m_v3_pitch_max",  1, 1
        cabbageSetValue "m_v3_bal_min",    0, 1
        cabbageSetValue "m_v3_bal_max",    1, 1
        cabbageSetValue "m_v3_bw_min",     0, 1
        cabbageSetValue "m_v3_bw_max",     1, 1
        cabbageSetValue "m_v3_amp_min",    0, 1
        cabbageSetValue "m_v3_amp_max",    1, 1
        cabbageSetValue "m_master_pan_min", 0, 1
        cabbageSetValue "m_master_pan_max", 1, 1
        cabbageSetValue "master_reset", 0, 1
    endif

    kmaster_cal chnget "master_cal"
    ktrig_mc changed kmaster_cal
    if (ktrig_mc > 0) then
        kv1p  chnget "m_v1_pitch_src"
        kv1b  chnget "m_v1_bal_src"
        kv1bw chnget "m_v1_bw_src"
        kv1a  chnget "m_v1_amp_src"
        kv2p  chnget "m_v2_pitch_src"
        kv2b  chnget "m_v2_bal_src"
        kv2bw chnget "m_v2_bw_src"
        kv2a  chnget "m_v2_amp_src"
        kv3p  chnget "m_v3_pitch_src"
        kv3b  chnget "m_v3_bal_src"
        kv3bw chnget "m_v3_bw_src"
        kv3a  chnget "m_v3_amp_src"
        kpans chnget "m_master_pan_src"
        if (kmaster_cal > 0.5) then
            if (kv1p > 1.5) then
                cabbageSetValue "m_v1_pitch_cal", 1, 1
            endif
            if (kv1b > 1.5) then
                cabbageSetValue "m_v1_bal_cal", 1, 1
            endif
            if (kv1bw > 1.5) then
                cabbageSetValue "m_v1_bw_cal", 1, 1
            endif
            if (kv1a > 1.5) then
                cabbageSetValue "m_v1_amp_cal", 1, 1
            endif
            if (kv2p > 1.5) then
                cabbageSetValue "m_v2_pitch_cal", 1, 1
            endif
            if (kv2b > 1.5) then
                cabbageSetValue "m_v2_bal_cal", 1, 1
            endif
            if (kv2bw > 1.5) then
                cabbageSetValue "m_v2_bw_cal", 1, 1
            endif
            if (kv2a > 1.5) then
                cabbageSetValue "m_v2_amp_cal", 1, 1
            endif
            if (kv3p > 1.5) then
                cabbageSetValue "m_v3_pitch_cal", 1, 1
            endif
            if (kv3b > 1.5) then
                cabbageSetValue "m_v3_bal_cal", 1, 1
            endif
            if (kv3bw > 1.5) then
                cabbageSetValue "m_v3_bw_cal", 1, 1
            endif
            if (kv3a > 1.5) then
                cabbageSetValue "m_v3_amp_cal", 1, 1
            endif
            if (kpans > 1.5) then
                cabbageSetValue "m_master_pan_cal", 1, 1
            endif
        else
            cabbageSetValue "m_v1_pitch_cal",  0, 1
            cabbageSetValue "m_v1_bal_cal",    0, 1
            cabbageSetValue "m_v1_bw_cal",     0, 1
            cabbageSetValue "m_v1_amp_cal",    0, 1
            cabbageSetValue "m_v2_pitch_cal",  0, 1
            cabbageSetValue "m_v2_bal_cal",    0, 1
            cabbageSetValue "m_v2_bw_cal",     0, 1
            cabbageSetValue "m_v2_amp_cal",    0, 1
            cabbageSetValue "m_v3_pitch_cal",  0, 1
            cabbageSetValue "m_v3_bal_cal",    0, 1
            cabbageSetValue "m_v3_bw_cal",     0, 1
            cabbageSetValue "m_v3_amp_cal",    0, 1
            cabbageSetValue "m_master_pan_cal", 0, 1
        endif
    endif
endin

; OSC receive + channel routing/mixes
instr 10
    ; Predeclare OSC target variables to avoid parser ambiguity.
    kslo0 init 0
    ksmid0 init 0
    kshigh0 init 0
    ktlo0 init 0
    ktmid0 init 0
    kthigh0 init 0
    ksc0 init 0
    ktc0 init 0
    kwl0 init 0
    kslo1 init 0
    ksmid1 init 0
    kshigh1 init 0
    ktlo1 init 0
    ktmid1 init 0
    kthigh1 init 0
    ksc1 init 0
    ktc1 init 0
    kwl1 init 0
    kslo2 init 0
    ksmid2 init 0
    kshigh2 init 0
    ktlo2 init 0
    ktmid2 init 0
    kthigh2 init 0
    ksc2 init 0
    ktc2 init 0
    kwl2 init 0
    kslo3 init 0
    ksmid3 init 0
    kshigh3 init 0
    ktlo3 init 0
    ktmid3 init 0
    kthigh3 init 0
    ksc3 init 0
    ktc3 init 0
    kwl3 init 0
    kslo4 init 0
    ksmid4 init 0
    kshigh4 init 0
    ktlo4 init 0
    ktmid4 init 0
    kthigh4 init 0
    ksc4 init 0
    ktc4 init 0
    kwl4 init 0
    kfdeg init 0
    kfmag init 0
    ksdeg init 0
    ksmag init 0
    kls0 init 0
    klco0 init 0
    klco1 init 0
    klco2 init 0
    klco3 init 0
    klco4 init 0
    klsg init 0
    kls1 init 0
    kls2 init 0
    kls3 init 0
    kls4 init 0

    kfcoh init 0
    kscoh init 0
    kfq1d init 0
    kfq1a init 0
    kfq2d init 0
    kfq2a init 0
    kfq3d init 0
    kfq3a init 0
    kfq4d init 0
    kfq4a init 0
    ksq1d init 0
    ksq1a init 0
    ksq2d init 0
    ksq2a init 0
    ksq3d init 0
    ksq3a init 0
    ksq4d init 0
    ksq4a init 0

    ; Packed pyramid global: s0,s1,s2,t0,t1,t2,centroid_s,centroid_t
read_pyr_global:
    ktr OSClisten gihOsc, "/wave/pyramid/global/pack", "ffffffff", kslo0, ksmid0, kshigh0, ktlo0, ktmid0, kthigh0, ksc0, ktc0
    if (ktr == 1) then
        chnset kslo0, "s_lo_0"
        chnset ksmid0, "s_mid_0"
        chnset kshigh0, "s_high_0"
        chnset ktlo0, "t_lo_0"
        chnset ktmid0, "t_mid_0"
        chnset kthigh0, "t_high_0"
        chnset ksc0, "s_centr_0"
        chnset ktc0, "t_centr_0"
        kgoto read_pyr_global
    endif

    ; Packed pyramid quadrants: s0,s1,s2,t0,t1,t2,centroid_s,centroid_t
    read_pyr_ul:
    ktr OSClisten gihOsc, "/wave/pyramid/ul/pack", "ffffffff", kslo1, ksmid1, kshigh1, ktlo1, ktmid1, kthigh1, ksc1, ktc1
    if (ktr == 1) then
        chnset kslo1, "s_lo_1"
        chnset ksmid1, "s_mid_1"
        chnset kshigh1, "s_high_1"
        chnset ktlo1, "t_lo_1"
        chnset ktmid1, "t_mid_1"
        chnset kthigh1, "t_high_1"
        chnset ksc1, "s_centr_1"
        chnset ktc1, "t_centr_1"
        kgoto read_pyr_ul
    endif

    read_pyr_ur:
    ktr OSClisten gihOsc, "/wave/pyramid/ur/pack", "ffffffff", kslo2, ksmid2, kshigh2, ktlo2, ktmid2, kthigh2, ksc2, ktc2
    if (ktr == 1) then
        chnset kslo2, "s_lo_2"
        chnset ksmid2, "s_mid_2"
        chnset kshigh2, "s_high_2"
        chnset ktlo2, "t_lo_2"
        chnset ktmid2, "t_mid_2"
        chnset kthigh2, "t_high_2"
        chnset ksc2, "s_centr_2"
        chnset ktc2, "t_centr_2"
        kgoto read_pyr_ur
    endif

    read_pyr_ll:
    ktr OSClisten gihOsc, "/wave/pyramid/ll/pack", "ffffffff", kslo3, ksmid3, kshigh3, ktlo3, ktmid3, kthigh3, ksc3, ktc3
    if (ktr == 1) then
        chnset kslo3, "s_lo_3"
        chnset ksmid3, "s_mid_3"
        chnset kshigh3, "s_high_3"
        chnset ktlo3, "t_lo_3"
        chnset ktmid3, "t_mid_3"
        chnset kthigh3, "t_high_3"
        chnset ksc3, "s_centr_3"
        chnset ktc3, "t_centr_3"
        kgoto read_pyr_ll
    endif

    read_pyr_lr:
    ktr OSClisten gihOsc, "/wave/pyramid/lr/pack", "ffffffff", kslo4, ksmid4, kshigh4, ktlo4, ktmid4, kthigh4, ksc4, ktc4
    if (ktr == 1) then
        chnset kslo4, "s_lo_4"
        chnset ksmid4, "s_mid_4"
        chnset kshigh4, "s_high_4"
        chnset ktlo4, "t_lo_4"
        chnset ktmid4, "t_mid_4"
        chnset kthigh4, "t_high_4"
        chnset ksc4, "s_centr_4"
        chnset ktc4, "t_centr_4"
        kgoto read_pyr_lr
    endif

    ; Packed wavelength global + UL/UR/LL/LR
read_wl_pack:
    ktr OSClisten gihOsc, "/wave/wavelength/pack", "fffff", kwl0, kwl1, kwl2, kwl3, kwl4
    if (ktr == 1) then
        chnset kwl0, "wl_0"
        chnset kwl1, "wl_1"
        chnset kwl2, "wl_2"
        chnset kwl3, "wl_3"
        chnset kwl4, "wl_4"
        kgoto read_wl_pack
    endif

    ; Packed flow: direction_deg, activity, coherence
read_flow_fast:
    ktr OSClisten gihOsc, "/wave/flow/fast_pack", "fff", kfdeg, kfmag, kfcoh
    if (ktr == 1) then
        kfrad = kfdeg * $M_PI / 180
        chnset cos(kfrad) / 2 + 0.5, "flow_fast_x_0"
        chnset sin(kfrad) / 2 + 0.5, "flow_fast_y_0"
        chnset kfmag, "flow_fast_mag_0"
        kgoto read_flow_fast
    endif

read_flow_slow:
    ktr OSClisten gihOsc, "/wave/flow/slow_pack", "fff", ksdeg, ksmag, kscoh
    if (ktr == 1) then
        ksrad = ksdeg * $M_PI / 180
        chnset cos(ksrad) / 2 + 0.5, "flow_slow_x_0"
        chnset sin(ksrad) / 2 + 0.5, "flow_slow_y_0"
        chnset ksmag, "flow_slow_mag_0"
        kgoto read_flow_slow
    endif

    ; Per-quadrant fast flow: [UL_dir, UL_act, UR_dir, UR_act, LL_dir, LL_act, LR_dir, LR_act]
read_flow_fast_quad:
    ktr OSClisten gihOsc, "/wave/flow/fast_quad_pack", "ffffffff", kfq1d, kfq1a, kfq2d, kfq2a, kfq3d, kfq3a, kfq4d, kfq4a
    if (ktr == 1) then
        kfq1r = kfq1d * $M_PI / 180
        chnset cos(kfq1r) / 2 + 0.5, "flow_fast_x_1"
        chnset sin(kfq1r) / 2 + 0.5, "flow_fast_y_1"
        chnset kfq1a, "flow_fast_mag_1"
        kfq2r = kfq2d * $M_PI / 180
        chnset cos(kfq2r) / 2 + 0.5, "flow_fast_x_2"
        chnset sin(kfq2r) / 2 + 0.5, "flow_fast_y_2"
        chnset kfq2a, "flow_fast_mag_2"
        kfq3r = kfq3d * $M_PI / 180
        chnset cos(kfq3r) / 2 + 0.5, "flow_fast_x_3"
        chnset sin(kfq3r) / 2 + 0.5, "flow_fast_y_3"
        chnset kfq3a, "flow_fast_mag_3"
        kfq4r = kfq4d * $M_PI / 180
        chnset cos(kfq4r) / 2 + 0.5, "flow_fast_x_4"
        chnset sin(kfq4r) / 2 + 0.5, "flow_fast_y_4"
        chnset kfq4a, "flow_fast_mag_4"
        kgoto read_flow_fast_quad
    endif

    ; Per-quadrant slow flow: [UL_dir, UL_act, UR_dir, UR_act, LL_dir, LL_act, LR_dir, LR_act]
read_flow_slow_quad:
    ktr OSClisten gihOsc, "/wave/flow/slow_quad_pack", "ffffffff", ksq1d, ksq1a, ksq2d, ksq2a, ksq3d, ksq3a, ksq4d, ksq4a
    if (ktr == 1) then
        ksq1r = ksq1d * $M_PI / 180
        chnset cos(ksq1r) / 2 + 0.5, "flow_slow_x_1"
        chnset sin(ksq1r) / 2 + 0.5, "flow_slow_y_1"
        chnset ksq1a, "flow_slow_mag_1"
        ksq2r = ksq2d * $M_PI / 180
        chnset cos(ksq2r) / 2 + 0.5, "flow_slow_x_2"
        chnset sin(ksq2r) / 2 + 0.5, "flow_slow_y_2"
        chnset ksq2a, "flow_slow_mag_2"
        ksq3r = ksq3d * $M_PI / 180
        chnset cos(ksq3r) / 2 + 0.5, "flow_slow_x_3"
        chnset sin(ksq3r) / 2 + 0.5, "flow_slow_y_3"
        chnset ksq3a, "flow_slow_mag_3"
        ksq4r = ksq4d * $M_PI / 180
        chnset cos(ksq4r) / 2 + 0.5, "flow_slow_x_4"
        chnset sin(ksq4r) / 2 + 0.5, "flow_slow_y_4"
        chnset ksq4a, "flow_slow_mag_4"
        kgoto read_flow_slow_quad
    endif

    ; Packed LBP global + UL/UR/LL/LR
read_lbp_smooth:
    ktr OSClisten gihOsc, "/wave/lbp/smooth_pack", "fffff", klsg, kls1, kls2, kls3, kls4
    if (ktr == 1) then
        chnset klsg, "lbp_smooth_0"
        chnset kls1, "lbp_smooth_1"
        chnset kls2, "lbp_smooth_2"
        chnset kls3, "lbp_smooth_3"
        chnset kls4, "lbp_smooth_4"
        kgoto read_lbp_smooth
    endif

read_lbp_orderchaos:
    ktr OSClisten gihOsc, "/wave/lbp/orderchaos_pack", "fffff", klco0, klco1, klco2, klco3, klco4
    if (ktr == 1) then
        chnset klco0, "lbp_orderchaos_0"
        chnset klco1, "lbp_orderchaos_1"
        chnset klco2, "lbp_orderchaos_2"
        chnset klco3, "lbp_orderchaos_3"
        chnset klco4, "lbp_orderchaos_4"
        kgoto read_lbp_orderchaos
    endif

    ; Build side/row means for all key parameters
    UpdateMixSums "s_lo"
    UpdateMixSums "s_mid"
    UpdateMixSums "s_high"
    UpdateMixSums "t_lo"
    UpdateMixSums "t_mid"
    UpdateMixSums "t_high"
    UpdateMixSums "s_centr"
    UpdateMixSums "t_centr"
    UpdateMixSums "wl"
    UpdateMixSums "flow_fast_x"
    UpdateMixSums "flow_fast_y"
    UpdateMixSums "flow_fast_mag"
    UpdateMixSums "flow_slow_x"
    UpdateMixSums "flow_slow_y"
    UpdateMixSums "flow_slow_mag"
    UpdateMixSums "lbp_smooth"
    UpdateMixSums "lbp_orderchaos"
endin

instr 31
    ; Base controls
    kf1b chnget "v1_pitch_base"
    kb1b chnget "v1_bal_base"
    kw1b chnget "v1_bw_base"
    ka1b chnget "v1_amp_base"

    kf2b chnget "v2_pitch_base"
    kb2b chnget "v2_bal_base"
    kw2b chnget "v2_bw_base"
    ka2b chnget "v2_amp_base"

    kf3b chnget "v3_pitch_base"
    kb3b chnget "v3_bal_base"
    kw3b chnget "v3_bw_base"
    ka3b chnget "v3_amp_base"

    kpanb chnget "master_pan_base"

    ; Mod router controls
    ks chnget "m_v1_pitch_src"
    kmin chnget "m_v1_pitch_min"
    kmax chnget "m_v1_pitch_max"
    kexp chnget "m_v1_pitch_exp"
    ko chnget "m_v1_pitch_offs"
    kg chnget "m_v1_pitch_gain"
    ka chnget "m_v1_pitch_area"
    km chnget "m_v1_pitch_mode"
    kc chnget "m_v1_pitch_cal"
    klp chnget "m_v1_pitch_lp"
    kf1 ApplyMod "v1_pitch", kf1b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v1_bal_src"
    kmin chnget "m_v1_bal_min"
    kmax chnget "m_v1_bal_max"
    kexp chnget "m_v1_bal_exp"
    ko chnget "m_v1_bal_offs"
    kg chnget "m_v1_bal_gain"
    ka chnget "m_v1_bal_area"
    km chnget "m_v1_bal_mode"
    kc chnget "m_v1_bal_cal"
    klp chnget "m_v1_bal_lp"
    kb1 ApplyMod "v1_bal", kb1b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v1_bw_src"
    kmin chnget "m_v1_bw_min"
    kmax chnget "m_v1_bw_max"
    kexp chnget "m_v1_bw_exp"
    ko chnget "m_v1_bw_offs"
    kg chnget "m_v1_bw_gain"
    ka chnget "m_v1_bw_area"
    km chnget "m_v1_bw_mode"
    kc chnget "m_v1_bw_cal"
    klp chnget "m_v1_bw_lp"
    kw1 ApplyMod "v1_bw", kw1b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v1_amp_src"
    kmin chnget "m_v1_amp_min"
    kmax chnget "m_v1_amp_max"
    kexp chnget "m_v1_amp_exp"
    ko chnget "m_v1_amp_offs"
    kg chnget "m_v1_amp_gain"
    ka chnget "m_v1_amp_area"
    km chnget "m_v1_amp_mode"
    kc chnget "m_v1_amp_cal"
    klp chnget "m_v1_amp_lp"
    ka1 ApplyMod "v1_amp", ka1b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v2_pitch_src"
    kmin chnget "m_v2_pitch_min"
    kmax chnget "m_v2_pitch_max"
    kexp chnget "m_v2_pitch_exp"
    ko chnget "m_v2_pitch_offs"
    kg chnget "m_v2_pitch_gain"
    ka chnget "m_v2_pitch_area"
    km chnget "m_v2_pitch_mode"
    kc chnget "m_v2_pitch_cal"
    klp chnget "m_v2_pitch_lp"
    kf2 ApplyMod "v2_pitch", kf2b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v2_bal_src"
    kmin chnget "m_v2_bal_min"
    kmax chnget "m_v2_bal_max"
    kexp chnget "m_v2_bal_exp"
    ko chnget "m_v2_bal_offs"
    kg chnget "m_v2_bal_gain"
    ka chnget "m_v2_bal_area"
    km chnget "m_v2_bal_mode"
    kc chnget "m_v2_bal_cal"
    klp chnget "m_v2_bal_lp"
    kb2 ApplyMod "v2_bal", kb2b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v2_bw_src"
    kmin chnget "m_v2_bw_min"
    kmax chnget "m_v2_bw_max"
    kexp chnget "m_v2_bw_exp"
    ko chnget "m_v2_bw_offs"
    kg chnget "m_v2_bw_gain"
    ka chnget "m_v2_bw_area"
    km chnget "m_v2_bw_mode"
    kc chnget "m_v2_bw_cal"
    klp chnget "m_v2_bw_lp"
    kw2 ApplyMod "v2_bw", kw2b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v2_amp_src"
    kmin chnget "m_v2_amp_min"
    kmax chnget "m_v2_amp_max"
    kexp chnget "m_v2_amp_exp"
    ko chnget "m_v2_amp_offs"
    kg chnget "m_v2_amp_gain"
    ka chnget "m_v2_amp_area"
    km chnget "m_v2_amp_mode"
    kc chnget "m_v2_amp_cal"
    klp chnget "m_v2_amp_lp"
    ka2 ApplyMod "v2_amp", ka2b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v3_pitch_src"
    kmin chnget "m_v3_pitch_min"
    kmax chnget "m_v3_pitch_max"
    kexp chnget "m_v3_pitch_exp"
    ko chnget "m_v3_pitch_offs"
    kg chnget "m_v3_pitch_gain"
    ka chnget "m_v3_pitch_area"
    km chnget "m_v3_pitch_mode"
    kc chnget "m_v3_pitch_cal"
    klp chnget "m_v3_pitch_lp"
    kf3 ApplyMod "v3_pitch", kf3b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v3_bal_src"
    kmin chnget "m_v3_bal_min"
    kmax chnget "m_v3_bal_max"
    kexp chnget "m_v3_bal_exp"
    ko chnget "m_v3_bal_offs"
    kg chnget "m_v3_bal_gain"
    ka chnget "m_v3_bal_area"
    km chnget "m_v3_bal_mode"
    kc chnget "m_v3_bal_cal"
    klp chnget "m_v3_bal_lp"
    kb3 ApplyMod "v3_bal", kb3b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v3_bw_src"
    kmin chnget "m_v3_bw_min"
    kmax chnget "m_v3_bw_max"
    kexp chnget "m_v3_bw_exp"
    ko chnget "m_v3_bw_offs"
    kg chnget "m_v3_bw_gain"
    ka chnget "m_v3_bw_area"
    km chnget "m_v3_bw_mode"
    kc chnget "m_v3_bw_cal"
    klp chnget "m_v3_bw_lp"
    kw3 ApplyMod "v3_bw", kw3b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_v3_amp_src"
    kmin chnget "m_v3_amp_min"
    kmax chnget "m_v3_amp_max"
    kexp chnget "m_v3_amp_exp"
    ko chnget "m_v3_amp_offs"
    kg chnget "m_v3_amp_gain"
    ka chnget "m_v3_amp_area"
    km chnget "m_v3_amp_mode"
    kc chnget "m_v3_amp_cal"
    klp chnget "m_v3_amp_lp"
    ka3 ApplyMod "v3_amp", ka3b, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_master_pan_src"
    kmin chnget "m_master_pan_min"
    kmax chnget "m_master_pan_max"
    kexp chnget "m_master_pan_exp"
    ko chnget "m_master_pan_offs"
    kg chnget "m_master_pan_gain"
    ka chnget "m_master_pan_area"
    km chnget "m_master_pan_mode"
    kc chnget "m_master_pan_cal"
    klp chnget "m_master_pan_lp"
    kpan ApplyMod "master_pan", kpanb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ; Print the most recently changed mod source channel for quick debugging.
    kAnyModChange init 0
    SLastModSource init "none"
    kAnyModChange = 0

    ksSel chnget "m_v1_pitch_src"
    kaSel chnget "m_v1_pitch_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v1_bal_src"
    kaSel chnget "m_v1_bal_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v1_bw_src"
    kaSel chnget "m_v1_bw_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v1_amp_src"
    kaSel chnget "m_v1_amp_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v2_pitch_src"
    kaSel chnget "m_v2_pitch_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v2_bal_src"
    kaSel chnget "m_v2_bal_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v2_bw_src"
    kaSel chnget "m_v2_bw_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v2_amp_src"
    kaSel chnget "m_v2_amp_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v3_pitch_src"
    kaSel chnget "m_v3_pitch_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v3_bal_src"
    kaSel chnget "m_v3_bal_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v3_bw_src"
    kaSel chnget "m_v3_bw_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_v3_amp_src"
    kaSel chnget "m_v3_amp_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    ksSel chnget "m_master_pan_src"
    kaSel chnget "m_master_pan_area"
    ktrig = changed(ksSel)
    ktrig2 = changed(kaSel)
    if (ktrig == 1 || ktrig2 == 1) then
        SLastModSource GetModSourceName ksSel, kaSel
        kAnyModChange = 1
    endif

    puts SLastModSource, kAnyModChange+1

    ; Update display rings through Cabbage GUI queue with k-rate triggers.
    ktrig_v1_pitch = changed(kf1)
    cabbageSetValue "v1_pitch_view", kf1, ktrig_v1_pitch
    ktrig_v1_bal = changed(kb1)
    cabbageSetValue "v1_bal_view", kb1, ktrig_v1_bal
    ktrig_v1_bw = changed(kw1)
    cabbageSetValue "v1_bw_view", kw1, ktrig_v1_bw
    ktrig_v1_amp = changed(ka1)
    cabbageSetValue "v1_amp_view", ka1, ktrig_v1_amp

    ktrig_v2_pitch = changed(kf2)
    cabbageSetValue "v2_pitch_view", kf2, ktrig_v2_pitch
    ktrig_v2_bal = changed(kb2)
    cabbageSetValue "v2_bal_view", kb2, ktrig_v2_bal
    ktrig_v2_bw = changed(kw2)
    cabbageSetValue "v2_bw_view", kw2, ktrig_v2_bw
    ktrig_v2_amp = changed(ka2)
    cabbageSetValue "v2_amp_view", ka2, ktrig_v2_amp

    ktrig_v3_pitch = changed(kf3)
    cabbageSetValue "v3_pitch_view", kf3, ktrig_v3_pitch
    ktrig_v3_bal = changed(kb3)
    cabbageSetValue "v3_bal_view", kb3, ktrig_v3_bal
    ktrig_v3_bw = changed(kw3)
    cabbageSetValue "v3_bw_view", kw3, ktrig_v3_bw
    ktrig_v3_amp = changed(ka3)
    cabbageSetValue "v3_amp_view", ka3, ktrig_v3_amp

    ktrig_pan = changed(kpan)
    cabbageSetValue "master_pan_view", kpan, ktrig_pan

    ; Voice helper calculations
    krelbw1 = max(1e-4, pow(2, kw1 * 0.5) - pow(2, -kw1 * 0.5))
    kbwhz1 = max(8, kf1 * krelbw1)
    kncomp1 = 0.50 / sqrt(krelbw1)

    krelbw2 = max(1e-4, pow(2, kw2 * 0.5) - pow(2, -kw2 * 0.5))
    kbwhz2 = max(8, kf2 * krelbw2)
    kncomp2 = 0.50 / sqrt(krelbw2)

    krelbw3 = max(1e-4, pow(2, kw3 * 0.5) - pow(2, -kw3 * 0.5))
    kbwhz3 = max(8, kf3 * krelbw3)
    kncomp3 = 0.50 / sqrt(krelbw3)

    at1 oscili 1, kf1
    an1 rand 1
    abn1 butbp an1, kf1, kbwhz1
    av1 = ((1 - kb1) * at1 + kb1 * (abn1 * 3 * kncomp1)) * ka1

    at2 oscili 1, kf2
    an2 rand 1
    abn2 butbp an2, kf2, kbwhz2
    av2 = ((1 - kb2) * at2 + kb2 * (abn2 * 3 * kncomp2)) * ka2

    at3 oscili 1, kf3
    an3 rand 1
    abn3 butbp an3, kf3, kbwhz3
    av3 = ((1 - kb3) * at3 + kb3 * (abn3 * 3 * kncomp3)) * ka3

    amix = av1 + av2 + av3
    aL, aR pan2 amix, kpan

    ; Mild global trim
    outs aL * 0.25, aR * 0.25
endin

</CsInstruments>
<CsScore>
i1 0 z
i10 0 z
</CsScore>
</CsoundSynthesizer>













