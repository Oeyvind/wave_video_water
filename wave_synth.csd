<Cabbage>
form caption("Wave Synth") size(1260, 1285), guiMode("queue"), pluginId("wSyn"), colour(20, 20, 20)

csoundoutput bounds(5, 5, 1250, 60), channel("csoundoutput")

; --- Instrument 31 controls ---

groupbox bounds(5, 70, 1250, 290), channel("ngGroup"), text("Instr32: NoiseGrains + Mod Routing"), colour(40, 48, 58), fontColour(220, 220, 220), outlineColour(80, 100, 92) {
button bounds(3, 50, 36, 24), channel("inst32_on"), text("On"), value(0), colour:0("#3c4652"), colour:1("#2ecc71"), fontColour("white")
button bounds(3, 78, 36, 18), channel("ng_collapse"), text("+","-"), value(0), colour:0(60,30,30), colour:1(30,60,30), fontColour("white")
button bounds(32, 142, 6, 34), channel("ng_master_reset"), value(0), colour:0(100,20,20), colour:1(200,40,40)
button bounds(41, 142, 6, 34), channel("ng_master_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
; Module ng_amp (Amp, col 1 - has labels)
rslider channel("ng_amp_view"), bounds(48, 25, 70, 70), text(""), range(0, 1, 0.5, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_amp_base"), bounds(48, 29, 70, 70), text("Amp"), range(0, 1, 0.5, 1, 0.001), trackerColour(40,80,200)
combobox bounds(48, 102, 85, 18), channel("m_ng_amp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(48, 122, 85, 18), channel("m_ng_amp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(48, 142, 50, 16), channel("m_ng_amp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(100, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 160, 50, 16), channel("m_ng_amp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(100, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(99, 142, 6, 34), channel("m_ng_amp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(48, 178, 50, 16), channel("m_ng_amp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(100, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 196, 50, 16), channel("m_ng_amp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(48, 214, 50, 16), channel("m_ng_amp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(100, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(100, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 232, 50, 16), channel("m_ng_amp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(100, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(48, 250, 50, 16), channel("m_ng_amp_mode"), value(1), text("Add","Mul")
label bounds(100, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_wfreq (WavFreq, col 2 - no labels)
rslider channel("ng_wfreq_view"), bounds(140, 25, 70, 70), text(""), range(1, 5000, 440, 0.3, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_wfreq_base"), bounds(140, 29, 70, 70), text("WavFreq"), range(1, 5000, 440, 0.3, 1), trackerColour(40,80,200)
combobox bounds(140, 102, 85, 18), channel("m_ng_wfreq_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(140, 122, 85, 18), channel("m_ng_wfreq_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(140, 142, 50, 16), channel("m_ng_wfreq_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(140, 160, 50, 16), channel("m_ng_wfreq_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(191, 142, 6, 34), channel("m_ng_wfreq_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(140, 178, 50, 16), channel("m_ng_wfreq_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(140, 196, 50, 16), channel("m_ng_wfreq_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(140, 214, 50, 16), channel("m_ng_wfreq_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(140, 232, 50, 16), channel("m_ng_wfreq_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(140, 250, 50, 16), channel("m_ng_wfreq_mode"), value(1), text("Add","Mul")
; Module ng_wfreq_rdev (WavFreq RDev, col 3 - has labels)
rslider channel("ng_wfreq_rdev_view"), bounds(232, 25, 70, 70), text(""), range(0, 1, 0, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_wfreq_rdev_base"), bounds(232, 29, 70, 70), text("WF RDev"), range(0, 20, 0, 1, 0.001), trackerColour(40,80,200)
combobox bounds(232, 102, 85, 18), channel("m_ng_wfreq_rdev_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(232, 122, 85, 18), channel("m_ng_wfreq_rdev_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(232, 142, 50, 16), channel("m_ng_wfreq_rdev_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(284, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 160, 50, 16), channel("m_ng_wfreq_rdev_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(284, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(283, 142, 6, 34), channel("m_ng_wfreq_rdev_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(232, 178, 50, 16), channel("m_ng_wfreq_rdev_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(284, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 196, 50, 16), channel("m_ng_wfreq_rdev_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(232, 214, 50, 16), channel("m_ng_wfreq_rdev_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(284, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(284, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 232, 50, 16), channel("m_ng_wfreq_rdev_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(284, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(232, 250, 50, 16), channel("m_ng_wfreq_rdev_mode"), value(1), text("Add","Mul")
label bounds(284, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_grate (GrainRate, col 4 - no labels)
rslider channel("ng_grate_view"), bounds(324, 25, 70, 70), text(""), range(0.2, 200, 10, 0.3, 0.1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_grate_base"), bounds(324, 29, 70, 70), text("GrainRate"), range(0.2, 200, 10, 0.3, 0.1), trackerColour(40,80,200)
combobox bounds(324, 102, 85, 18), channel("m_ng_grate_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(324, 122, 85, 18), channel("m_ng_grate_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(324, 142, 50, 16), channel("m_ng_grate_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(324, 160, 50, 16), channel("m_ng_grate_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(375, 142, 6, 34), channel("m_ng_grate_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(324, 178, 50, 16), channel("m_ng_grate_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(324, 196, 50, 16), channel("m_ng_grate_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(324, 214, 50, 16), channel("m_ng_grate_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(324, 232, 50, 16), channel("m_ng_grate_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(324, 250, 50, 16), channel("m_ng_grate_mode"), value(1), text("Add","Mul")
; Module ng_grate_rdev (GrainRate RDev, col 5 - has labels)
rslider channel("ng_grate_rdev_view"), bounds(416, 25, 70, 70), text(""), range(0, 1, 0, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_grate_rdev_base"), bounds(416, 29, 70, 70), text("GR RDev"), range(0, 20, 0, 1, 0.001), trackerColour(40,80,200)
combobox bounds(416, 102, 85, 18), channel("m_ng_grate_rdev_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(416, 122, 85, 18), channel("m_ng_grate_rdev_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(416, 142, 50, 16), channel("m_ng_grate_rdev_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(468, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 160, 50, 16), channel("m_ng_grate_rdev_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(468, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(467, 142, 6, 34), channel("m_ng_grate_rdev_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(416, 178, 50, 16), channel("m_ng_grate_rdev_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(468, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 196, 50, 16), channel("m_ng_grate_rdev_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(416, 214, 50, 16), channel("m_ng_grate_rdev_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(468, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(468, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 232, 50, 16), channel("m_ng_grate_rdev_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(468, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(416, 250, 50, 16), channel("m_ng_grate_rdev_mode"), value(1), text("Add","Mul")
label bounds(468, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_gdur (GrainDur, col 6 - no labels)
rslider channel("ng_gdur_view"), bounds(508, 25, 70, 70), text(""), range(0, 1, 0.2, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_gdur_base"), bounds(508, 29, 70, 70), text("GrainDur"), range(0, 1, 0.2, 1, 0.001), trackerColour(40,80,200)
combobox bounds(508, 102, 85, 18), channel("m_ng_gdur_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(508, 122, 85, 18), channel("m_ng_gdur_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(508, 142, 50, 16), channel("m_ng_gdur_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(508, 160, 50, 16), channel("m_ng_gdur_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(559, 142, 6, 34), channel("m_ng_gdur_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(508, 178, 50, 16), channel("m_ng_gdur_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(508, 196, 50, 16), channel("m_ng_gdur_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(508, 214, 50, 16), channel("m_ng_gdur_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(508, 232, 50, 16), channel("m_ng_gdur_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(508, 250, 50, 16), channel("m_ng_gdur_mode"), value(1), text("Add","Mul")
; Module ng_adr (A/D Ratio, col 7 - has labels)
rslider channel("ng_adr_view"), bounds(600, 25, 70, 70), text(""), range(0.005, 1, 0.5, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_adr_base"), bounds(600, 29, 70, 70), text("A/D"), range(0.005, 1, 0.5, 1, 0.001), trackerColour(40,80,200)
combobox bounds(600, 102, 85, 18), channel("m_ng_adr_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(600, 122, 85, 18), channel("m_ng_adr_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(600, 142, 50, 16), channel("m_ng_adr_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(652, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 160, 50, 16), channel("m_ng_adr_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(652, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(651, 142, 6, 34), channel("m_ng_adr_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(600, 178, 50, 16), channel("m_ng_adr_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(652, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 196, 50, 16), channel("m_ng_adr_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(600, 214, 50, 16), channel("m_ng_adr_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(652, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(652, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 232, 50, 16), channel("m_ng_adr_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(652, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(600, 250, 50, 16), channel("m_ng_adr_mode"), value(1), text("Add","Mul")
label bounds(652, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_sus (Sustain, col 8 - no labels)
rslider channel("ng_sus_view"), bounds(692, 25, 70, 70), text(""), range(0, 0.99, 0.5, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_sus_base"), bounds(692, 29, 70, 70), text("Sustain"), range(0, 0.99, 0.5, 1, 0.001), trackerColour(40,80,200)
combobox bounds(692, 102, 85, 18), channel("m_ng_sus_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(692, 122, 85, 18), channel("m_ng_sus_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(692, 142, 50, 16), channel("m_ng_sus_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(692, 160, 50, 16), channel("m_ng_sus_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(743, 142, 6, 34), channel("m_ng_sus_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(692, 178, 50, 16), channel("m_ng_sus_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(692, 196, 50, 16), channel("m_ng_sus_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(692, 214, 50, 16), channel("m_ng_sus_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(692, 232, 50, 16), channel("m_ng_sus_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(692, 250, 50, 16), channel("m_ng_sus_mode"), value(1), text("Add","Mul")
; Module ng_out_lp (Output LPF, col 9 - has labels)
rslider channel("ng_out_lp_view"), bounds(784, 25, 70, 70), text(""), range(20, 20000, 20000, 0.3, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng_out_lp_base"), bounds(784, 29, 70, 70), text("LPF"), range(20, 20000, 20000, 0.3, 1), trackerColour(40,80,200)
combobox bounds(784, 102, 85, 18), channel("m_ng_out_lp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(784, 122, 85, 18), channel("m_ng_out_lp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(784, 142, 50, 16), channel("m_ng_out_lp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(836, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 160, 50, 16), channel("m_ng_out_lp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(836, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(835, 142, 6, 34), channel("m_ng_out_lp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(784, 178, 50, 16), channel("m_ng_out_lp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(836, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 196, 50, 16), channel("m_ng_out_lp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(784, 214, 50, 16), channel("m_ng_out_lp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(836, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(836, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 232, 50, 16), channel("m_ng_out_lp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(836, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(784, 250, 50, 16), channel("m_ng_out_lp_mode"), value(1), text("Add","Mul")
label bounds(836, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
label bounds(878, 16, 48, 12), text("Source"), fontColour(220,220,220), fontSize(11)
combobox bounds(878, 30, 85, 22), channel("ng_source_sel"), value(1), text("Sine","Noise","Wave")
rslider channel("ng_samplepos"), bounds(966, 25, 58, 62), text("S.pos"), range(0, 1, 0, 1, 0.001), trackerColour(40,80,200)

nslider channel("ng_wfreq_rdev_minfreq"), bounds(222, 5, 40, 30), text("WFmn"), range(0.05, 100, 0.50, 1, 0.01), fontSize(13)
nslider channel("ng_wfreq_rdev_maxfreq"), bounds(272, 5, 40, 30), text("WFmx"), range(0.05, 100, 4.00, 1, 0.01), fontSize(13)
nslider channel("ng_grate_rdev_minfreq"), bounds(406, 5, 40, 30), text("GRmn"), range(0.05, 100, 0.50, 1, 0.01), fontSize(13)
nslider channel("ng_grate_rdev_maxfreq"), bounds(456, 5, 40, 30), text("GRmx"), range(0.05, 100, 4.00, 1, 0.01), fontSize(13)
nslider channel("GainMask"), bounds(1034, 5, 60, 30), text("GainMask"), range(1, 4, 1, 1, 0.01), fontSize(13)
nslider channel("ChanMask"), bounds(1124, 5, 60, 30), text("ChanMask"), range(1, 4, 1, 1, 0.01), fontSize(13)
}

groupbox bounds(5, 365, 1250, 290), channel("ng2Group"), text("Instr35: NoiseGrains 2 + Mod Routing"), colour(40, 48, 58), fontColour(220, 220, 220), outlineColour(80, 100, 92) {
button bounds(3, 50, 36, 24), channel("inst35_on"), text("On"), value(0), colour:0("#3c4652"), colour:1("#2ecc71"), fontColour("white")
button bounds(3, 78, 36, 18), channel("ng2_collapse"), text("+","-"), value(0), colour:0(60,30,30), colour:1(30,60,30), fontColour("white")
button bounds(32, 142, 6, 34), channel("ng2_master_reset"), value(0), colour:0(100,20,20), colour:1(200,40,40)
button bounds(41, 142, 6, 34), channel("ng2_master_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
; Module ng_amp (Amp, col 1 - has labels)
rslider channel("ng2_amp_view"), bounds(48, 25, 70, 70), text(""), range(0, 1, 0.5, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_amp_base"), bounds(48, 29, 70, 70), text("Amp"), range(0, 1, 0.5, 1, 0.001), trackerColour(40,80,200)
combobox bounds(48, 102, 85, 18), channel("m_ng2_amp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(48, 122, 85, 18), channel("m_ng2_amp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(48, 142, 50, 16), channel("m_ng2_amp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(100, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 160, 50, 16), channel("m_ng2_amp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(100, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(99, 142, 6, 34), channel("m_ng2_amp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(48, 178, 50, 16), channel("m_ng2_amp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(100, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 196, 50, 16), channel("m_ng2_amp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(48, 214, 50, 16), channel("m_ng2_amp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(100, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(100, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 232, 50, 16), channel("m_ng2_amp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(100, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(48, 250, 50, 16), channel("m_ng2_amp_mode"), value(1), text("Add","Mul")
label bounds(100, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_wfreq (WavFreq, col 2 - no labels)
rslider channel("ng2_wfreq_view"), bounds(140, 25, 70, 70), text(""), range(1, 5000, 440, 0.3, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_wfreq_base"), bounds(140, 29, 70, 70), text("WavFreq"), range(1, 5000, 440, 0.3, 1), trackerColour(40,80,200)
combobox bounds(140, 102, 85, 18), channel("m_ng2_wfreq_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(140, 122, 85, 18), channel("m_ng2_wfreq_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(140, 142, 50, 16), channel("m_ng2_wfreq_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(140, 160, 50, 16), channel("m_ng2_wfreq_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(191, 142, 6, 34), channel("m_ng2_wfreq_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(140, 178, 50, 16), channel("m_ng2_wfreq_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(140, 196, 50, 16), channel("m_ng2_wfreq_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(140, 214, 50, 16), channel("m_ng2_wfreq_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(140, 232, 50, 16), channel("m_ng2_wfreq_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(140, 250, 50, 16), channel("m_ng2_wfreq_mode"), value(1), text("Add","Mul")
; Module ng_wfreq_rdev (WavFreq RDev, col 3 - has labels)
rslider channel("ng2_wfreq_rdev_view"), bounds(232, 25, 70, 70), text(""), range(0, 1, 0, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_wfreq_rdev_base"), bounds(232, 29, 70, 70), text("WF RDev"), range(0, 20, 0, 1, 0.001), trackerColour(40,80,200)
combobox bounds(232, 102, 85, 18), channel("m_ng2_wfreq_rdev_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(232, 122, 85, 18), channel("m_ng2_wfreq_rdev_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(232, 142, 50, 16), channel("m_ng2_wfreq_rdev_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(284, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 160, 50, 16), channel("m_ng2_wfreq_rdev_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(284, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(283, 142, 6, 34), channel("m_ng2_wfreq_rdev_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(232, 178, 50, 16), channel("m_ng2_wfreq_rdev_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(284, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 196, 50, 16), channel("m_ng2_wfreq_rdev_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(232, 214, 50, 16), channel("m_ng2_wfreq_rdev_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(284, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(284, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 232, 50, 16), channel("m_ng2_wfreq_rdev_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(284, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(232, 250, 50, 16), channel("m_ng2_wfreq_rdev_mode"), value(1), text("Add","Mul")
label bounds(284, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_grate (GrainRate, col 4 - no labels)
rslider channel("ng2_grate_view"), bounds(324, 25, 70, 70), text(""), range(0.2, 200, 10, 0.3, 0.1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_grate_base"), bounds(324, 29, 70, 70), text("GrainRate"), range(0.2, 200, 10, 0.3, 0.1), trackerColour(40,80,200)
combobox bounds(324, 102, 85, 18), channel("m_ng2_grate_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(324, 122, 85, 18), channel("m_ng2_grate_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(324, 142, 50, 16), channel("m_ng2_grate_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(324, 160, 50, 16), channel("m_ng2_grate_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(375, 142, 6, 34), channel("m_ng2_grate_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(324, 178, 50, 16), channel("m_ng2_grate_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(324, 196, 50, 16), channel("m_ng2_grate_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(324, 214, 50, 16), channel("m_ng2_grate_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(324, 232, 50, 16), channel("m_ng2_grate_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(324, 250, 50, 16), channel("m_ng2_grate_mode"), value(1), text("Add","Mul")
; Module ng_grate_rdev (GrainRate RDev, col 5 - has labels)
rslider channel("ng2_grate_rdev_view"), bounds(416, 25, 70, 70), text(""), range(0, 1, 0, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_grate_rdev_base"), bounds(416, 29, 70, 70), text("GR RDev"), range(0, 20, 0, 1, 0.001), trackerColour(40,80,200)
combobox bounds(416, 102, 85, 18), channel("m_ng2_grate_rdev_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(416, 122, 85, 18), channel("m_ng2_grate_rdev_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(416, 142, 50, 16), channel("m_ng2_grate_rdev_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(468, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 160, 50, 16), channel("m_ng2_grate_rdev_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(468, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(467, 142, 6, 34), channel("m_ng2_grate_rdev_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(416, 178, 50, 16), channel("m_ng2_grate_rdev_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(468, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 196, 50, 16), channel("m_ng2_grate_rdev_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(416, 214, 50, 16), channel("m_ng2_grate_rdev_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(468, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(468, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 232, 50, 16), channel("m_ng2_grate_rdev_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(468, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(416, 250, 50, 16), channel("m_ng2_grate_rdev_mode"), value(1), text("Add","Mul")
label bounds(468, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_gdur (GrainDur, col 6 - no labels)
rslider channel("ng2_gdur_view"), bounds(508, 25, 70, 70), text(""), range(0, 1, 0.2, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_gdur_base"), bounds(508, 29, 70, 70), text("GrainDur"), range(0, 1, 0.2, 1, 0.001), trackerColour(40,80,200)
combobox bounds(508, 102, 85, 18), channel("m_ng2_gdur_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(508, 122, 85, 18), channel("m_ng2_gdur_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(508, 142, 50, 16), channel("m_ng2_gdur_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(508, 160, 50, 16), channel("m_ng2_gdur_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(559, 142, 6, 34), channel("m_ng2_gdur_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(508, 178, 50, 16), channel("m_ng2_gdur_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(508, 196, 50, 16), channel("m_ng2_gdur_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(508, 214, 50, 16), channel("m_ng2_gdur_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(508, 232, 50, 16), channel("m_ng2_gdur_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(508, 250, 50, 16), channel("m_ng2_gdur_mode"), value(1), text("Add","Mul")
; Module ng_adr (A/D Ratio, col 7 - has labels)
rslider channel("ng2_adr_view"), bounds(600, 25, 70, 70), text(""), range(0.005, 1, 0.5, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_adr_base"), bounds(600, 29, 70, 70), text("A/D"), range(0.005, 1, 0.5, 1, 0.001), trackerColour(40,80,200)
combobox bounds(600, 102, 85, 18), channel("m_ng2_adr_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(600, 122, 85, 18), channel("m_ng2_adr_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(600, 142, 50, 16), channel("m_ng2_adr_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(652, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 160, 50, 16), channel("m_ng2_adr_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(652, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(651, 142, 6, 34), channel("m_ng2_adr_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(600, 178, 50, 16), channel("m_ng2_adr_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(652, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 196, 50, 16), channel("m_ng2_adr_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(600, 214, 50, 16), channel("m_ng2_adr_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(652, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(652, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 232, 50, 16), channel("m_ng2_adr_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(652, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(600, 250, 50, 16), channel("m_ng2_adr_mode"), value(1), text("Add","Mul")
label bounds(652, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
; Module ng_sus (Sustain, col 8 - no labels)
rslider channel("ng2_sus_view"), bounds(692, 25, 70, 70), text(""), range(0, 0.99, 0.5, 1, 0.001), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_sus_base"), bounds(692, 29, 70, 70), text("Sustain"), range(0, 0.99, 0.5, 1, 0.001), trackerColour(40,80,200)
combobox bounds(692, 102, 85, 18), channel("m_ng2_sus_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(692, 122, 85, 18), channel("m_ng2_sus_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(692, 142, 50, 16), channel("m_ng2_sus_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(692, 160, 50, 16), channel("m_ng2_sus_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(743, 142, 6, 34), channel("m_ng2_sus_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(692, 178, 50, 16), channel("m_ng2_sus_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(692, 196, 50, 16), channel("m_ng2_sus_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(692, 214, 50, 16), channel("m_ng2_sus_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(692, 232, 50, 16), channel("m_ng2_sus_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(692, 250, 50, 16), channel("m_ng2_sus_mode"), value(1), text("Add","Mul")
; Module ng2_out_lp (Output LPF, col 9 - has labels)
rslider channel("ng2_out_lp_view"), bounds(784, 25, 70, 70), text(""), range(20, 20000, 20000, 0.3, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ng2_out_lp_base"), bounds(784, 29, 70, 70), text("LPF"), range(20, 20000, 20000, 0.3, 1), trackerColour(40,80,200)
combobox bounds(784, 102, 85, 18), channel("m_ng2_out_lp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(784, 122, 85, 18), channel("m_ng2_out_lp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(784, 142, 50, 16), channel("m_ng2_out_lp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(836, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 160, 50, 16), channel("m_ng2_out_lp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(836, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(835, 142, 6, 34), channel("m_ng2_out_lp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(784, 178, 50, 16), channel("m_ng2_out_lp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(836, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 196, 50, 16), channel("m_ng2_out_lp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(784, 214, 50, 16), channel("m_ng2_out_lp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(836, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(836, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 232, 50, 16), channel("m_ng2_out_lp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(836, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(784, 250, 50, 16), channel("m_ng2_out_lp_mode"), value(1), text("Add","Mul")
label bounds(836, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
label bounds(878, 16, 48, 12), text("Source"), fontColour(220,220,220), fontSize(11)
combobox bounds(878, 30, 85, 22), channel("ng2_source_sel"), value(1), text("Sine","Noise","Wave")
rslider channel("ng2_samplepos"), bounds(966, 25, 58, 62), text("S.pos"), range(0, 1, 0, 1, 0.001), trackerColour(40,80,200)

nslider channel("ng2_wfreq_rdev_minfreq"), bounds(222, 5, 40, 30), text("WFmn"), range(0.05, 100, 0.50, 1, 0.01), fontSize(13)
nslider channel("ng2_wfreq_rdev_maxfreq"), bounds(272, 5, 40, 30), text("WFmx"), range(0.05, 100, 4.00, 1, 0.01), fontSize(13)
nslider channel("ng2_grate_rdev_minfreq"), bounds(406, 5, 40, 30), text("GRmn"), range(0.05, 100, 0.50, 1, 0.01), fontSize(13)
nslider channel("ng2_grate_rdev_maxfreq"), bounds(456, 5, 40, 30), text("GRmx"), range(0.05, 100, 4.00, 1, 0.01), fontSize(13)
nslider channel("GainMask2"), bounds(1034, 5, 60, 30), text("GainMask"), range(1, 4, 1, 1, 0.01), fontSize(13)
nslider channel("ChanMask2"), bounds(1124, 5, 60, 30), text("ChanMask"), range(1, 4, 1, 1, 0.01), fontSize(13)
}

groupbox bounds(5, 660, 1250, 290), channel("ccMapGroup"), text("Instr33: Video Mod -> MIDI CC Mapper"), colour(48, 44, 58), fontColour(220, 220, 220), outlineColour(88, 88, 102) {
button bounds(3, 50, 36, 24), channel("ccmap_on"), text("On"), value(1), colour:0("#3c4652"), colour:1("#2ecc71"), fontColour("white")
button bounds(3, 78, 36, 18), channel("ccmap_collapse"), text("+","-"), value(0), colour:0(60,30,30), colour:1(30,60,30), fontColour("white")
button bounds(32, 142, 6, 34), channel("ccm_master_reset"), value(0), colour:0(100,20,20), colour:1(200,40,40)
button bounds(41, 142, 6, 34), channel("ccm_master_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")

; CC Map 1 (CC32)
rslider channel("ccm1_view"), bounds(48, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm1_base"), bounds(48, 29, 70, 70), text("CC32"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(48, 102, 85, 18), channel("m_ccm1_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(48, 122, 85, 18), channel("m_ccm1_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(48, 142, 50, 16), channel("m_ccm1_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(100, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 160, 50, 16), channel("m_ccm1_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(100, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(99, 142, 6, 34), channel("m_ccm1_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(48, 178, 50, 16), channel("m_ccm1_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(100, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 196, 50, 16), channel("m_ccm1_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(48, 214, 50, 16), channel("m_ccm1_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
label bounds(100, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(100, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 232, 50, 16), channel("m_ccm1_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(100, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(48, 250, 50, 16), channel("m_ccm1_mode"), value(1), text("Add","Mul")
label bounds(100, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; CC Map 2..8 (CC33..CC39)
rslider channel("ccm2_view"), bounds(140, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm2_base"), bounds(140, 29, 70, 70), text("CC33"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(140, 102, 85, 18), channel("m_ccm2_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(140, 122, 85, 18), channel("m_ccm2_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(140, 142, 50, 16), channel("m_ccm2_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(140, 160, 50, 16), channel("m_ccm2_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(191, 142, 6, 34), channel("m_ccm2_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(140, 178, 50, 16), channel("m_ccm2_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(140, 196, 50, 16), channel("m_ccm2_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(140, 214, 50, 16), channel("m_ccm2_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
nslider bounds(140, 232, 50, 16), channel("m_ccm2_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(140, 250, 50, 16), channel("m_ccm2_mode"), value(1), text("Add","Mul")

rslider channel("ccm3_view"), bounds(232, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm3_base"), bounds(232, 29, 70, 70), text("CC34"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(232, 102, 85, 18), channel("m_ccm3_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(232, 122, 85, 18), channel("m_ccm3_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(232, 142, 50, 16), channel("m_ccm3_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(284, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 160, 50, 16), channel("m_ccm3_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(284, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(283, 142, 6, 34), channel("m_ccm3_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(232, 178, 50, 16), channel("m_ccm3_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(284, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 196, 50, 16), channel("m_ccm3_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(232, 214, 50, 16), channel("m_ccm3_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
label bounds(284, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(284, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 232, 50, 16), channel("m_ccm3_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(284, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(232, 250, 50, 16), channel("m_ccm3_mode"), value(1), text("Add","Mul")
label bounds(284, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

rslider channel("ccm4_view"), bounds(324, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm4_base"), bounds(324, 29, 70, 70), text("CC35"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(324, 102, 85, 18), channel("m_ccm4_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(324, 122, 85, 18), channel("m_ccm4_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(324, 142, 50, 16), channel("m_ccm4_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(324, 160, 50, 16), channel("m_ccm4_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(375, 142, 6, 34), channel("m_ccm4_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(324, 178, 50, 16), channel("m_ccm4_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(324, 196, 50, 16), channel("m_ccm4_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(324, 214, 50, 16), channel("m_ccm4_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
nslider bounds(324, 232, 50, 16), channel("m_ccm4_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(324, 250, 50, 16), channel("m_ccm4_mode"), value(1), text("Add","Mul")

rslider channel("ccm5_view"), bounds(416, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm5_base"), bounds(416, 29, 70, 70), text("CC36"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(416, 102, 85, 18), channel("m_ccm5_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(416, 122, 85, 18), channel("m_ccm5_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(416, 142, 50, 16), channel("m_ccm5_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(468, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 160, 50, 16), channel("m_ccm5_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(468, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(467, 142, 6, 34), channel("m_ccm5_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(416, 178, 50, 16), channel("m_ccm5_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(468, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 196, 50, 16), channel("m_ccm5_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(416, 214, 50, 16), channel("m_ccm5_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
label bounds(468, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(468, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 232, 50, 16), channel("m_ccm5_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(468, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(416, 250, 50, 16), channel("m_ccm5_mode"), value(1), text("Add","Mul")
label bounds(468, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

rslider channel("ccm6_view"), bounds(508, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm6_base"), bounds(508, 29, 70, 70), text("CC37"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(508, 102, 85, 18), channel("m_ccm6_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(508, 122, 85, 18), channel("m_ccm6_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(508, 142, 50, 16), channel("m_ccm6_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(508, 160, 50, 16), channel("m_ccm6_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(559, 142, 6, 34), channel("m_ccm6_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(508, 178, 50, 16), channel("m_ccm6_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(508, 196, 50, 16), channel("m_ccm6_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(508, 214, 50, 16), channel("m_ccm6_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
nslider bounds(508, 232, 50, 16), channel("m_ccm6_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(508, 250, 50, 16), channel("m_ccm6_mode"), value(1), text("Add","Mul")

rslider channel("ccm7_view"), bounds(600, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm7_base"), bounds(600, 29, 70, 70), text("CC38"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(600, 102, 85, 18), channel("m_ccm7_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(600, 122, 85, 18), channel("m_ccm7_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(600, 142, 50, 16), channel("m_ccm7_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(652, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 160, 50, 16), channel("m_ccm7_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(652, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(651, 142, 6, 34), channel("m_ccm7_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(600, 178, 50, 16), channel("m_ccm7_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(652, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 196, 50, 16), channel("m_ccm7_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(600, 214, 50, 16), channel("m_ccm7_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
label bounds(652, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(652, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 232, 50, 16), channel("m_ccm7_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(652, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(600, 250, 50, 16), channel("m_ccm7_mode"), value(1), text("Add","Mul")
label bounds(652, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

rslider channel("ccm8_view"), bounds(692, 25, 70, 70), text(""), range(0, 127, 64, 1, 1), markerThickness(0), outlineColour(0,0,0,0), trackerInsideRadius(0.8), colour("black")
rslider channel("ccm8_base"), bounds(692, 29, 70, 70), text("CC39"), range(0, 127, 64, 1, 1), trackerColour(40,80,200)
combobox bounds(692, 102, 85, 18), channel("m_ccm8_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(692, 122, 85, 18), channel("m_ccm8_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(692, 142, 50, 16), channel("m_ccm8_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(692, 160, 50, 16), channel("m_ccm8_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(743, 142, 6, 34), channel("m_ccm8_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(692, 178, 50, 16), channel("m_ccm8_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(692, 196, 50, 16), channel("m_ccm8_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(692, 214, 50, 16), channel("m_ccm8_gain"), text(""), range(-127,127,0,1,0.001), fontSize(13)
nslider bounds(692, 232, 50, 16), channel("m_ccm8_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(692, 250, 50, 16), channel("m_ccm8_mode"), value(1), text("Add","Mul")
}

groupbox bounds(5, 955, 1250, 110), channel("wave34Group"), text("Instr34: Slit Oscillator / Waveshaper"), colour(52, 50, 42), fontColour(220, 220, 220), outlineColour(100, 96, 82) {
combobox bounds(3, 29, 46, 18), channel("W_mode"), value(1), text("Osc","Shp")
button bounds(3, 50, 36, 24), channel("inst34_on"), text("On"), value(0), colour:0("#3c4652"), colour:1("#2ecc71"), fontColour("white")
button bounds(3, 78, 36, 18), channel("wave34_collapse"), text("+","-"), value(0), colour:0(60,30,30), colour:1(30,60,30), fontColour("white")
button bounds(32, 142, 6, 34), channel("w_master_reset"), value(0), colour:0(100,20,20), colour:1(200,40,40)
button bounds(41, 142, 6, 34), channel("w_master_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")

; W_freq (col 1, x=48, has labels)
rslider channel("W_freq"), bounds(48, 29, 70, 70), text("Freq"), range(20, 4000, 220, 0.3, 0.01), trackerColour(180,120,60)
combobox bounds(48, 102, 85, 18), channel("m_w_freq_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(48, 122, 85, 18), channel("m_w_freq_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(48, 142, 50, 16), channel("m_w_freq_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(100, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 160, 50, 16), channel("m_w_freq_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(100, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(99, 142, 6, 34), channel("m_w_freq_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(48, 178, 50, 16), channel("m_w_freq_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(100, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 196, 50, 16), channel("m_w_freq_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(48, 214, 50, 16), channel("m_w_freq_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(100, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(100, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(48, 232, 50, 16), channel("m_w_freq_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(100, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(48, 250, 50, 16), channel("m_w_freq_mode"), value(1), text("Add","Mul")
label bounds(100, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; W_start (col 2, x=140)
rslider channel("W_start"), bounds(140, 29, 70, 70), text("Start"), range(0, 511, 0, 1, 1), trackerColour(180,120,60)
combobox bounds(140, 102, 85, 18), channel("m_w_start_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(140, 122, 85, 18), channel("m_w_start_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(140, 142, 50, 16), channel("m_w_start_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(140, 160, 50, 16), channel("m_w_start_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(191, 142, 6, 34), channel("m_w_start_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(140, 178, 50, 16), channel("m_w_start_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(140, 196, 50, 16), channel("m_w_start_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(140, 214, 50, 16), channel("m_w_start_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(140, 232, 50, 16), channel("m_w_start_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(140, 250, 50, 16), channel("m_w_start_mode"), value(1), text("Add","Mul")

; W_end (col 3, x=232, has labels)
rslider channel("W_end"), bounds(232, 29, 70, 70), text("End"), range(0, 511, 511, 1, 1), trackerColour(180,120,60)
combobox bounds(232, 102, 85, 18), channel("m_w_end_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(232, 122, 85, 18), channel("m_w_end_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(232, 142, 50, 16), channel("m_w_end_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(284, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 160, 50, 16), channel("m_w_end_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(284, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(283, 142, 6, 34), channel("m_w_end_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(232, 178, 50, 16), channel("m_w_end_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(284, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 196, 50, 16), channel("m_w_end_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(232, 214, 50, 16), channel("m_w_end_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(284, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(284, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(232, 232, 50, 16), channel("m_w_end_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(284, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(232, 250, 50, 16), channel("m_w_end_mode"), value(1), text("Add","Mul")
label bounds(284, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; W_Amp (col 4, x=324)
rslider channel("W_Amp"), bounds(324, 29, 70, 70), text("Amp dB"), range(-96, 6, -12, 1, 0.1), trackerColour(180,120,60)
combobox bounds(324, 102, 85, 18), channel("m_w_amp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(324, 122, 85, 18), channel("m_w_amp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(324, 142, 50, 16), channel("m_w_amp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(324, 160, 50, 16), channel("m_w_amp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(375, 142, 6, 34), channel("m_w_amp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(324, 178, 50, 16), channel("m_w_amp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(324, 196, 50, 16), channel("m_w_amp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(324, 214, 50, 16), channel("m_w_amp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(324, 232, 50, 16), channel("m_w_amp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(324, 250, 50, 16), channel("m_w_amp_mode"), value(1), text("Add","Mul")

; W_tone (col 5, x=416, has labels)
rslider channel("W_tone"), bounds(416, 29, 70, 70), text("Tone"), range(0, 1, 0.5, 1, 0.001), trackerColour(180,120,60)
combobox bounds(416, 102, 85, 18), channel("m_w_tone_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(416, 122, 85, 18), channel("m_w_tone_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(416, 142, 50, 16), channel("m_w_tone_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(468, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 160, 50, 16), channel("m_w_tone_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(468, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(467, 142, 6, 34), channel("m_w_tone_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(416, 178, 50, 16), channel("m_w_tone_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(468, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 196, 50, 16), channel("m_w_tone_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(416, 214, 50, 16), channel("m_w_tone_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(468, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(468, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(416, 232, 50, 16), channel("m_w_tone_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(468, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(416, 250, 50, 16), channel("m_w_tone_mode"), value(1), text("Add","Mul")
label bounds(468, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; W_brite (col 6, x=508)
rslider channel("W_brite"), bounds(508, 29, 70, 70), text("Brite"), range(-3, 3, 0, 1, 0.001), trackerColour(180,120,60)
combobox bounds(508, 102, 85, 18), channel("m_w_brite_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(508, 122, 85, 18), channel("m_w_brite_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(508, 142, 50, 16), channel("m_w_brite_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(508, 160, 50, 16), channel("m_w_brite_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(559, 142, 6, 34), channel("m_w_brite_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(508, 178, 50, 16), channel("m_w_brite_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(508, 196, 50, 16), channel("m_w_brite_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(508, 214, 50, 16), channel("m_w_brite_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(508, 232, 50, 16), channel("m_w_brite_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(508, 250, 50, 16), channel("m_w_brite_mode"), value(1), text("Add","Mul")

; W_gain (col 7, x=600, has labels)
rslider channel("W_gain"), bounds(600, 29, 70, 70), text("Gain"), range(0, 1, 0.6, 1, 0.001), trackerColour(180,120,60)
combobox bounds(600, 102, 85, 18), channel("m_w_gain_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(600, 122, 85, 18), channel("m_w_gain_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(600, 142, 50, 16), channel("m_w_gain_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(652, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 160, 50, 16), channel("m_w_gain_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(652, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(651, 142, 6, 34), channel("m_w_gain_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(600, 178, 50, 16), channel("m_w_gain_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(652, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 196, 50, 16), channel("m_w_gain_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(600, 214, 50, 16), channel("m_w_gain_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(652, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(652, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(600, 232, 50, 16), channel("m_w_gain_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(652, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(600, 250, 50, 16), channel("m_w_gain_mode"), value(1), text("Add","Mul")
label bounds(652, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)

; W_center (col 8, x=692)
rslider channel("W_center"), bounds(692, 29, 70, 70), text("Center"), range(-1, 1, 0, 1, 0.001), trackerColour(180,120,60)
combobox bounds(692, 102, 85, 18), channel("m_w_center_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(692, 122, 85, 18), channel("m_w_center_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(692, 142, 50, 16), channel("m_w_center_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
nslider bounds(692, 160, 50, 16), channel("m_w_center_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
button bounds(743, 142, 6, 34), channel("m_w_center_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(692, 178, 50, 16), channel("m_w_center_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
nslider bounds(692, 196, 50, 16), channel("m_w_center_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(692, 214, 50, 16), channel("m_w_center_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
nslider bounds(692, 232, 50, 16), channel("m_w_center_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
combobox bounds(692, 250, 50, 16), channel("m_w_center_mode"), value(1), text("Add","Mul")

; W_lopass (col 9, x=784, has labels)
rslider channel("W_lopass"), bounds(784, 29, 70, 70), text("LP Hz"), range(50, 18000, 6000, 0.4, 1), trackerColour(180,120,60)
combobox bounds(784, 102, 85, 18), channel("m_w_lp_src"), value(1), text("None","SLo","SMid","SHi","TLo","TMid","THi","SCtr","TCtr","WL","FFMag","FSMag","FFX","FFY","FSX","FSY","LBPSm","LBPOC","FFTCr","Act")
combobox bounds(784, 122, 85, 18), channel("m_w_lp_area"), value(1), text("G","UL","UR","LL","LR","Up","Lo","Lf","Rt")
nslider bounds(784, 142, 50, 16), channel("m_w_lp_min"), text(""), range(0,1,0,1,0.001), fontSize(13)
label bounds(836, 142, 50, 16), text("Min"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 160, 50, 16), channel("m_w_lp_max"), text(""), range(0,1,1,1,0.001), fontSize(13)
label bounds(836, 160, 50, 16), text("Max"), fontColour(220,220,220), fontSize(13)
button bounds(835, 142, 6, 34), channel("m_w_lp_cal"), value(0), colour:0("#3c4652"), colour:1("#2ecc71")
nslider bounds(784, 178, 50, 16), channel("m_w_lp_exp"), text(""), range(0.1,4,1,1,0.001), fontSize(13)
label bounds(836, 178, 50, 16), text("Exp"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 196, 50, 16), channel("m_w_lp_offs"), text(""), range(-1,1,0,1,0.001), fontSize(13)
nslider bounds(784, 214, 50, 16), channel("m_w_lp_gain"), text(""), range(-9999,9999,0,1,0.001), fontSize(13)
label bounds(836, 196, 50, 16), text("Offs"), fontColour(220,220,220), fontSize(13)
label bounds(836, 214, 50, 16), text("Gain"), fontColour(220,220,220), fontSize(13)
nslider bounds(784, 232, 50, 16), channel("m_w_lp_lp"), text(""), range(0.1,20,20,1,0.01), fontSize(13)
label bounds(836, 232, 50, 16), text("LP Hz"), fontColour(220,220,220), fontSize(13)
combobox bounds(784, 250, 50, 16), channel("m_w_lp_mode"), value(1), text("Add","Mul")
label bounds(836, 250, 50, 16), text("Mode"), fontColour(220,220,220), fontSize(13)
}

csoundoutput bounds(5, 1070, 1250, 200), channel("csoundoutput")

</Cabbage>

<CsoundSynthesizer>
<CsOptions>
-n -d -+rtmidi=NULL -Q0 -m0d
</CsOptions>
<CsInstruments>

sr = 48000
ksmps = 32
nchnls = 2
0dbfs = 1

gihOsc OSCinit 8100

  ; classic waveforms
	giSine		ftgen	0, 0, 65537, 10, 1					; sine wave
	giCosine	ftgen	0, 0, 8193, 9, 1, 1, 90					; cosine wave
	giTri		ftgen	0, 0, 8193, 7, 0, 2048, 1, 4096, -1, 2048, 0		; triangle wave 
	giNoise		ftgen	0, 0, 65536, 21, 1
    ; Live central-slit waveform table (512 points) + chunk-state table.
    ; Filled from OSC /slit/chunk packets in instr 10.
    giSlitWave	ftgen	0, 0, 512, 7, 0, 512, 0
    giSlitSeen	ftgen	0, 0, 32, 7, 0, 32, 0

	; grain envelope tables
	giSigmoRise 	ftgen	0, 0, 8193, 19, 0.5, 1, 270, 1				; rising sigmoid
	giSigmoFall 	ftgen	0, 0, 8193, 19, 0.5, 1, 90, 1				; falling sigmoid
	giExpFall	ftgen	0, 0, 8193, 5, 1, 8193, 0.00001				; exponential decay
	giTriangleWin 	ftgen	0, 0, 8193, 7, 0, 4096, 1, 4096, 0			; triangular window 


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
    elseif (ksel < 18.5) then
        Schan sprintfk "lbp_orderchaos%s", Ssuffix
    elseif (ksel < 19.5) then
        Schan sprintfk "fft_centr%s", Ssuffix
    elseif (ksel < 20.5) then
        Schan sprintfk "act%s", Ssuffix
    else
        Schan sprintfk "act%s", Ssuffix
    endif

    if (ksel >= 1.5) then
        ksrc chnget Schan
        if (ksel >= 17.5 && ksel < 18.5) then
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
    elseif (ksel < 18.5) then
        Schan sprintfk "lbp_orderchaos%s", Ssuffix
    elseif (ksel < 19.5) then
        Schan sprintfk "fft_centr%s", Ssuffix
    elseif (ksel < 20.5) then
        Schan sprintfk "act%s", Ssuffix
    else
        Schan sprintfk "act%s", Ssuffix
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

opcode NoiseGrains, aa, kkkkkkkkkk
    kamp, kwavfreq, kgrainrate, kgraindur, ka_d_ratio, ksustain_amount, ksource_sel, ksamplepos, kgainmask, kchanmask xin

; grain rate
    agrainrate = kgrainrate
    async = 0

; distribution 
	kdistribution = 0; chnget "Distribution"			; grain random distribution in time
	idisttab ftgentmp	0, 0, 16, 16, 1, 16, -10, 0	; probability distribution for random grain masking

; grain shape
	kduration = divz(1,kgrainrate,1)*kgraindur*1000; 

	ienv_attack	= giSigmoRise 			; grain attack shape (from table)
	ienv_decay = giSigmoFall 			; grain decay shape (from table)
	;ksustain_amount	= 0.0					  ; balance between enveloped time(attack+decay) and sustain level time, 0.0 = no time at sustain level
	;ka_d_ratio = 0.1      					; balance between attack time and decay time, 0.0 = zero attack time and full decay time
	kenv2amt = 0                    ; amount of secondary enveloping per grain (e.g. for fof synthesis)
	ienv2tab = giExpFall 				  ; secondary grain shape (from table), enveloping the whole grain if used

; select source waveform
    kwaveform_sel = giSine
    if (ksource_sel > 2.5) then
        kwaveform_sel = giSlitWave
    elseif (ksource_sel > 1.5) then
        kwaveform_sel = giNoise
    endif
    ktablen_sel tableng kwaveform_sel
    if (ksource_sel > 2.5) then
        ; Scale read rate by table length for non-periodic slit-wave playback.
        kwavfreq = (kwavfreq / max(1, ktablen_sel))*100
    endif

; original pitch for each waveform, use if they should be transposed individually
    kwavekey1	= 1
    kwavekey2	= 1
    kwavekey3	= 1
    kwavekey4	= 1
    asamplepos	= limit(ksamplepos, 0, 1)

; "master" grain pitch (transpose for all 4 source waveforms)
    ;kwavfreq	= kwavfreq					; transposition factor (playback speed) of audio inside grains, 
  
; pitch sweep
	ksweepshape		= 0.5						; grain wave pitch sweep shape (sweep speed), 0.5 is linear sweep
	iwavfreqstarttab 	ftgentmp	0, 0, 16, -2, 0, 0,   1		; start freq scalers, per grain
	iwavfreqendtab		ftgentmp	0, 0, 16, -2, 0, 0,   1		; end freq scalers, per grain

; FM of grain pitch (playback speed)
	awavfm = 0

; trainlet parameters (not using trainlets)
	icosine	= giCosine
	kTrainCps	= 100		
	knumpartials = 1	
	kchroma = 1	

	; gain masking table, amplitude for individual grains
    igainmasks ftgentmp 0, 0, 16, -2, 0, 11,  1,1,1,1,1,1,1,1,1,1,1,1
    igainmask1 ftgentmp 0, 0, 16, -2, 0, 11,  1,1,1,1,1,1,1,1,1,1,1,1
    igainmask2 ftgentmp 0, 0, 16, -2, 0, 11,  1,0.5,1,0.5,1,0.5,1,0.5,1,0.5,1,0.5
    igainmask3 ftgentmp 0, 0, 16, -2, 0, 11,  1,0.5,0.25,1,0.5,0.25,1,0.5,0.25,1,0.5,0.25
    igainmask4 ftgentmp 0, 0, 16, -2, 0, 11,  1,0.5,0.25,0.1,1,0.5,0.25,0.1,1,0.5,0.25,0.1
    igainmaskstab ftgentmp 0, 0, 4, -2, igainmask1, igainmask2, igainmask3, igainmask4;, igainmask5, igainmask6, igainmask7, igainmask8
    ftmorf kgainmask-1, igainmaskstab, igainmasks
    

	; channel masking table, output routing for individual grains (zero based, a value of 0.0 routes to output 1)
    ichanmasks ftgentmp 0, 0, 16, -2, 0, 11, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
    ichanmask1 ftgentmp 0, 0, 16, -2, 0, 11, 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
	ichanmask2 ftgentmp 0, 0, 16, -2, 0, 11, 0,1,0,1,0,1,0,1,0,1,0,1
    ichanmask3 ftgentmp 0, 0, 16, -2, 0, 11, 0,0.5,1,0,0.5,1,0,0.5,1,0,0.5,1
    ichanmask4 ftgentmp 0, 0, 16, -2, 0, 11, 0,0.5,1,0.5,0,0.5,1,0.5,0,0.5,1,0.5
    ichanmaskstab ftgentmp 0, 0, 4, -2, ichanmask1, ichanmask2, ichanmask3, ichanmask4;, ichanmask5, ichanmask6, ichanmask7, ichanmask8
    ftmorf kchanmask-1, ichanmaskstab, ichanmasks


	; random masking (muting) of individual grains
	krandommask	=0;chnget "RandMask"

	; wave mix masking: single selected source.
    iwaveamptab	ftgentmp 0, 0, 32, -2,   0, 0,  1,0,0,0,0

; system parameter
	imax_grains	= 100				; max number of grains per k-period
        
	a1,a2	partikkel \					; 					
			agrainrate, \						; grains per second			
			kdistribution, idisttab, async, \			; synchronous/asynchronous		
			kenv2amt, ienv2tab, ienv_attack, ienv_decay, \		; grain envelope (advanced)		
			ksustain_amount, ka_d_ratio, kduration, \		; grain envelope 			
			kamp, \							; amp					
			igainmasks, \						; gain masks (advanced)			
			kwavfreq, \						; grain pitch (playback frequency)	
			ksweepshape, iwavfreqstarttab, iwavfreqendtab, \	; grain pith sweeps (advanced)		
			awavfm, -1, -1, \				; grain pitch FM (advanced)		
			icosine, kTrainCps, knumpartials, kchroma, \		; trainlets				
			ichanmasks, \ 					        ; channel mask (advanced)
			krandommask, \						; random masking of single grains	
            kwaveform_sel, kwaveform_sel, kwaveform_sel, kwaveform_sel, \	; selected source waveform
			iwaveamptab, \						; mix source waveforms (remember, we can use different samplepos and transposition for each)
			asamplepos, asamplepos, asamplepos, asamplepos, \	; read position for source waves	
			kwavekey1, kwavekey2, kwavekey3, kwavekey4, \		; individual transpose for each source
			imax_grains						; system parameter (advanced)
  
  xout(a1,a2)
endop

; GUI handling and button -> instrument events
instr 1
    kng_col  chnget "ng_collapse"
    kng2_col chnget "ng2_collapse"
    kccm_col chnget "ccmap_collapse"
    kwave34_col chnget "wave34_collapse"

    ktrig_ng changed kng_col
    ktrig_ng2 changed kng2_col
    ktrig_ccm changed kccm_col
    ktrig_wave34 changed kwave34_col

    ; Only one module section can be expanded at a time.
    if (ktrig_ng == 1 && kng_col > 0.5) then
        if (kng2_col > 0.5) then
            cabbageSetValue "ng2_collapse", 0, 1
            kng2_col = 0
        endif
        if (kccm_col > 0.5) then
            cabbageSetValue "ccmap_collapse", 0, 1
            kccm_col = 0
        endif
    elseif (ktrig_ng2 == 1 && kng2_col > 0.5) then
        if (kng_col > 0.5) then
            cabbageSetValue "ng_collapse", 0, 1
            kng_col = 0
        endif
        if (kccm_col > 0.5) then
            cabbageSetValue "ccmap_collapse", 0, 1
            kccm_col = 0
        endif
    elseif (ktrig_ccm == 1 && kccm_col > 0.5) then
        if (kng_col > 0.5) then
            cabbageSetValue "ng_collapse", 0, 1
            kng_col = 0
        endif
        if (kng2_col > 0.5) then
            cabbageSetValue "ng2_collapse", 0, 1
            kng2_col = 0
        endif
    endif

    kboot init 1
    ktrig_layout = (ktrig_ng + ktrig_ng2 + ktrig_ccm + ktrig_wave34)
    if (kboot == 1 || ktrig_layout > 0) then
        if (kng_col > 0.5) then
            cabbageSet 1, "ngGroup", "bounds(5, 70, 1250, 290)"
            cabbageSet 1, "ng2Group", "bounds(5, 365, 1250, 100)"
            cabbageSet 1, "ccMapGroup", "bounds(5, 470, 1250, 100)"
            ky_ccm = 470
        elseif (kng2_col > 0.5) then
            cabbageSet 1, "ngGroup", "bounds(5, 70, 1250, 100)"
            cabbageSet 1, "ng2Group", "bounds(5, 175, 1250, 290)"
            cabbageSet 1, "ccMapGroup", "bounds(5, 470, 1250, 100)"
            ky_ccm = 470
        elseif (kccm_col > 0.5) then
            cabbageSet 1, "ngGroup", "bounds(5, 70, 1250, 100)"
            cabbageSet 1, "ng2Group", "bounds(5, 175, 1250, 100)"
            cabbageSet 1, "ccMapGroup", "bounds(5, 280, 1250, 290)"
            ky_ccm = 280
        else
            cabbageSet 1, "ngGroup", "bounds(5, 70, 1250, 100)"
            cabbageSet 1, "ng2Group", "bounds(5, 175, 1250, 100)"
            cabbageSet 1, "ccMapGroup", "bounds(5, 280, 1250, 100)"
            ky_ccm = 280
        endif

        kh_ccm_vis = (kccm_col > 0.5 ? 290 : 100)
        ky_wave34 = ky_ccm + kh_ccm_vis + 5
        kh_wave34_vis = (kwave34_col > 0.5 ? 290 : 100)
        Sw34 sprintfk "bounds(5, %.0f, 1250, %.0f)", ky_wave34, kh_wave34_vis
        cabbageSet 1, "wave34Group", Sw34
        kboot = 0
    endif

    ButtonEvent "inst31_on", 31, 0
    ButtonEvent "inst32_on", 32, 0
    ButtonEvent "inst35_on", 35, 0
    ButtonEvent "ccmap_on", 33, 0
    ButtonEvent "inst34_on", 34, 0

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

    ; instr 32 master reset
    kng_reset chnget "ng_master_reset"
    ktrig_ngr changed kng_reset
    if (ktrig_ngr > 0 && kng_reset > 0.5) then
        cabbageSetValue "m_ng_amp_min",   0, 1
        cabbageSetValue "m_ng_amp_max",   1, 1
        cabbageSetValue "m_ng_wfreq_min", 0, 1
        cabbageSetValue "m_ng_wfreq_max", 1, 1
        cabbageSetValue "m_ng_grate_min", 0, 1
        cabbageSetValue "m_ng_grate_max", 1, 1
        cabbageSetValue "m_ng_gdur_min",  0, 1
        cabbageSetValue "m_ng_gdur_max",  1, 1
        cabbageSetValue "m_ng_adr_min",   0, 1
        cabbageSetValue "m_ng_adr_max",   1, 1
        cabbageSetValue "m_ng_sus_min",   0, 1
        cabbageSetValue "m_ng_sus_max",   1, 1
        cabbageSetValue "m_ng_out_lp_min", 0, 1
        cabbageSetValue "m_ng_out_lp_max", 1, 1
        cabbageSetValue "ng_master_reset", 0, 1
    endif

    ; instr 32 master cal
    kng_cal chnget "ng_master_cal"
    ktrig_ngc changed kng_cal
    if (ktrig_ngc > 0) then
        kng_amp_s   chnget "m_ng_amp_src"
        kng_wfreq_s chnget "m_ng_wfreq_src"
        kng_grate_s chnget "m_ng_grate_src"
        kng_gdur_s  chnget "m_ng_gdur_src"
        kng_adr_s   chnget "m_ng_adr_src"
        kng_sus_s   chnget "m_ng_sus_src"
        kng_out_lp_s chnget "m_ng_out_lp_src"
        if (kng_cal > 0.5) then
            if (kng_amp_s > 1.5) then
                cabbageSetValue "m_ng_amp_cal", 1, 1
            endif
            if (kng_wfreq_s > 1.5) then
                cabbageSetValue "m_ng_wfreq_cal", 1, 1
            endif
            if (kng_grate_s > 1.5) then
                cabbageSetValue "m_ng_grate_cal", 1, 1
            endif
            if (kng_gdur_s > 1.5) then
                cabbageSetValue "m_ng_gdur_cal", 1, 1
            endif
            if (kng_adr_s > 1.5) then
                cabbageSetValue "m_ng_adr_cal", 1, 1
            endif
            if (kng_sus_s > 1.5) then
                cabbageSetValue "m_ng_sus_cal", 1, 1
            endif
            if (kng_out_lp_s > 1.5) then
                cabbageSetValue "m_ng_out_lp_cal", 1, 1
            endif
        else
            cabbageSetValue "m_ng_amp_cal",   0, 1
            cabbageSetValue "m_ng_wfreq_cal", 0, 1
            cabbageSetValue "m_ng_grate_cal", 0, 1
            cabbageSetValue "m_ng_gdur_cal",  0, 1
            cabbageSetValue "m_ng_adr_cal",   0, 1
            cabbageSetValue "m_ng_sus_cal",   0, 1
            cabbageSetValue "m_ng_out_lp_cal", 0, 1
        endif
    endif

    ; instr 35 master reset
    kng2_reset chnget "ng2_master_reset"
    ktrig_ng2r changed kng2_reset
    if (ktrig_ng2r > 0 && kng2_reset > 0.5) then
        cabbageSetValue "m_ng2_amp_min",   0, 1
        cabbageSetValue "m_ng2_amp_max",   1, 1
        cabbageSetValue "m_ng2_wfreq_min", 0, 1
        cabbageSetValue "m_ng2_wfreq_max", 1, 1
        cabbageSetValue "m_ng2_grate_min", 0, 1
        cabbageSetValue "m_ng2_grate_max", 1, 1
        cabbageSetValue "m_ng2_gdur_min",  0, 1
        cabbageSetValue "m_ng2_gdur_max",  1, 1
        cabbageSetValue "m_ng2_adr_min",   0, 1
        cabbageSetValue "m_ng2_adr_max",   1, 1
        cabbageSetValue "m_ng2_sus_min",   0, 1
        cabbageSetValue "m_ng2_sus_max",   1, 1
        cabbageSetValue "m_ng2_out_lp_min", 0, 1
        cabbageSetValue "m_ng2_out_lp_max", 1, 1
        cabbageSetValue "ng2_master_reset", 0, 1
    endif

    ; instr 35 master cal
    kng2_cal chnget "ng2_master_cal"
    ktrig_ng2c changed kng2_cal
    if (ktrig_ng2c > 0) then
        kng2_amp_s   chnget "m_ng2_amp_src"
        kng2_wfreq_s chnget "m_ng2_wfreq_src"
        kng2_grate_s chnget "m_ng2_grate_src"
        kng2_gdur_s  chnget "m_ng2_gdur_src"
        kng2_adr_s   chnget "m_ng2_adr_src"
        kng2_sus_s   chnget "m_ng2_sus_src"
        kng2_out_lp_s chnget "m_ng2_out_lp_src"
        if (kng2_cal > 0.5) then
            if (kng2_amp_s > 1.5) then
                cabbageSetValue "m_ng2_amp_cal", 1, 1
            endif
            if (kng2_wfreq_s > 1.5) then
                cabbageSetValue "m_ng2_wfreq_cal", 1, 1
            endif
            if (kng2_grate_s > 1.5) then
                cabbageSetValue "m_ng2_grate_cal", 1, 1
            endif
            if (kng2_gdur_s > 1.5) then
                cabbageSetValue "m_ng2_gdur_cal", 1, 1
            endif
            if (kng2_adr_s > 1.5) then
                cabbageSetValue "m_ng2_adr_cal", 1, 1
            endif
            if (kng2_sus_s > 1.5) then
                cabbageSetValue "m_ng2_sus_cal", 1, 1
            endif
            if (kng2_out_lp_s > 1.5) then
                cabbageSetValue "m_ng2_out_lp_cal", 1, 1
            endif
        else
            cabbageSetValue "m_ng2_amp_cal",   0, 1
            cabbageSetValue "m_ng2_wfreq_cal", 0, 1
            cabbageSetValue "m_ng2_grate_cal", 0, 1
            cabbageSetValue "m_ng2_gdur_cal",  0, 1
            cabbageSetValue "m_ng2_adr_cal",   0, 1
            cabbageSetValue "m_ng2_sus_cal",   0, 1
            cabbageSetValue "m_ng2_out_lp_cal", 0, 1
        endif
    endif

    ; instr 33 master reset
    kccm_reset chnget "ccm_master_reset"
    ktrig_ccmr changed kccm_reset
    if (ktrig_ccmr > 0 && kccm_reset > 0.5) then
        cabbageSetValue "m_ccm1_min", 0, 1
        cabbageSetValue "m_ccm1_max", 1, 1
        cabbageSetValue "m_ccm2_min", 0, 1
        cabbageSetValue "m_ccm2_max", 1, 1
        cabbageSetValue "m_ccm3_min", 0, 1
        cabbageSetValue "m_ccm3_max", 1, 1
        cabbageSetValue "m_ccm4_min", 0, 1
        cabbageSetValue "m_ccm4_max", 1, 1
        cabbageSetValue "m_ccm5_min", 0, 1
        cabbageSetValue "m_ccm5_max", 1, 1
        cabbageSetValue "m_ccm6_min", 0, 1
        cabbageSetValue "m_ccm6_max", 1, 1
        cabbageSetValue "m_ccm7_min", 0, 1
        cabbageSetValue "m_ccm7_max", 1, 1
        cabbageSetValue "m_ccm8_min", 0, 1
        cabbageSetValue "m_ccm8_max", 1, 1
        cabbageSetValue "ccm_master_reset", 0, 1
    endif

    ; instr 33 master cal
    kccm_cal chnget "ccm_master_cal"
    ktrig_ccmc changed kccm_cal
    if (ktrig_ccmc > 0) then
        kccm1_s chnget "m_ccm1_src"
        kccm2_s chnget "m_ccm2_src"
        kccm3_s chnget "m_ccm3_src"
        kccm4_s chnget "m_ccm4_src"
        kccm5_s chnget "m_ccm5_src"
        kccm6_s chnget "m_ccm6_src"
        kccm7_s chnget "m_ccm7_src"
        kccm8_s chnget "m_ccm8_src"
        if (kccm_cal > 0.5) then
            if (kccm1_s > 1.5) then
                cabbageSetValue "m_ccm1_cal", 1, 1
            endif
            if (kccm2_s > 1.5) then
                cabbageSetValue "m_ccm2_cal", 1, 1
            endif
            if (kccm3_s > 1.5) then
                cabbageSetValue "m_ccm3_cal", 1, 1
            endif
            if (kccm4_s > 1.5) then
                cabbageSetValue "m_ccm4_cal", 1, 1
            endif
            if (kccm5_s > 1.5) then
                cabbageSetValue "m_ccm5_cal", 1, 1
            endif
            if (kccm6_s > 1.5) then
                cabbageSetValue "m_ccm6_cal", 1, 1
            endif
            if (kccm7_s > 1.5) then
                cabbageSetValue "m_ccm7_cal", 1, 1
            endif
            if (kccm8_s > 1.5) then
                cabbageSetValue "m_ccm8_cal", 1, 1
            endif
        else
            cabbageSetValue "m_ccm1_cal", 0, 1
            cabbageSetValue "m_ccm2_cal", 0, 1
            cabbageSetValue "m_ccm3_cal", 0, 1
            cabbageSetValue "m_ccm4_cal", 0, 1
            cabbageSetValue "m_ccm5_cal", 0, 1
            cabbageSetValue "m_ccm6_cal", 0, 1
            cabbageSetValue "m_ccm7_cal", 0, 1
            cabbageSetValue "m_ccm8_cal", 0, 1
        endif
    endif
endin

    ; wave34 master reset
    kw_reset chnget "w_master_reset"
    ktrig_wr changed kw_reset
    if (ktrig_wr > 0 && kw_reset > 0.5) then
        cabbageSetValue "m_w_freq_min",   0, 1
        cabbageSetValue "m_w_freq_max",   1, 1
        cabbageSetValue "m_w_start_min",  0, 1
        cabbageSetValue "m_w_start_max",  1, 1
        cabbageSetValue "m_w_end_min",    0, 1
        cabbageSetValue "m_w_end_max",    1, 1
        cabbageSetValue "m_w_amp_min",    0, 1
        cabbageSetValue "m_w_amp_max",    1, 1
        cabbageSetValue "m_w_tone_min",   0, 1
        cabbageSetValue "m_w_tone_max",   1, 1
        cabbageSetValue "m_w_brite_min",  0, 1
        cabbageSetValue "m_w_brite_max",  1, 1
        cabbageSetValue "m_w_gain_min",   0, 1
        cabbageSetValue "m_w_gain_max",   1, 1
        cabbageSetValue "m_w_center_min", 0, 1
        cabbageSetValue "m_w_center_max", 1, 1
        cabbageSetValue "m_w_lp_min",     0, 1
        cabbageSetValue "m_w_lp_max",     1, 1
        cabbageSetValue "w_master_reset", 0, 1
    endif

    ; wave34 master cal
    kw_cal chnget "w_master_cal"
    ktrig_wc changed kw_cal
    if (ktrig_wc > 0) then
        kw_freq_s   chnget "m_w_freq_src"
        kw_start_s  chnget "m_w_start_src"
        kw_end_s    chnget "m_w_end_src"
        kw_amp_s    chnget "m_w_amp_src"
        kw_tone_s   chnget "m_w_tone_src"
        kw_brite_s  chnget "m_w_brite_src"
        kw_gain_s   chnget "m_w_gain_src"
        kw_center_s chnget "m_w_center_src"
        kw_lp_s     chnget "m_w_lp_src"
        if (kw_cal > 0.5) then
            if (kw_freq_s   > 1.5) then
                cabbageSetValue "m_w_freq_cal",   1, 1
            endif
            if (kw_start_s  > 1.5) then
                cabbageSetValue "m_w_start_cal",  1, 1
            endif
            if (kw_end_s    > 1.5) then
                cabbageSetValue "m_w_end_cal",    1, 1
            endif
            if (kw_amp_s    > 1.5) then
                cabbageSetValue "m_w_amp_cal",    1, 1
            endif
            if (kw_tone_s   > 1.5) then
                cabbageSetValue "m_w_tone_cal",   1, 1
            endif
            if (kw_brite_s  > 1.5) then
                cabbageSetValue "m_w_brite_cal",  1, 1
            endif
            if (kw_gain_s   > 1.5) then
                cabbageSetValue "m_w_gain_cal",   1, 1
            endif
            if (kw_center_s > 1.5) then
                cabbageSetValue "m_w_center_cal", 1, 1
            endif
            if (kw_lp_s     > 1.5) then
                cabbageSetValue "m_w_lp_cal",     1, 1
            endif
        else
            cabbageSetValue "m_w_freq_cal",   0, 1
            cabbageSetValue "m_w_start_cal",  0, 1
            cabbageSetValue "m_w_end_cal",    0, 1
            cabbageSetValue "m_w_amp_cal",    0, 1
            cabbageSetValue "m_w_tone_cal",   0, 1
            cabbageSetValue "m_w_brite_cal",  0, 1
            cabbageSetValue "m_w_gain_cal",   0, 1
            cabbageSetValue "m_w_center_cal", 0, 1
            cabbageSetValue "m_w_lp_cal",     0, 1
        endif
    endif

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
    kact0 init 0
    kact1 init 0
    kact2 init 0
    kact3 init 0
    kact4 init 0
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
    kslFrame init -1
    kslChunk init 0
    kslCount init 0
    ksf00 init 0
    ksf01 init 0
    ksf02 init 0
    ksf03 init 0
    ksf04 init 0
    ksf05 init 0
    ksf06 init 0
    ksf07 init 0
    ksf08 init 0
    ksf09 init 0
    ksf10 init 0
    ksf11 init 0
    ksf12 init 0
    ksf13 init 0
    ksf14 init 0
    ksf15 init 0
    kSlitFrameId init -1
    kSlitChunkCount init 32
    kSlitReceived init 0
    kfftCentroid init 0

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
        chnset (-sin(kfrad)) / 2 + 0.5, "flow_fast_y_0"
        chnset kfmag, "flow_fast_mag_0"
        kgoto read_flow_fast
    endif

read_flow_slow:
    ktr OSClisten gihOsc, "/wave/flow/slow_pack", "fff", ksdeg, ksmag, kscoh
    if (ktr == 1) then
        ksrad = ksdeg * $M_PI / 180
        chnset cos(ksrad) / 2 + 0.5, "flow_slow_x_0"
        chnset (-sin(ksrad)) / 2 + 0.5, "flow_slow_y_0"
        chnset ksmag, "flow_slow_mag_0"
        kgoto read_flow_slow
    endif

    ; Per-quadrant fast flow: [UL_dir, UL_act, UR_dir, UR_act, LL_dir, LL_act, LR_dir, LR_act]
read_flow_fast_quad:
    ktr OSClisten gihOsc, "/wave/flow/fast_quad_pack", "ffffffff", kfq1d, kfq1a, kfq2d, kfq2a, kfq3d, kfq3a, kfq4d, kfq4a
    if (ktr == 1) then
        kfq1r = kfq1d * $M_PI / 180
        chnset cos(kfq1r) / 2 + 0.5, "flow_fast_x_1"
        chnset (-sin(kfq1r)) / 2 + 0.5, "flow_fast_y_1"
        chnset kfq1a, "flow_fast_mag_1"
        kfq2r = kfq2d * $M_PI / 180
        chnset cos(kfq2r) / 2 + 0.5, "flow_fast_x_2"
        chnset (-sin(kfq2r)) / 2 + 0.5, "flow_fast_y_2"
        chnset kfq2a, "flow_fast_mag_2"
        kfq3r = kfq3d * $M_PI / 180
        chnset cos(kfq3r) / 2 + 0.5, "flow_fast_x_3"
        chnset (-sin(kfq3r)) / 2 + 0.5, "flow_fast_y_3"
        chnset kfq3a, "flow_fast_mag_3"
        kfq4r = kfq4d * $M_PI / 180
        chnset cos(kfq4r) / 2 + 0.5, "flow_fast_x_4"
        chnset (-sin(kfq4r)) / 2 + 0.5, "flow_fast_y_4"
        chnset kfq4a, "flow_fast_mag_4"
        kgoto read_flow_fast_quad
    endif

    ; Per-quadrant slow flow: [UL_dir, UL_act, UR_dir, UR_act, LL_dir, LL_act, LR_dir, LR_act]
read_flow_slow_quad:
    ktr OSClisten gihOsc, "/wave/flow/slow_quad_pack", "ffffffff", ksq1d, ksq1a, ksq2d, ksq2a, ksq3d, ksq3a, ksq4d, ksq4a
    if (ktr == 1) then
        ksq1r = ksq1d * $M_PI / 180
        chnset cos(ksq1r) / 2 + 0.5, "flow_slow_x_1"
        chnset (-sin(ksq1r)) / 2 + 0.5, "flow_slow_y_1"
        chnset ksq1a, "flow_slow_mag_1"
        ksq2r = ksq2d * $M_PI / 180
        chnset cos(ksq2r) / 2 + 0.5, "flow_slow_x_2"
        chnset (-sin(ksq2r)) / 2 + 0.5, "flow_slow_y_2"
        chnset ksq2a, "flow_slow_mag_2"
        ksq3r = ksq3d * $M_PI / 180
        chnset cos(ksq3r) / 2 + 0.5, "flow_slow_x_3"
        chnset (-sin(ksq3r)) / 2 + 0.5, "flow_slow_y_3"
        chnset ksq3a, "flow_slow_mag_3"
        ksq4r = ksq4d * $M_PI / 180
        chnset cos(ksq4r) / 2 + 0.5, "flow_slow_x_4"
        chnset (-sin(ksq4r)) / 2 + 0.5, "flow_slow_y_4"
        chnset ksq4a, "flow_slow_mag_4"
        kgoto read_flow_slow_quad
    endif

    ; Fused activity: global + UL/UR/LL/LR
read_act:
    ktr OSClisten gihOsc, "/wave/activity/pack", "fffff", kact0, kact1, kact2, kact3, kact4
    if (ktr == 1) then
        chnset kact0, "act_0"
        chnset kact1, "act_1"
        chnset kact2, "act_2"
        chnset kact3, "act_3"
        chnset kact4, "act_4"
        kgoto read_act
    endif

read_slit_fft_centroid:
    ktr OSClisten gihOsc, "/wave/slit_fft/centroid", "f", kfftCentroid
    if (ktr == 1) then
        chnset kfftCentroid, "fft_centr_0"
        chnset kfftCentroid, "fft_centr_1"
        chnset kfftCentroid, "fft_centr_2"
        chnset kfftCentroid, "fft_centr_3"
        chnset kfftCentroid, "fft_centr_4"
        kgoto read_slit_fft_centroid
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

    ; Live slit waveform chunks: frame_id, chunk_idx, chunk_count, 16 samples
read_slit_chunk:
    ktr OSClisten gihOsc, "/slit/chunk", "iiiffffffffffffffff", \
        kslFrame, kslChunk, kslCount, \
        ksf00, ksf01, ksf02, ksf03, ksf04, ksf05, ksf06, ksf07, \
        ksf08, ksf09, ksf10, ksf11, ksf12, ksf13, ksf14, ksf15
    if (ktr == 1) then
        if (kslCount > 0 && kslCount <= 32) then
            if (kslFrame != kSlitFrameId || kslCount != kSlitChunkCount) then
                kSlitFrameId = kslFrame
                kSlitChunkCount = kslCount
                kSlitReceived = 0
                kidx = 0
                while (kidx < 32) do
                    tablew 0, kidx, giSlitSeen
                    kidx += 1
                od
                chnset 0, "slit_ready"
            endif

            if (kslChunk >= 0 && kslChunk < kSlitChunkCount) then
                kseen table kslChunk, giSlitSeen
                if (kseen < 0.5) then
                    tablew 1, kslChunk, giSlitSeen
                    kSlitReceived += 1
                endif

                kbase = kslChunk * 16
                if (kbase + 15 < 512) then
                    tablew ksf00, kbase + 0, giSlitWave
                    tablew ksf01, kbase + 1, giSlitWave
                    tablew ksf02, kbase + 2, giSlitWave
                    tablew ksf03, kbase + 3, giSlitWave
                    tablew ksf04, kbase + 4, giSlitWave
                    tablew ksf05, kbase + 5, giSlitWave
                    tablew ksf06, kbase + 6, giSlitWave
                    tablew ksf07, kbase + 7, giSlitWave
                    tablew ksf08, kbase + 8, giSlitWave
                    tablew ksf09, kbase + 9, giSlitWave
                    tablew ksf10, kbase + 10, giSlitWave
                    tablew ksf11, kbase + 11, giSlitWave
                    tablew ksf12, kbase + 12, giSlitWave
                    tablew ksf13, kbase + 13, giSlitWave
                    tablew ksf14, kbase + 14, giSlitWave
                    tablew ksf15, kbase + 15, giSlitWave
                endif
            endif

            if (kSlitReceived >= kSlitChunkCount && kSlitChunkCount == 32) then
                chnset 1, "slit_ready"
                chnset kSlitFrameId, "slit_frame"
            endif
        endif
        kgoto read_slit_chunk
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
    UpdateMixSums "fft_centr"
    UpdateMixSums "act"
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



instr 32

    ; Base values from GUI rsliders
    kampb    chnget "ng_amp_base"
    kwfreqb  chnget "ng_wfreq_base"
    kgrateb  chnget "ng_grate_base"
    kgdurb   chnget "ng_gdur_base"
    kadrb    chnget "ng_adr_base"
    ksusb    chnget "ng_sus_base"
    ksource_sel chnget "ng_source_sel"
    ksamplepos chnget "ng_samplepos"

    ; Shared mod router temps
    ks   init 0
    kmin init 0
    kmax init 1
    kexp init 1
    ko   init 0
    kg   init 0
    ka   init 1
    km   init 1
    kc   init 0
    klp  init 20

    ks chnget "m_ng_amp_src"
    kmin chnget "m_ng_amp_min"
    kmax chnget "m_ng_amp_max"
    kexp chnget "m_ng_amp_exp"
    ko chnget "m_ng_amp_offs"
    kg chnget "m_ng_amp_gain"
    ka chnget "m_ng_amp_area"
    km chnget "m_ng_amp_mode"
    kc chnget "m_ng_amp_cal"
    klp chnget "m_ng_amp_lp"
    kamp ApplyMod "ng_amp", kampb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng_wfreq_src"
    kmin chnget "m_ng_wfreq_min"
    kmax chnget "m_ng_wfreq_max"
    kexp chnget "m_ng_wfreq_exp"
    ko chnget "m_ng_wfreq_offs"
    kg chnget "m_ng_wfreq_gain"
    ka chnget "m_ng_wfreq_area"
    km chnget "m_ng_wfreq_mode"
    kc chnget "m_ng_wfreq_cal"
    klp chnget "m_ng_wfreq_lp"
    kwavfreq ApplyMod "ng_wfreq", kwfreqb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng_grate_src"
    kmin chnget "m_ng_grate_min"
    kmax chnget "m_ng_grate_max"
    kexp chnget "m_ng_grate_exp"
    ko chnget "m_ng_grate_offs"
    kg chnget "m_ng_grate_gain"
    ka chnget "m_ng_grate_area"
    km chnget "m_ng_grate_mode"
    kc chnget "m_ng_grate_cal"
    klp chnget "m_ng_grate_lp"
    kgrainrate ApplyMod "ng_grate", kgrateb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ; Random spline deviation for wavfreq and grainrate.
    ; Deviation is additive, but scaled by the current parameter value:
    ; kparam_eff = kparam + (kparam * krdev), where krdev is in [0, rdev_amp].
    kwfreq_rdev_amp chnget "ng_wfreq_rdev_base"
    kwfreq_rdev_amp limit kwfreq_rdev_amp, 0, 20
    kwfreq_rdev_minfreq chnget "ng_wfreq_rdev_minfreq"
    kwfreq_rdev_maxfreq chnget "ng_wfreq_rdev_maxfreq"
    kwfreq_rdev_minfreq = max(0, kwfreq_rdev_minfreq)
    kwfreq_rdev_maxfreq = max(kwfreq_rdev_minfreq, kwfreq_rdev_maxfreq)
    kwfreq_rdev rspline 0, kwfreq_rdev_amp, kwfreq_rdev_minfreq, kwfreq_rdev_maxfreq
    kwavfreq = kwavfreq + (kwavfreq * kwfreq_rdev)
    kwavfreq = max(0.001, kwavfreq)

    kgrate_rdev_amp chnget "ng_grate_rdev_base"
    kgrate_rdev_amp limit kgrate_rdev_amp, 0, 20
    kgrate_rdev_minfreq chnget "ng_grate_rdev_minfreq"
    kgrate_rdev_maxfreq chnget "ng_grate_rdev_maxfreq"
    kgrate_rdev_minfreq = max(0, kgrate_rdev_minfreq)
    kgrate_rdev_maxfreq = max(kgrate_rdev_minfreq, kgrate_rdev_maxfreq)
    kgrate_rdev rspline 0, kgrate_rdev_amp, kgrate_rdev_minfreq, kgrate_rdev_maxfreq
    kgrainrate = kgrainrate + (kgrainrate * kgrate_rdev)
    kgrainrate = max(0.001, kgrainrate)

    ks chnget "m_ng_gdur_src"
    kmin chnget "m_ng_gdur_min"
    kmax chnget "m_ng_gdur_max"
    kexp chnget "m_ng_gdur_exp"
    ko chnget "m_ng_gdur_offs"
    kg chnget "m_ng_gdur_gain"
    ka chnget "m_ng_gdur_area"
    km chnget "m_ng_gdur_mode"
    kc chnget "m_ng_gdur_cal"
    klp chnget "m_ng_gdur_lp"
    kgraindur ApplyMod "ng_gdur", kgdurb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng_adr_src"
    kmin chnget "m_ng_adr_min"
    kmax chnget "m_ng_adr_max"
    kexp chnget "m_ng_adr_exp"
    ko chnget "m_ng_adr_offs"
    kg chnget "m_ng_adr_gain"
    ka chnget "m_ng_adr_area"
    km chnget "m_ng_adr_mode"
    kc chnget "m_ng_adr_cal"
    klp chnget "m_ng_adr_lp"
    ka_d_ratio ApplyMod "ng_adr", kadrb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng_sus_src"
    kmin chnget "m_ng_sus_min"
    kmax chnget "m_ng_sus_max"
    kexp chnget "m_ng_sus_exp"
    ko chnget "m_ng_sus_offs"
    kg chnget "m_ng_sus_gain"
    ka chnget "m_ng_sus_area"
    km chnget "m_ng_sus_mode"
    kc chnget "m_ng_sus_cal"
    klp chnget "m_ng_sus_lp"
    ksustain_amount ApplyMod "ng_sus", ksusb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    kout_lpfb chnget "ng_out_lp_base"
    ks chnget "m_ng_out_lp_src"
    kmin chnget "m_ng_out_lp_min"
    kmax chnget "m_ng_out_lp_max"
    kexp chnget "m_ng_out_lp_exp"
    ko chnget "m_ng_out_lp_offs"
    kg chnget "m_ng_out_lp_gain"
    ka chnget "m_ng_out_lp_area"
    km chnget "m_ng_out_lp_mode"
    kc chnget "m_ng_out_lp_cal"
    klp chnget "m_ng_out_lp_lp"
    kout_lpf ApplyMod "ng_out_lp", kout_lpfb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp
    kout_lpf limit kout_lpf, 20, 20000

    ; Update view rsliders
    ktrig_ng_amp = changed(kamp)
    cabbageSetValue "ng_amp_view", kamp, ktrig_ng_amp
    ktrig_ng_wfreq = changed(kwavfreq)
    cabbageSetValue "ng_wfreq_view", kwavfreq, ktrig_ng_wfreq
    ktrig_ng_grate = changed(kgrainrate)
    cabbageSetValue "ng_grate_view", kgrainrate, ktrig_ng_grate
    ktrig_ng_gdur = changed(kgraindur)
    cabbageSetValue "ng_gdur_view", kgraindur, ktrig_ng_gdur
    ktrig_ng_adr = changed(ka_d_ratio)
    cabbageSetValue "ng_adr_view", ka_d_ratio, ktrig_ng_adr
    ktrig_ng_sus = changed(ksustain_amount)
    cabbageSetValue "ng_sus_view", ksustain_amount, ktrig_ng_sus
    ktrig_ng_out_lp = changed(kout_lpf)
    cabbageSetValue "ng_out_lp_view", kout_lpf, ktrig_ng_out_lp

    kgainmask chnget "GainMask" 
    kchanmask chnget "ChanMask"

    a1,a2 NoiseGrains kamp, kwavfreq, kgrainrate, kgraindur, ka_d_ratio, ksustain_amount, ksource_sel, ksamplepos, kgainmask, kchanmask
    a1 lpf18 a1, kout_lpf, 0.3, 0.3
    a2 lpf18 a2, kout_lpf, 0.3, 0.3
    outs a1, a2

endin

instr 35

    ; Base values from GUI rsliders
    kampb    chnget "ng2_amp_base"
    kwfreqb  chnget "ng2_wfreq_base"
    kgrateb  chnget "ng2_grate_base"
    kgdurb   chnget "ng2_gdur_base"
    kadrb    chnget "ng2_adr_base"
    ksusb    chnget "ng2_sus_base"
    ksource_sel chnget "ng2_source_sel"
    ksamplepos chnget "ng2_samplepos"

    ; Shared mod router temps
    ks   init 0
    kmin init 0
    kmax init 1
    kexp init 1
    ko   init 0
    kg   init 0
    ka   init 1
    km   init 1
    kc   init 0
    klp  init 20

    ks chnget "m_ng2_amp_src"
    kmin chnget "m_ng2_amp_min"
    kmax chnget "m_ng2_amp_max"
    kexp chnget "m_ng2_amp_exp"
    ko chnget "m_ng2_amp_offs"
    kg chnget "m_ng2_amp_gain"
    ka chnget "m_ng2_amp_area"
    km chnget "m_ng2_amp_mode"
    kc chnget "m_ng2_amp_cal"
    klp chnget "m_ng2_amp_lp"
    kamp ApplyMod "ng2_amp", kampb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng2_wfreq_src"
    kmin chnget "m_ng2_wfreq_min"
    kmax chnget "m_ng2_wfreq_max"
    kexp chnget "m_ng2_wfreq_exp"
    ko chnget "m_ng2_wfreq_offs"
    kg chnget "m_ng2_wfreq_gain"
    ka chnget "m_ng2_wfreq_area"
    km chnget "m_ng2_wfreq_mode"
    kc chnget "m_ng2_wfreq_cal"
    klp chnget "m_ng2_wfreq_lp"
    kwavfreq ApplyMod "ng2_wfreq", kwfreqb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng2_grate_src"
    kmin chnget "m_ng2_grate_min"
    kmax chnget "m_ng2_grate_max"
    kexp chnget "m_ng2_grate_exp"
    ko chnget "m_ng2_grate_offs"
    kg chnget "m_ng2_grate_gain"
    ka chnget "m_ng2_grate_area"
    km chnget "m_ng2_grate_mode"
    kc chnget "m_ng2_grate_cal"
    klp chnget "m_ng2_grate_lp"
    kgrainrate ApplyMod "ng2_grate", kgrateb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ; Random spline deviation for wavfreq and grainrate.
    ; Deviation is additive, but scaled by the current parameter value:
    ; kparam_eff = kparam + (kparam * krdev), where krdev is in [0, rdev_amp].
    kwfreq_rdev_amp chnget "ng2_wfreq_rdev_base"
    kwfreq_rdev_amp limit kwfreq_rdev_amp, 0, 20
    kwfreq_rdev_minfreq chnget "ng2_wfreq_rdev_minfreq"
    kwfreq_rdev_maxfreq chnget "ng2_wfreq_rdev_maxfreq"
    kwfreq_rdev_minfreq = max(0, kwfreq_rdev_minfreq)
    kwfreq_rdev_maxfreq = max(kwfreq_rdev_minfreq, kwfreq_rdev_maxfreq)
    kwfreq_rdev rspline 0, kwfreq_rdev_amp, kwfreq_rdev_minfreq, kwfreq_rdev_maxfreq
    kwavfreq = kwavfreq + (kwavfreq * kwfreq_rdev)
    kwavfreq = max(0.001, kwavfreq)

    kgrate_rdev_amp chnget "ng2_grate_rdev_base"
    kgrate_rdev_amp limit kgrate_rdev_amp, 0, 20
    kgrate_rdev_minfreq chnget "ng2_grate_rdev_minfreq"
    kgrate_rdev_maxfreq chnget "ng2_grate_rdev_maxfreq"
    kgrate_rdev_minfreq = max(0, kgrate_rdev_minfreq)
    kgrate_rdev_maxfreq = max(kgrate_rdev_minfreq, kgrate_rdev_maxfreq)
    kgrate_rdev rspline 0, kgrate_rdev_amp, kgrate_rdev_minfreq, kgrate_rdev_maxfreq
    kgrainrate = kgrainrate + (kgrainrate * kgrate_rdev)
    kgrainrate = max(0.001, kgrainrate)

    ks chnget "m_ng2_gdur_src"
    kmin chnget "m_ng2_gdur_min"
    kmax chnget "m_ng2_gdur_max"
    kexp chnget "m_ng2_gdur_exp"
    ko chnget "m_ng2_gdur_offs"
    kg chnget "m_ng2_gdur_gain"
    ka chnget "m_ng2_gdur_area"
    km chnget "m_ng2_gdur_mode"
    kc chnget "m_ng2_gdur_cal"
    klp chnget "m_ng2_gdur_lp"
    kgraindur ApplyMod "ng2_gdur", kgdurb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng2_adr_src"
    kmin chnget "m_ng2_adr_min"
    kmax chnget "m_ng2_adr_max"
    kexp chnget "m_ng2_adr_exp"
    ko chnget "m_ng2_adr_offs"
    kg chnget "m_ng2_adr_gain"
    ka chnget "m_ng2_adr_area"
    km chnget "m_ng2_adr_mode"
    kc chnget "m_ng2_adr_cal"
    klp chnget "m_ng2_adr_lp"
    ka_d_ratio ApplyMod "ng2_adr", kadrb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ng2_sus_src"
    kmin chnget "m_ng2_sus_min"
    kmax chnget "m_ng2_sus_max"
    kexp chnget "m_ng2_sus_exp"
    ko chnget "m_ng2_sus_offs"
    kg chnget "m_ng2_sus_gain"
    ka chnget "m_ng2_sus_area"
    km chnget "m_ng2_sus_mode"
    kc chnget "m_ng2_sus_cal"
    klp chnget "m_ng2_sus_lp"
    ksustain_amount ApplyMod "ng2_sus", ksusb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    kout_lpfb chnget "ng2_out_lp_base"
    ks chnget "m_ng2_out_lp_src"
    kmin chnget "m_ng2_out_lp_min"
    kmax chnget "m_ng2_out_lp_max"
    kexp chnget "m_ng2_out_lp_exp"
    ko chnget "m_ng2_out_lp_offs"
    kg chnget "m_ng2_out_lp_gain"
    ka chnget "m_ng2_out_lp_area"
    km chnget "m_ng2_out_lp_mode"
    kc chnget "m_ng2_out_lp_cal"
    klp chnget "m_ng2_out_lp_lp"
    kout_lpf ApplyMod "ng2_out_lp", kout_lpfb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp
    kout_lpf limit kout_lpf, 20, 20000

    ; Update view rsliders
    ktrig_ng_amp = changed(kamp)
    cabbageSetValue "ng2_amp_view", kamp, ktrig_ng_amp
    ktrig_ng_wfreq = changed(kwavfreq)
    cabbageSetValue "ng2_wfreq_view", kwavfreq, ktrig_ng_wfreq
    ktrig_ng_grate = changed(kgrainrate)
    cabbageSetValue "ng2_grate_view", kgrainrate, ktrig_ng_grate
    ktrig_ng_gdur = changed(kgraindur)
    cabbageSetValue "ng2_gdur_view", kgraindur, ktrig_ng_gdur
    ktrig_ng_adr = changed(ka_d_ratio)
    cabbageSetValue "ng2_adr_view", ka_d_ratio, ktrig_ng_adr
    ktrig_ng_sus = changed(ksustain_amount)
    cabbageSetValue "ng2_sus_view", ksustain_amount, ktrig_ng_sus
    ktrig_ng_out_lp = changed(kout_lpf)
    cabbageSetValue "ng2_out_lp_view", kout_lpf, ktrig_ng_out_lp

    kgainmask chnget "GainMask2" 
    kchanmask chnget "ChanMask2"

    a1,a2 NoiseGrains kamp, kwavfreq, kgrainrate, kgraindur, ka_d_ratio, ksustain_amount, ksource_sel, ksamplepos, kgainmask, kchanmask
    a1 lpf18 a1, kout_lpf, 0.3, 0.3
    a2 lpf18 a2, kout_lpf, 0.3, 0.3
    outs a1, a2

endin

instr 33
    kb1 chnget "ccm1_base"
    kb2 chnget "ccm2_base"
    kb3 chnget "ccm3_base"
    kb4 chnget "ccm4_base"
    kb5 chnget "ccm5_base"
    kb6 chnget "ccm6_base"
    kb7 chnget "ccm7_base"
    kb8 chnget "ccm8_base"

    ks init 0
    kmin init 0
    kmax init 1
    kexp init 1
    ko init 0
    kg init 0
    ka init 1
    km init 1
    kc init 0
    klp init 20

    ks chnget "m_ccm1_src"
    kmin chnget "m_ccm1_min"
    kmax chnget "m_ccm1_max"
    kexp chnget "m_ccm1_exp"
    ko chnget "m_ccm1_offs"
    kg chnget "m_ccm1_gain"
    ka chnget "m_ccm1_area"
    km chnget "m_ccm1_mode"
    kc chnget "m_ccm1_cal"
    klp chnget "m_ccm1_lp"
    k1 ApplyMod "ccm1", kb1, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ccm2_src"
    kmin chnget "m_ccm2_min"
    kmax chnget "m_ccm2_max"
    kexp chnget "m_ccm2_exp"
    ko chnget "m_ccm2_offs"
    kg chnget "m_ccm2_gain"
    ka chnget "m_ccm2_area"
    km chnget "m_ccm2_mode"
    kc chnget "m_ccm2_cal"
    klp chnget "m_ccm2_lp"
    k2 ApplyMod "ccm2", kb2, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ccm3_src"
    kmin chnget "m_ccm3_min"
    kmax chnget "m_ccm3_max"
    kexp chnget "m_ccm3_exp"
    ko chnget "m_ccm3_offs"
    kg chnget "m_ccm3_gain"
    ka chnget "m_ccm3_area"
    km chnget "m_ccm3_mode"
    kc chnget "m_ccm3_cal"
    klp chnget "m_ccm3_lp"
    k3 ApplyMod "ccm3", kb3, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ccm4_src"
    kmin chnget "m_ccm4_min"
    kmax chnget "m_ccm4_max"
    kexp chnget "m_ccm4_exp"
    ko chnget "m_ccm4_offs"
    kg chnget "m_ccm4_gain"
    ka chnget "m_ccm4_area"
    km chnget "m_ccm4_mode"
    kc chnget "m_ccm4_cal"
    klp chnget "m_ccm4_lp"
    k4 ApplyMod "ccm4", kb4, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ccm5_src"
    kmin chnget "m_ccm5_min"
    kmax chnget "m_ccm5_max"
    kexp chnget "m_ccm5_exp"
    ko chnget "m_ccm5_offs"
    kg chnget "m_ccm5_gain"
    ka chnget "m_ccm5_area"
    km chnget "m_ccm5_mode"
    kc chnget "m_ccm5_cal"
    klp chnget "m_ccm5_lp"
    k5 ApplyMod "ccm5", kb5, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ccm6_src"
    kmin chnget "m_ccm6_min"
    kmax chnget "m_ccm6_max"
    kexp chnget "m_ccm6_exp"
    ko chnget "m_ccm6_offs"
    kg chnget "m_ccm6_gain"
    ka chnget "m_ccm6_area"
    km chnget "m_ccm6_mode"
    kc chnget "m_ccm6_cal"
    klp chnget "m_ccm6_lp"
    k6 ApplyMod "ccm6", kb6, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ccm7_src"
    kmin chnget "m_ccm7_min"
    kmax chnget "m_ccm7_max"
    kexp chnget "m_ccm7_exp"
    ko chnget "m_ccm7_offs"
    kg chnget "m_ccm7_gain"
    ka chnget "m_ccm7_area"
    km chnget "m_ccm7_mode"
    kc chnget "m_ccm7_cal"
    klp chnget "m_ccm7_lp"
    k7 ApplyMod "ccm7", kb7, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_ccm8_src"
    kmin chnget "m_ccm8_min"
    kmax chnget "m_ccm8_max"
    kexp chnget "m_ccm8_exp"
    ko chnget "m_ccm8_offs"
    kg chnget "m_ccm8_gain"
    ka chnget "m_ccm8_area"
    km chnget "m_ccm8_mode"
    kc chnget "m_ccm8_cal"
    klp chnget "m_ccm8_lp"
    k8 ApplyMod "ccm8", kb8, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ktr1 changed k1
    cabbageSetValue "ccm1_view", k1, ktr1
    ktr2 changed k2
    cabbageSetValue "ccm2_view", k2, ktr2
    ktr3 changed k3
    cabbageSetValue "ccm3_view", k3, ktr3
    ktr4 changed k4
    cabbageSetValue "ccm4_view", k4, ktr4
    ktr5 changed k5
    cabbageSetValue "ccm5_view", k5, ktr5
    ktr6 changed k6
    cabbageSetValue "ccm6_view", k6, ktr6
    ktr7 changed k7
    cabbageSetValue "ccm7_view", k7, ktr7
    ktr8 changed k8
    cabbageSetValue "ccm8_view", k8, ktr8

    kchan1 = 1
    kv1 = int(limit(k1, 0, 127) + 0.5)
    kv2 = int(limit(k2, 0, 127) + 0.5)
    kv3 = int(limit(k3, 0, 127) + 0.5)
    kv4 = int(limit(k4, 0, 127) + 0.5)
    kv5 = int(limit(k5, 0, 127) + 0.5)
    kv6 = int(limit(k6, 0, 127) + 0.5)
    kv7 = int(limit(k7, 0, 127) + 0.5)
    kv8 = int(limit(k8, 0, 127) + 0.5)

    kchg1 changed kv1
    if (kchg1 == 1) then
        midiout 176, kchan1, 32, kv1
    endif
    kchg2 changed kv2
    if (kchg2 == 1) then
        midiout 176, kchan1, 33, kv2
    endif
    kchg3 changed kv3
    if (kchg3 == 1) then
        midiout 176, kchan1, 34, kv3
    endif
    kchg4 changed kv4
    if (kchg4 == 1) then
        midiout 176, kchan1, 35, kv4
    endif
    kchg5 changed kv5
    if (kchg5 == 1) then
        midiout 176, kchan1, 36, kv5
    endif
    kchg6 changed kv6
    if (kchg6 == 1) then
        midiout 176, kchan1, 37, kv6
    endif
    kchg7 changed kv7
    if (kchg7 == 1) then
        midiout 176, kchan1, 38, kv7
    endif
    kchg8 changed kv8
    if (kchg8 == 1) then
        midiout 176, kchan1, 39, kv8
    endif
endin

instr 34
    kmode       chnget "W_mode"
    ; --- Base values from sliders ---
    kfreqb      chnget "W_freq"
    kstartb     chnget "W_start"
    kendb       chnget "W_end"
    kampb_dB    chnget "W_Amp"
    ktoneb      chnget "W_tone"
    kbriteb     chnget "W_brite"
    kgainb      chnget "W_gain"
    kcenterb    chnget "W_center"
    klpfqb      chnget "W_lopass"

    ; --- Shared mod router temps ---
    ks   init 0
    kmin init 0
    kmax init 1
    kexp init 1
    ko   init 0
    kg   init 0
    ka   init 1
    km   init 1
    kc   init 0
    klp  init 20

    ks chnget "m_w_freq_src"
    kmin chnget "m_w_freq_min"
    kmax chnget "m_w_freq_max"
    kexp chnget "m_w_freq_exp"
    ko chnget "m_w_freq_offs"
    kg chnget "m_w_freq_gain"
    ka chnget "m_w_freq_area"
    km chnget "m_w_freq_mode"
    kc chnget "m_w_freq_cal"
    klp chnget "m_w_freq_lp"
    kfreq ApplyMod "w_freq", kfreqb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_start_src"
    kmin chnget "m_w_start_min"
    kmax chnget "m_w_start_max"
    kexp chnget "m_w_start_exp"
    ko chnget "m_w_start_offs"
    kg chnget "m_w_start_gain"
    ka chnget "m_w_start_area"
    km chnget "m_w_start_mode"
    kc chnget "m_w_start_cal"
    klp chnget "m_w_start_lp"
    kstart ApplyMod "w_start", kstartb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_end_src"
    kmin chnget "m_w_end_min"
    kmax chnget "m_w_end_max"
    kexp chnget "m_w_end_exp"
    ko chnget "m_w_end_offs"
    kg chnget "m_w_end_gain"
    ka chnget "m_w_end_area"
    km chnget "m_w_end_mode"
    kc chnget "m_w_end_cal"
    klp chnget "m_w_end_lp"
    kend ApplyMod "w_end", kendb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_amp_src"
    kmin chnget "m_w_amp_min"
    kmax chnget "m_w_amp_max"
    kexp chnget "m_w_amp_exp"
    ko chnget "m_w_amp_offs"
    kg chnget "m_w_amp_gain"
    ka chnget "m_w_amp_area"
    km chnget "m_w_amp_mode"
    kc chnget "m_w_amp_cal"
    klp chnget "m_w_amp_lp"
    kamp_dB ApplyMod "w_amp", kampb_dB, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_tone_src"
    kmin chnget "m_w_tone_min"
    kmax chnget "m_w_tone_max"
    kexp chnget "m_w_tone_exp"
    ko chnget "m_w_tone_offs"
    kg chnget "m_w_tone_gain"
    ka chnget "m_w_tone_area"
    km chnget "m_w_tone_mode"
    kc chnget "m_w_tone_cal"
    klp chnget "m_w_tone_lp"
    ktone ApplyMod "w_tone", ktoneb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_brite_src"
    kmin chnget "m_w_brite_min"
    kmax chnget "m_w_brite_max"
    kexp chnget "m_w_brite_exp"
    ko chnget "m_w_brite_offs"
    kg chnget "m_w_brite_gain"
    ka chnget "m_w_brite_area"
    km chnget "m_w_brite_mode"
    kc chnget "m_w_brite_cal"
    klp chnget "m_w_brite_lp"
    kbrite ApplyMod "w_brite", kbriteb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_gain_src"
    kmin chnget "m_w_gain_min"
    kmax chnget "m_w_gain_max"
    kexp chnget "m_w_gain_exp"
    ko chnget "m_w_gain_offs"
    kg chnget "m_w_gain_gain"
    ka chnget "m_w_gain_area"
    km chnget "m_w_gain_mode"
    kc chnget "m_w_gain_cal"
    klp chnget "m_w_gain_lp"
    kgain ApplyMod "w_gain", kgainb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_center_src"
    kmin chnget "m_w_center_min"
    kmax chnget "m_w_center_max"
    kexp chnget "m_w_center_exp"
    ko chnget "m_w_center_offs"
    kg chnget "m_w_center_gain"
    ka chnget "m_w_center_area"
    km chnget "m_w_center_mode"
    kc chnget "m_w_center_cal"
    klp chnget "m_w_center_lp"
    kcenter ApplyMod "w_center", kcenterb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    ks chnget "m_w_lp_src"
    kmin chnget "m_w_lp_min"
    kmax chnget "m_w_lp_max"
    kexp chnget "m_w_lp_exp"
    ko chnget "m_w_lp_offs"
    kg chnget "m_w_lp_gain"
    ka chnget "m_w_lp_area"
    km chnget "m_w_lp_mode"
    kc chnget "m_w_lp_cal"
    klp chnget "m_w_lp_lp"
    klpfq ApplyMod "w_lp", klpfqb, ks, ka, kmin, kmax, kexp, ko, kg, km, kc, klp

    kamp = ampdbfs(kamp_dB)

    ; --- Osc mode: phasor table reader + lpf18 ---
    kspan = max(1, kend - kstart)
    aphase phasor kfreq
    apos = (kstart + aphase * kspan) / 512
    aosc tablei apos, giSlitWave, 1
    aosc = (aosc * 2.0) - 1.0
    aosc dcblock aosc
    aosc lpf18 aosc, klpfq, 0.5, 0.3

    ; --- Shape mode: hsboscil -> slit table shaper + lpf18 ---
    if changed(kfreq) > 0 then
        reinit generator
    endif

generator:
    ioctfn ftgentmp 0, 0, 1024, -19, 1, 0.5, 270, 0.5
    ibasfreq = i(kfreq)
    rireturn

    a1L hsboscil kgain, 0.5, kbrite, ibasfreq, giSine, ioctfn
    a1R hsboscil kgain, ktone, kbrite, ibasfreq, giSine, ioctfn
    ashape_idx_L = ((a1L + kcenter) * 0.5) + 0.5
    ashape_idx_R = ((a1R + kcenter) * 0.5) + 0.5
    ashape_idx_L = limit(ashape_idx_L, 0, 1)
    ashape_idx_R = limit(ashape_idx_R, 0, 1)
    ashapeL tablei ashape_idx_L, giSlitWave, 1
    ashapeR tablei ashape_idx_R, giSlitWave, 1
    ashapeL = (ashapeL * 2.0) - 1.0
    ashapeR = (ashapeR * 2.0) - 1.0
    ashapeL dcblock ashapeL
    ashapeR dcblock ashapeR
    ashapeL lpf18 ashapeL, klpfq, 0.5, 0.3
    ashapeR lpf18 ashapeR, klpfq, 0.5, 0.3

    if (kmode < 1.5) then
        outs aosc * kamp, aosc * kamp
    else
        outs ashapeL * kamp, ashapeR * kamp
    endif
endin

</CsInstruments>
<CsScore>
i1 0 z
i10 0 z
i33 0 z
</CsScore>
</CsoundSynthesizer>













