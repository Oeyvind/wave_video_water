<CsoundSynthesizer>
<CsOptions>
-n -d
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 1
0dbfs = 1

giTab ftgen 1, 0, 8, -2, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0

instr 1
  ktrig metro 2
  if (ktrig == 1) then
    OSCsend ktrig, "127.0.0.1", 8022, "/arr/test", "G", giTab
    printks "sent G packet\n", 0
  endif
endin
</CsInstruments>
<CsScore>
i1 0 1
</CsScore>
</CsoundSynthesizer>
