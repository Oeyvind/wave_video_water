<CsoundSynthesizer>
<CsOptions>
-n -d
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 1
0dbfs = 1

gih OSCinit 8022
gkData[] init 8

instr 1
  kans OSClisten gih, "/arr/test", "G", gkData
  if (kans == 1) then
    printks "recv G packet first=%f last=%f\n", 0, gkData[0], gkData[7]
  endif
endin
</CsInstruments>
<CsScore>
i1 0 2
</CsScore>
</CsoundSynthesizer>
