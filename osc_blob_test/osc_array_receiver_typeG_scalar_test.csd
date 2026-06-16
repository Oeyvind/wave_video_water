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

instr 1
  ktab init 0
  kans OSClisten gih, "/arr/test", "G", ktab
  if (kans == 1) then
    printks "recv G scalar=%f\n", 0, ktab
  endif
endin
</CsInstruments>
<CsScore>
i1 0 2
</CsScore>
</CsoundSynthesizer>
