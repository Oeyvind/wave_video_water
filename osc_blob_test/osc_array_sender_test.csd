<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 1
0dbfs = 1

instr 1
  ktrig metro 5
  kcount init 0
  iport = 8021
  karr[] init 8
  karr[] fillarray 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0

  if (ktrig == 1) then
    OSCsend ktrig, "127.0.0.1", iport, "/arr/test", "A", karr
    kcount += 1
    printks "sent A-array packet %d\\n", 0, kcount
  endif
endin
</CsInstruments>
<CsScore>
i 1 0 3
</CsScore>
</CsoundSynthesizer>
