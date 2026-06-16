<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 1
0dbfs = 1

gihandle OSCinit 8021

instr 1
  kdata[] init 8
  kexpected[] fillarray 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0

nextmsg:
  kans, kdata[] OSClisten gihandle, "/arr/test", "A"
  if (kans == 0) goto done

  klen lenarray kdata
  if (klen != 8) then
    printks "OSClisten A-array FAIL: expected len=8 got len=%d\\n", 0, klen
    kgoto nextmsg
  endif

  kmaxerr = 0.0
  ksum = 0.0
  kidx = 0
  while (kidx < 8) do
    kerr = abs(kdata[kidx] - kexpected[kidx])
    if (kerr > kmaxerr) then
      kmaxerr = kerr
    endif
    ksum += kdata[kidx]
    kidx += 1
  od

  if (kmaxerr <= 0.00001) then
    printks "OSClisten array integrity PASS: maxerr=%f sum=%f\\n", 0, kmaxerr, ksum
  else
    printks "OSClisten array integrity FAIL: maxerr=%f sum=%f\\n", 0, kmaxerr, ksum
    printks "  recv: [%f, %f, %f, %f, %f, %f, %f, %f]\\n", 0, kdata[0], kdata[1], kdata[2], kdata[3], kdata[4], kdata[5], kdata[6], kdata[7]
  endif

  kgoto nextmsg

done:
endin
</CsInstruments>
<CsScore>
i 1 0 20
</CsScore>
</CsoundSynthesizer>
