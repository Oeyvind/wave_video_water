<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>
ksmps = 64
nchnls = 1
0dbfs = 1

giPort = 8011

instr 1
  ; Keep this larger than expected message component count.
  Smess[] init 64

nextmsg:
  Smess, klen OSCraw giPort
  if (klen <= 0) goto done

  printks "\nOSCraw received: items=%d\n", 0, klen

  if (klen >= 1) then
    Saddr = Smess[0]
    kaddrlen strlenk Saddr
    printks "  [0] addr(len=%d): %s\n", 0, kaddrlen, Saddr
  endif
  if (klen >= 2) then
    Stypes = Smess[1]
    ktypeslen strlenk Stypes
    printks "  [1] types(len=%d): %s\n", 0, ktypeslen, Stypes
  endif

  ; If a blob is present, OSCraw places it as a string item.
  ; strlenk helps us see whether binary zeros truncate string content.
  if (klen >= 3) then
    Sblob = Smess[2]
    kbloblen strlenk Sblob
    printks "  [2] blob-as-string length=%d\n", 0, kbloblen
  endif

  ; Print all remaining items for inspection.
  kndx = 2
  while (kndx < klen) do
    Sitem = Smess[kndx]
    kitemlen strlenk Sitem
    printks "  [%d] len=%d data=%s\n", 0, kndx, kitemlen, Sitem
    kndx += 1
  od

  kgoto nextmsg

done:
endin
</CsInstruments>
<CsScore>
i 1 0 60
</CsScore>
</CsoundSynthesizer>
