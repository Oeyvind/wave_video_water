<CsoundSynthesizer>
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 1
0dbfs = 1

gihIn OSCinit 8031

gkCurrentFrame init -1
gkChunkCount init 32
gkReceivedCount init 0
gkAckTrig init 0

gkSlit[] init 512
gkChunkSeen[] init 32

instr 1
  kframe init 0
  kchunk init 0
  kcount init 0
  kf00 init 0
  kf01 init 0
  kf02 init 0
  kf03 init 0
  kf04 init 0
  kf05 init 0
  kf06 init 0
  kf07 init 0
  kf08 init 0
  kf09 init 0
  kf10 init 0
  kf11 init 0
  kf12 init 0
  kf13 init 0
  kf14 init 0
  kf15 init 0

nextmsg:
  kans OSClisten gihIn, "/slit/chunk", "iiiffffffffffffffff", \
       kframe, kchunk, kcount, \
       kf00, kf01, kf02, kf03, kf04, kf05, kf06, kf07, \
       kf08, kf09, kf10, kf11, kf12, kf13, kf14, kf15
  if (kans == 0) goto done

  if (kcount <= 0) then
    kgoto nextmsg
  endif

  if (kframe != gkCurrentFrame || kcount != gkChunkCount) then
    gkCurrentFrame = kframe
    gkChunkCount = kcount
    gkReceivedCount = 0

    kclr = 0
    while (kclr < 32) do
      gkChunkSeen[kclr] = 0
      kclr += 1
    od
  endif

  if (kchunk < 0 || kchunk >= gkChunkCount || kchunk >= 32) then
    kgoto nextmsg
  endif

  if (gkChunkSeen[kchunk] < 0.5) then
    gkChunkSeen[kchunk] = 1
    gkReceivedCount += 1
  endif

  kbase = kchunk * 16
  if (kbase + 15 < 512) then
    gkSlit[kbase + 0] = kf00
    gkSlit[kbase + 1] = kf01
    gkSlit[kbase + 2] = kf02
    gkSlit[kbase + 3] = kf03
    gkSlit[kbase + 4] = kf04
    gkSlit[kbase + 5] = kf05
    gkSlit[kbase + 6] = kf06
    gkSlit[kbase + 7] = kf07
    gkSlit[kbase + 8] = kf08
    gkSlit[kbase + 9] = kf09
    gkSlit[kbase + 10] = kf10
    gkSlit[kbase + 11] = kf11
    gkSlit[kbase + 12] = kf12
    gkSlit[kbase + 13] = kf13
    gkSlit[kbase + 14] = kf14
    gkSlit[kbase + 15] = kf15
  endif

  if (gkReceivedCount >= gkChunkCount) then
    kmaxerr = 0
    ksum = 0
    kidx = 0
    while (kidx < 512) do
      kexp = ((kidx + (gkCurrentFrame * 7)) % 512) / 511.0
      kval = gkSlit[kidx]
      ksum += kval
      kerr = abs(kval - kexp)
      if (kerr > kmaxerr) then
        kmaxerr = kerr
      endif
      kidx += 1
    od

    gkAckTrig += 1
    OSCsend gkAckTrig, "127.0.0.1", 8032, "/slit/ack", "iif", gkCurrentFrame, gkReceivedCount, kmaxerr

    if (kmaxerr <= 0.00001) then
      printks "frame %d PASS chunks=%d maxerr=%f sum=%f\\n", 0, gkCurrentFrame, gkReceivedCount, kmaxerr, ksum
    else
      printks "frame %d FAIL chunks=%d maxerr=%f sum=%f\\n", 0, gkCurrentFrame, gkReceivedCount, kmaxerr, ksum
    endif

    gkReceivedCount = 0
    kclr2 = 0
    while (kclr2 < 32) do
      gkChunkSeen[kclr2] = 0
      kclr2 += 1
    od
  endif

  kgoto nextmsg

done:
endin
</CsInstruments>
<CsScore>
i 1 0 120
</CsScore>
</CsoundSynthesizer>
