<Cabbage>
form caption("Circular Water Wave Oscillator") size(420, 330), guiMode("queue"), pluginId("wOsc"), colour(30,20,20)
label bounds(15, 12, 390, 24), text("Circular surface waveform"), fontSize(18), fontColour(235, 235, 235)
rslider bounds(80, 55, 90, 90), channel("amplitude"), text("Master volume"), range(0, 1, 0.5, 1, 0.001), trackerColour(40, 200, 120)
rslider bounds(210, 55, 90, 90), channel("lowpass"), text("Lowpass"), range(20, 20000, 12000, 0.3, 1), trackerColour(220, 170, 50)
csoundoutput bounds(0, 180, 420, 150), channel("csoundoutput")
</Cabbage>
<CsOptions>
-n -d -M0 -+rtmidi=NULL
</CsOptions>
<CsInstruments>
sr = 48000
ksmps = 32
nchnls = 2
0dbfs = 1

; Route incoming MIDI notes to the wavetable oscillator.
massign 0, 2

giWaveSize = 512
gihOsc OSCinit 8101
giWave ftgen 1, 0, giWaveSize, 10, 1
giSeen ftgen 2, 0, 32, -2, 0
gkFrame init -1
gkReceived init 0

instr 1
    ktrig init 0
    kframe init 0
    kchunk init 0
    kcount init 0
    k00 init 0
    k01 init 0
    k02 init 0
    k03 init 0
    k04 init 0
    k05 init 0
    k06 init 0
    k07 init 0
    k08 init 0
    k09 init 0
    k10 init 0
    k11 init 0
    k12 init 0
    k13 init 0
    k14 init 0
    k15 init 0
    kstatusTrig init 0

read_chunk:
    ktrig OSClisten gihOsc, "/wave/circle/chunk", "iiiffffffffffffffff", kframe, kchunk, kcount, k00, k01, k02, k03, k04, k05, k06, k07, k08, k09, k10, k11, k12, k13, k14, k15
    if ktrig == 1 then
        if kcount == 32 then
            if kframe != gkFrame then
                gkFrame = kframe
                gkReceived = 0
                kidx init 0
                while kidx < 32 do
                    tablew 0, kidx, giSeen
                    kidx += 1
                od
            endif
            if kchunk >= 0 && kchunk < 32 then
                kseen table kchunk, giSeen
                if kseen < 0.5 then
                    tablew 1, kchunk, giSeen
                    gkReceived += 1
                endif
                kbase = kchunk * 16
                tablew k00, kbase + 0, giWave
                tablew k01, kbase + 1, giWave
                tablew k02, kbase + 2, giWave
                tablew k03, kbase + 3, giWave
                tablew k04, kbase + 4, giWave
                tablew k05, kbase + 5, giWave
                tablew k06, kbase + 6, giWave
                tablew k07, kbase + 7, giWave
                tablew k08, kbase + 8, giWave
                tablew k09, kbase + 9, giWave
                tablew k10, kbase + 10, giWave
                tablew k11, kbase + 11, giWave
                tablew k12, kbase + 12, giWave
                tablew k13, kbase + 13, giWave
                tablew k14, kbase + 14, giWave
                tablew k15, kbase + 15, giWave
            endif
        endif
        kgoto read_chunk
    endif
    kstatusTrig = changed(gkReceived)
    if gkReceived >= 32 then
      puts sprintfk("Waveform complete for frame %d", gkFrame), kstatusTrig
    endif
endin

instr 2
    ifreq cpsmidi
    print ifreq
    iamp ampmidi 1
    kamp chnget "amplitude"
    klpf chnget "lowpass"
    aphase phasor ifreq
    awave tablei aphase, giWave, 1
    awave = (awave * 2) - 1
    awave dcblock awave
    awave lpf18 awave, klpf, 0.5, 0.3
    outs awave * iamp * kamp, awave * iamp * kamp
endin
</CsInstruments>
<CsScore>
i1 0 z
</CsScore>
</CsoundSynthesizer>