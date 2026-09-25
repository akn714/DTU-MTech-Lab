### Image Noise
```
        INPUT IMAGE
             │
             ▼
     What is the problem?
             │
    ┌────────┼─────────┐
    ▼        ▼         ▼
 Low       Noise     Periodic
contrast             stripes
    │        │         │
    ▼        ▼         ▼
 HE/CLAHE  Identify   FFT
           noise       │
    │        │         ▼
    │   ┌────┴────┐  Notch
    │   ▼         ▼    │
    │ Median   Gaussian│
    │   │         │    │
    └───┴─────────┴────┘
             │
             ▼
        Enhanced Image
```

```
                         IMAGE GIVEN
                             │
                             ▼
                  ┌─────────────────────┐
                  │ WHAT IS WRONG?      │
                  └─────────────────────┘
                             │
       ┌─────────────┬───────┼────────┬─────────────┐
       ▼             ▼       ▼        ▼             ▼
   Brightness/     Noise   Blur/     Periodic     Need
    Contrast                Edges     Pattern     Multi-scale
       │             │       │          │             │
       ▼             ▼       ▼          ▼             ▼
   Histogram       What     What       FFT /       Pyramid
   Analysis        noise?   needed?    Frequency
       │             │       │          │
   ┌───┴───┐     ┌──┼──┐   ┌┴──┐    ┌──┴──┐
   ▼       ▼     ▼  ▼  ▼   ▼   ▼    ▼     ▼
 Dark/   Uneven  S&P Gaussian Blur Sharpen Periodic
 Low     light   noise noise   │      │     noise
 contrast   │      │     │     │      │       │
   │        ▼      ▼     ▼     ▼      ▼       ▼
   ▼       CLAHE  Median Gaussian Box  Sharpen FFT
   │                         │              │
   ▼                         ▼              ▼
 HE / CLAHE              Bilateral       Notch
```

### Image Degradation
```
                 NOISY IMAGE
                      ↓
             Is there obvious blur?
                      ↓
              Model degradation
                      ↓
           Look at a FLAT REGION
                      ↓
                  Histogram
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
     Bell curve     0/255       Signal-dependent
        ↓             ↓             ↓
     Gaussian      Impulse     Poisson/Speckle
        ↓             ↓             ↓
 Gaussian/         Median       Identify
 Bilateral                       multiplicative?
                                    ↓
                                  Speckle
```