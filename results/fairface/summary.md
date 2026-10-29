# Results summary: fairface

Images: 2100 | detectors: Deep-Fake-Detector-v2-Model | threshold set for pooled 5% FPR on unfiltered images

## Deep-Fake-Detector-v2-Model (threshold 0.745)
- **lighten** (strength 0 -> 1.0): very_light: 3.6% -> 3.6%; light: 2.5% -> 3.3%; intermediate: 3.1% -> 6.2%; tan: 4.9% -> 6.7%; brown: 4.2% -> 4.5%; dark: 6.7% -> 2.7%
- **smooth** (strength 0 -> 1.0): very_light: 3.6% -> 7.2%; light: 2.5% -> 5.0%; intermediate: 3.1% -> 10.0%; tan: 4.9% -> 12.9%; brown: 4.2% -> 16.5%; dark: 6.7% -> 23.6%
- **smooth_lighten** (strength 0 -> 1.0): very_light: 3.6% -> 4.8%; light: 2.5% -> 5.0%; intermediate: 3.1% -> 7.7%; tan: 4.9% -> 11.1%; brown: 4.2% -> 8.6%; dark: 6.7% -> 11.1%
- Interaction (lighten): OR=1.74 per SD of ITA, p=1.52e-06 -> filter effect hits lighter skin harder
- Interaction (smooth): OR=0.84 per SD of ITA, p=0.0997 -> filter effect hits darker skin harder
- Interaction (smooth_lighten): OR=1.15 per SD of ITA, p=0.249 -> filter effect hits lighter skin harder

## Label shift at max strength
- lighten: 87.4% of faces change skin-tone band
- smooth: 3.0% of faces change skin-tone band
- smooth_lighten: 87.3% of faces change skin-tone band

Bands with fewer than 20 images are hidden. Check CIs in fpr_by_band.csv before claiming a gap.