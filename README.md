# Awesome Speaker-Diarization [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Some comprehensive papers about speaker diarization (SD).
If you discover any unnoticed documents, please open issues or pull requests (recommended).

## Overview

- **DIHARD Keynote Session:** The yellow brick road of diarization, challenges and other neural paths **[[Slides]](https://dihardchallenge.github.io/dihard3workshop/slide/The%20yellow%20brick%20road%20of%20diarization,%20challenges%20and%20other%20neural%20paths.pdf)**  **[[Video]](https://www.youtube.com/watch?v=_usbos-SJlg&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=10)**

## Reviews

- “A review of speaker diarization: Recent advances with deep learning”, in *Computer Speech & Language, Volume 72,* 2023. (USC) [[paper](https://arxiv.org/abs/2101.09624)]

## EEND-based
- **SA-EEND (1)**: “End-to-End Neural Speaker Diarization with Self-attention”, in *Proc. ASRU, 2019*. (Hitachi) [[Paper](https://ieeexplore.ieee.org/abstract/document/9003959)] [[Github](https://github.com/hitachi-speech/EEND)] [[Pytorch](https://github.com/Xflick/EEND_PyTorch)] [[Review](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)]
- **SA-EEND (2)**: “End-to-End Neural Diarization: Reformulating Speaker Diarization as Simple Multi-label Classification”, in *arXiv:2003.02966,* 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2003.02966)] [[Review](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)]
- **EEND-EDA (1)**: “End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors”, in *Proc. Interspeech,* 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2005.09921)] [[Review](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers)]
- **EEND-EDA (2)**: “Encoder-Decoder Based Attractor Calculation for End-to-End Neural Diarization”, in *IEEE/ACM TASLP,* 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2106.10654)] [[Review](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers)]
- **RX-EEND**: “Auxiliary Loss of Transformer with Residual Connection for End-to-End Speaker Diarization”, in *Proc. ICASSP,* 2022. (GIST) [[Paper](https://arxiv.org/abs/2110.07116)] [[Review](https://velog.io/@fbdp1202/RX-EEND-%EB%A6%AC%EB%B7%B0-Auxiliary-Loss-of-Transformer-with-Residual-connection-For-End-to-end-Speaker-Diarization)]
- **EEND-SS**: "Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers”, in *Proc. SLT,* 2022. (CMU) [[Paper](https://arxiv.org/abs/2203.17068)]
- **EEND-NA**: “Neural Diarization with Non-Autoregressive Intermediate Attractors”, in *Proc. ICASSP,* 2023. (LINE)  [[Paper](https://arxiv.org/abs/2303.06806)]
- **EEND-EDA-SpkAtt**: Towrards End-to-end Speaker Diarzation in the Wild, in *Proc. ICASSP,* 2023. [[Paper](https://arxiv.org/abs/2211.01299)]

## Clustering-based
- “Analysis of Speaker Diarization Based on Bayesian HMM With Eigenvoice Priors”, *IEEE/ACM TASLP,* 2019. (BUT*)* [[Paper](https://ieeexplore.ieee.org/document/8910412)]
- **NME-SC**: “Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap”, *IEEE SPL,* 2019. [[Paper](https://arxiv.org/abs/2003.02405)] [[Github](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)]
- “Overlap-Aware Diarization: Resegmentation Using Neural End-to-End Overlapped Speech Detection” in *Proc. ICASSP*, 2020. [[Paper](https://ieeexplore.ieee.org/document/9053096)]
- “Similarity Measurement of Segment-Level Speaker Embeddings in Speaker Diarization”, *IEEE/ACM TASLP,* 2023. (DKU) [[Paper](https://ieeexplore.ieee.org/document/9849033)]

## TS-VAD
- **EDA-TS-VAD**: “Target Speaker Voice Activity Detection with Transformers and Its Integration with End-to-End Neural Diarization”, in **************Proc. ICASSP,************** 2023. (Microsoft) [[paper](https://arxiv.org/abs/2210.16127)]
- **Seq2Seq-TS-VAD**: “Target-Speaker Voice Activity Detection via Sequence-to-Sequence Prediction”, in *Proc. ICASSP,* 2023. (DKU) [[paper](https://arxiv.org/abs/2210.16127)]
- **AED-EEND**: “Attention-based Encoder-Decoder Network for End-to-End Neural Speaker Diarization with Target Speaker Attractor”, in *Proc. Interspeech,* 2023. (SJTU) [[paper](https://arxiv.org/abs/2305.10704)]

## Simulated Dataset

- **Concat and Sum Approach**: “End-to-end neuarl speaker diarization with permuation-free objectives”, in *Proc. Interspeech,* 2019. [Paper]
- **BUT Alogrithm:** “From simulated mixtures to simulated conversations as training data for end-to-end neural diarization” , in *Proc. Interspeech* 2022. (BUT) [Paper]
- **Hitach Algorithm**: “Improving the naturalness of simulated conversations for end-to-end neural diarization”, in *Proc. Odyssey,* 2022. (Hitachi) **[Paper]
- **EEND-EDA-SpkAtt:** Towrards End-to-end Speaker Diarzation in the Wild, in *Proc. ICASSP* 2023. [[Paper](https://arxiv.org/abs/2211.01299)]

## Self-suprvised
- “Self-supervised Speaker Diarization”, in *Proc. Interspeech,* 2022. [[Paper](https://arxiv.org/abs/2204.04166)]

## Measurement
- **BER:** “Balanced Error Rate For Speaker Diarization”, in *Proc. arXiv:2211.04304,* 2022 [[Paper](https://arxiv.org/abs/2211.04304)] [[Code](https://github.com/X-LANCE/BER)]

## Challenge

### 2022 VoxSRC Track4

[[Paper](https://arxiv.org/abs/2302.10248)] [[Workshop](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/interspeech2022.html)]

- 1st: **DKU-DukeECE [[Tech Report](https://arxiv.org/abs/2210.01677)] [[slides]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/DKU-DukeECE_slides.pdf) [[Videos](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/DKU-DukeECE_video.mp4)]**
- 2nd: **KristonAI [[Tech Report](https://arxiv.org/abs/2209.11433)] [[slides]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/voxsrc2022_kristonai_track4.pdf) [[Videos](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/voxsrc2022_kristonai_track4.mp4)]**
- 3rd: **GIST**-**AiTeR [[Tech Report](https://arxiv.org/abs/2209.10357)] [[slides]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/gist_slides.pdf) [[Videos](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/video_aiter.mp4)]**

### 2021 VosSRC Track4

[[Workshop](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/interspeech2021.html)]

- 1st: **DKU-DukeECE-Lenovo [[Tech Report]](https://arxiv.org/abs/2109.02002) [[Videos]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/DKU-DukeECE-Lenovo.mp4)**
- 2nd: **Bytedance [[Tech Report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/ByteDance_diarization.pdf)] [[Videos]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/Bytedance_SAMI.mp4)**
- 3rd: **Tencent  [[Tech Report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/Tencent_diarization.pdf)]**
