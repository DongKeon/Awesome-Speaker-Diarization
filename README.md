# Awesome-Speaker-Diarization [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Some comprehensive papers about speaker diarization (SD).

If you discover any unnoticed documents, please open issues or pull requests (recommended).

## Overview

- **DIHARD Keynote Session:** The yellow brick road of diarization, challenges and other neural paths [[Slides]](https://dihardchallenge.github.io/dihard3workshop/slide/The%20yellow%20brick%20road%20of%20diarization,%20challenges%20and%20other%20neural%20paths.pdf) [[Video]](https://www.youtube.com/watch?v=_usbos-SJlg&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=10)

## Reviews

- “A review of speaker diarization: Recent advances with deep learning”, in *Computer Speech & Language, Volume 72,* 2023. (USC) [[Paper](https://arxiv.org/abs/2101.09624)]

## EEND (End-to-End Neural Diarization)-based
- **BLSTM-EEND**: "End-to-End Neural Speaker Diarization with Permutation-Free Objectives", in *Proc. Interspeech*, 2019. (Hitachi) [[Paper](https://arxiv.org/abs/1909.05952)]
- **SA-EEND (1)**: “End-to-End Neural Speaker Diarization with Self-attention”, in *Proc. ASRU*, 2019. (Hitachi) [[Paper](https://ieeexplore.ieee.org/abstract/document/9003959)] [[Code](https://github.com/hitachi-speech/EEND)] [[Pytorch](https://github.com/Xflick/EEND_PyTorch)] [[Review](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)]
- **SA-EEND (2)**: “End-to-End Neural Diarization: Reformulating Speaker Diarization as Simple Multi-label Classification”, in *arXiv:2003.02966,* 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2003.02966)] [[Review](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)]
- **SC-EEND**: "Neural Speaker Diarization with Speaker-Wise Chain Rule", in *Proc. Interspeech*, 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2006.01796)] [[Review](https://velog.io/@fbdp1202/SC-EEND-%EB%A6%AC%EB%B7%B0-Neural-Speaker-Diarization-with-Speaker-Wise-Chain-Rule)]
- **EEND-EDA (1)**: “End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors”, in *Proc. Interspeech,* 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2005.09921)] [[Review](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers)] [[Code](https://github.com/butspeechfit/eend)]
- **EEND-EDA (2)**: “Encoder-Decoder Based Attractor Calculation for End-to-End Neural Diarization”, in *IEEE/ACM TASLP,* 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2106.10654)] [[Review](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers)] [[Code](https://github.com/butspeechfit/eend)]
- **CB-EEND**: "End-to-end Neural Diarization: From Transformer to Conformer", in *Proc. Interspeech*, 2021. (Amazon) [[Paper](https://arxiv.org/abs/2106.07167)] [[Review](https://velog.io/@fbdp1202/CB-EEND-%EB%A6%AC%EB%B7%B0-End-to-end-Neural-Diarization-From-Transformer-to-Conformer)]
- **TDCN-SA**: "End-to-End Diarization for Variable Number of Speakers with Local-Global Networks and Discriminative Speaker Embeddings", in *Proc. ICASSP*, 2021. (Google) [[Paper](https://arxiv.org/abs/2105.02096)] [[Review](https://velog.io/@fbdp1202/TDCN-SA-%EB%A6%AC%EB%B7%B0-End-to-End-Diarization-for-Variable-Number-of-Speakers-with-Local-Global-Networks-and-Discriminative-Speaker-Embeddings)]
- **EEND-VC (1)**: "Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds", in *Proc. ICASSP*, 2021. (NTT) [[Paper](https://arxiv.org/abs/2010.13366)] [[Review](https://velog.io/@fbdp1202/EEND-vector-clustering-%EB%A6%AC%EB%B7%B0-Integrating-end-to-end-neural-and-clustering-based-diarization-Getting-the-best-of-both-world)] [[Code](https://github.com/nttcslab-sp/EEND-vector-clustering)]
- **EEND-VC (2)**: "Advances in integration of end-to-end neural and clustering-based diarization for real conversational speech", in *Proc. Interspeech*, 2021. (NTT) [[Paper](https://arxiv.org/abs/2105.09040)] [[Review](https://velog.io/@fbdp1202/EEND-vector-clustering-%EB%A6%AC%EB%B7%B0-Integrating-end-to-end-neural-and-clustering-based-diarization-Getting-the-best-of-both-world)] [[Code](https://github.com/nttcslab-sp/EEND-vector-clustering)]
- **EEND-GLA**: "Towards Neural Diarization for Unlimited Numbers of Speakers Using Global and Local Attractors", in *Proc. ASRU*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2107.01545)] [[Reivew](https://velog.io/@fbdp1202/EEND-EDA-Clustering-%EB%A6%AC%EB%B7%B0-Towards-Neural-Diarization-for-Unlimited-Numbers-of-Speakers-using-Global-and-Local-Attractors)] 
- **DIVE**: "DIVE: End-to-end Speech Diarization via Iterative Speaker Embedding", in *Proc. ICASSP*, 2022. (Google) [[Paper](https://arxiv.org/abs/2105.13802)]
- **RX-EEND**: “Auxiliary Loss of Transformer with Residual Connection for End-to-End Speaker Diarization”, in *Proc. ICASSP,* 2022. (GIST) [[Paper](https://arxiv.org/abs/2110.07116)] [[Review](https://velog.io/@fbdp1202/RX-EEND-%EB%A6%AC%EB%B7%B0-Auxiliary-Loss-of-Transformer-with-Residual-connection-For-End-to-end-Speaker-Diarization)]
- **EEND-VC-iGMM**: "Tight integration of neural- and clustering-based diarization through deep unfolding of infinite Gaussian mixture model", in *Proc. ICASSP*, 2022. (NTT) [[Paper](https://arxiv.org/abs/2202.06524)]
- **EDA-RC**: "Robust End-to-end Speaker Diarization with Generic Neural Clustering", in *Proc. Interspeech*, 2022. (SJTU) [[Paper](https://arxiv.org/abs/2204.08164)]
- **EEND-NAA**: "End-to-End Neural Speaker Diarization with an Iterative Refinement of Non-Autoregressive Attention-based Attractors", in *Proc. Interspeech*, 2022. (JHU) [[Paper](https://www.isca-speech.org/archive/interspeech_2022/rybicka22_interspeech.html)]
- **Graph-PIT**: "Utterance-by-utterance overlap-aware neural diarization with Graph-PIT", in *Proc. Interspeech*, 2022. (NTT) [[Paper](https://arxiv.org/abs/2207.13888)] [[Code](https://github.com/fgnt/graph_pit)]
- **SOND**: "Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis", in *Proc. EMNLP*, 2022. (Alibaba) [[Paper](https://arxiv.org/abs/2211.10243)] [[Code](https://github.com/alibaba-damo-academy/FunASR)]
- "Improving Transformer-based End-to-End Speaker Diarization by Assigning Auxiliary Losses to Attention Heads", in *Proc. ICASSP,* 2023. (HU) [[Paper](https://arxiv.org/abs/2303.01192)]
- **EEND-NA**: “Neural Diarization with Non-Autoregressive Intermediate Attractors”, in *Proc. ICASSP,* 2023. (LINE)  [[Paper](https://arxiv.org/abs/2303.06806)]
- **EEND-EDA-SpkAtt**: "Towrards End-to-end Speaker Diarzation in the Wild", in *Proc. ICASSP,* 2023. [[Paper](https://arxiv.org/abs/2211.01299)]
- **TOLD**: "TOLD: A Novel Two-Stage Overlap-Aware Framework for Speaker Diarization", in *Proc. ICASSP*, 2023. (Alibaba) [[Paper](https://arxiv.org/abs/2303.05397)] [[Code](https://github.com/alibaba-damo-academy/FunASR)]

## Clustering-based
- **VB-HMM**: "Speaker Diarization based on Bayesian HMM with Eigenvoice Priors", in *Proc. Odyssey*, 2019. (BUT) [[Paper](https://www.isca-speech.org/archive/odyssey_2018/diez18_odyssey.html)]
- **VB-HMM**: “Analysis of Speaker Diarization Based on Bayesian HMM With Eigenvoice Priors”, *IEEE/ACM TASLP,* 2019. (BUT) [[Paper](https://ieeexplore.ieee.org/document/8910412)]
- **NME-SC**: “Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap”, *IEEE SPL,* 2019. [[Paper](https://arxiv.org/abs/2003.02405)] [[Code](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)]
- **Resegmentation with VB**: “Overlap-Aware Diarization: Resegmentation Using Neural End-to-End Overlapped Speech Detection”, in *Proc. ICASSP*, 2020. [[Paper](https://ieeexplore.ieee.org/document/9053096)]
- **Pyannote 2.0**: "End-to-end speaker segmentation for overlap-aware resegmentation", in *Proc. Interspeech*, 2021. (Pyannote) [[Paper](https://arxiv.org/abs/2104.04045)] [[Code](https://github.com/pyannote/pyannote-audio)]
- **UMAP-Leiden**: "Reformulating Speaker Diarization as Community Detection With Emphasis On Topological Structure", in *Proc. ICASSP*, 2022. (Alibaba) [[Paper](https://arxiv.org/abs/2204.12112)]
- **SCALE**: "Spectral Clustering-aware Learning of Embeddings for Speaker Diarisation", in *Proc. ICASSP*, 2023. (CAM) [[Paper](https://arxiv.org/abs/2210.13576)]
- **SHARC**: "Supervised Hierarchical Clustering using Graph Neural Networks for Speaker Diarization", in *Proc. ICASSP*, 2023. (IISC) [[Paper](https://arxiv.org/abs/2302.12716)]
- “Similarity Measurement of Segment-Level Speaker Embeddings in Speaker Diarization”, *IEEE/ACM TASLP,* 2023. (DKU) [[Paper](https://ieeexplore.ieee.org/document/9849033)]

## TS-VAD
- **TS-VAD**: "Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario", in *Proc. Interspeech*, 2020. [[Paper](https://arxiv.org/abs/2005.07272)]
- **MTFAD**: "Multi-target Filter and Detector for Unknown-number Speaker Diarization", in *IEEE SPL*, 2022. [[Paper](https://arxiv.org/abs/2203.16007)]
- **EDA-TS-VAD**: “Target Speaker Voice Activity Detection with Transformers and Its Integration with End-to-End Neural Diarization”, in *Proc. ICASSP*, 2023. (Microsoft) [[Paper](https://arxiv.org/abs/2210.16127)] [[Review](https://velog.io/@dongkeon/2023-EDA-TS-VAD)]
- **Seq2Seq-TS-VAD**: “Target-Speaker Voice Activity Detection via Sequence-to-Sequence Prediction”, in *Proc. ICASSP,* 2023. (DKU) [[Paper](https://arxiv.org/abs/2210.16127)] 
- **AED-EEND**: “Attention-based Encoder-Decoder Network for End-to-End Neural Speaker Diarization with Target Speaker Attractor”, in *Proc. Interspeech,* 2023. (SJTU) [[Paper](https://arxiv.org/abs/2305.10704)]

## Post-Processing
- **EENDasP**: "End-to-End Speaker Diarization as Post-Processing", in *Proc. ICASSP*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2012.10055)] [[Review](https://velog.io/@fbdp1202/EEND-asp-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-as-Post-Processing) [[Code](https://github.com/DongKeon/EENDasP)]
- **Dover-Lap**: "DOVER-Lap: A Method for Combining Overlap-aware Diarization Outputs", in *Proc. IEEE SLT*, 2021. (JHU) [[Paper](https://arxiv.org/abs/2011.01997)] [[Review](https://velog.io/@fbdp1202/Dover-lap-%EB%A6%AC%EB%B7%B0-A-method-for-combining-overlap-aware-diarization-outputs)] [[Code](https://github.com/desh2608/dover-lap)]
- **DiaCorrect**: DiaCorrect: End-to-end error correction for speaker diarization, in *Proc. arXiv:2210.17189*, 2022. [[Paper](https://arxiv.org/abs/2210.17189)]

## Embedding
- **MSDD**: "Multi-scale Speaker Diarization with Dynamic Scale Weighting", in *Proc. Interspeech*, 2022. (NVIDIA) [[Paper](https://arxiv.org/abs/2203.15974)] [[Code](https://github.com/NVIDIA/NeMo)] [[Blog](https://developer.nvidia.com/blog/dynamic-scale-weighting-through-multiscale-speaker-diarization/)]
- "In Search of Strong Embedding Extractors For Speaker Diarization", in *Proc. ICASSP*, 2023. (Naver) [[Paper](https://arxiv.org/abs/2210.14682)] [[Review](https://velog.io/@dongkeon/2023-In-Search-of-Strong-Embedding-Extractors-For-Speaker-Diarization-ICASSP)]
- **PRISM**: "PRISM: Pre-trained Indeterminate Speaker Representation Model for Speaker Diarization and Speaker Verification", in *Proc. Interspeech*, 2022. (Alibaba) [[Paper](https://arxiv.org/abs/2205.07450)]
- **HEE**: "High-resolution embedding extractor for speaker diarisation", in *Proc. ICASSP*, 2023. (Naver)  [[Paper](https://arxiv.org/abs/2211.04060)]

## Scoring
- **LSTM scoring**: "LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization", in *Proc. Interspeech*, 2019. [[Paper](https://arxiv.org/abs/1907.10393)]
- **GAT**: "Multi-scale speaker embedding-based graph attention networks for speaker diarisation", in *Proc. ICASSP*, 2022. (Naver) [[Paper](https://arxiv.org/abs/2110.03361)]

## Online
- **FS-EEND**: "Online Streaming End-to-End Neural Diarization Handling Overlapping Speech and Flexible Numbers of Speakers", in *Proc. Interspeech*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2101.08473)] [[Reivew](https://velog.io/@fbdp1202/FS-EEND-%EB%A6%AC%EB%B7%B0-Online-end-to-end-diarization-handling-overlapping-speech-and-flexible-numbers-of-speakers)] 
- "Absolute decision corrupts absolutely: conservative online speaker diarisation", in *Proc. ICASSP*, 2023. (Naver) [[Paper](https://arxiv.org/abs/2211.04768)]


## Simulated Dataset
- “End-to-end neuarl speaker diarization with permuation-free objectives”, in *Proc. Interspeech*, 2019. [[Paper](https://arxiv.org/abs/1909.05952)]
- “From simulated mixtures to simulated conversations as training data for end-to-end neural diarization” , in *Proc. Interspeech*, 2022. (BUT) [[Paper](https://arxiv.org/abs/2204.00890)] [[Code](https://github.com/BUTSpeechFIT/EEND_dataprep)]
- “Improving the naturalness of simulated conversations for end-to-end neural diarization”, in *Proc. Odyssey*, 2022. (Hitachi) [[Paper](https://arxiv.org/abs/2204.11232)]
- "Multi-Speaker and Wide-Band Simulated Conversations as Training Data for End-to-End Neural Diarization", in *Proc. ICASSP*, 2023. (BUT) [[Paper](https://arxiv.org/abs/2211.06750)] [[Code](https://github.com/BUTSpeechFIT/EEND_dataprep)]
- **EEND-EDA-SpkAtt:** Towrards End-to-end Speaker Diarzation in the Wild, in *Proc. ICASSP*, 2023. [[Paper](https://arxiv.org/abs/2211.01299)]

## Self-Suprvised
- “Self-supervised Speaker Diarization”, in *Proc. Interspeech,* 2022. [[Paper](https://arxiv.org/abs/2204.04166)]
- **CSDA**: "Continual Self-Supervised Domain Adaptation for End-to-End Speaker Diarization", in *Proc. IEEE SLT*, 2022. (Pyannote) [[Paper](https://ieeexplore.ieee.org/document/10023195)] [[Code](https://github.com/juanmc2005/CSDA)]

## Joint-Training
- **EEND-SS**: "Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers”, in *Proc. SLT,* 2022. (CMU) [[Paper](https://arxiv.org/abs/2203.17068)]

## Multi-Channel
- "Multi-Channel Speaker Diarization Using Spatial Features for Meetings", in *Proc. ICASSP*, 2022. (Tencent) [[Paper](https://ieeexplore.ieee.org/document/9747343)] 

## Measurement
- **BER:** “Balanced Error Rate For Speaker Diarization”, in *Proc. arXiv:2211.04304,* 2022 [[Paper](https://arxiv.org/abs/2211.04304)] [[Code](https://github.com/X-LANCE/BER)]

## Challenge

### 2022 VoxSRC Track4

[[Paper](https://arxiv.org/abs/2302.10248)] [[Workshop](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/interspeech2022.html)]

- 1st: **DKU-DukeECE** [[Tech Report](https://arxiv.org/abs/2210.01677)] [[slide]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/DKU-DukeECE_slides.pdf) [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/DKU-DukeECE_video.mp4)]
- 2nd: **KristonAI** [[Tech Report](https://arxiv.org/abs/2209.11433)] [[slide]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/voxsrc2022_kristonai_track4.pdf) [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/voxsrc2022_kristonai_track4.mp4)]
- 3rd: **GIST-AiTeR** [[Tech Report](https://arxiv.org/abs/2209.10357)] [[slide]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/gist_slides.pdf) [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/video_aiter.mp4)]

### 2021 VosSRC Track4

[[Workshop](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/interspeech2021.html)]

- 1st: **DKU-DukeECE-Lenovo** [[Tech Report]](https://arxiv.org/abs/2109.02002) [[Video]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/DKU-DukeECE-Lenovo.mp4)
- 2nd: **Bytedance** [[Tech Report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/ByteDance_diarization.pdf)] [[Video]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/Bytedance_SAMI.mp4)
- 3rd: **Tencent**  [[Tech Report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/Tencent_diarization.pdf)]

