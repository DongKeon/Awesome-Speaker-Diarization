# Awesome-Speaker-Diarization [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Some comprehensive papers about speaker diarization (SD).

If you discover any unnoticed documents, please open issues or pull requests (recommended).

---
# Table of Contents
- [Overview](#overview)
- [Reviews](#reviews)
- [EEND (End-to-End Neural Diarization)-based](#eend-end-to-end-neural-diarization-based)
  - [Simulated Dataset](#simulated-dataset)
  - [Post-Processing](#post-processing)
- [Using Target Speaker Embedding](#using-target-speaker-embedding)
- [Clustering-based](#clustering-based)
  - [Embedding](#embedding)
  - [VBx](#varational-bayes-and-hmm)
  - [Scoring](#scoring)
- [Online](#online)
- [Self-Supervised](#self-supervised)
- [Multitask](#multi-tasking)
  - [With Separation](#with-separation)
  - [With ASR](#with-asr)
- [Multi-Channel](#multi-channel)
- [Measurement](#measurement)
- [Multi-Modal](#multi-modal)
  - [With NLP](#with-nlp)
  - [With Vision](#with-vision)
- [Challenge](#challenge)
  - [VoxSRC (VoxCeleb Speaker Recognition Challenge)](#voxsRC-voxceleb-speaker-recognition-challenge)
  - [MISP (Multimodal Information Based Speech Processing) (ICASSP Challenge)](#misp-multimodal-information-based-speech-processing)
---
## Overview
- **DIHARD Keynote Session:** The yellow brick road of diarization, challenges and other neural paths [[Slides]](https://dihardchallenge.github.io/dihard3workshop/slide/The%20yellow%20brick%20road%20of%20diarization,%20challenges%20and%20other%20neural%20paths.pdf) [[Video]](https://www.youtube.com/watch?v=_usbos-SJlg&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=10)

## Reviews
- “A review of speaker diarization: Recent advances with deep learning”, in *Computer Speech & Language, Volume 72,* 2023. (USC) [[Paper](https://arxiv.org/abs/2101.09624)]
- "An Experimental Review of Speaker Diarization methods with application to Two-Speaker Conversational Telephone Speech recordings", in *Computer Speech & Language,* 2023. [[Paper](https://arxiv.org/abs/2305.18074)]
- "Overview of Speaker Modeling and Its Applications: From the Lens of Deep Speaker Representation Learning," in *Submitted to IEEE/ACM TASLP*, 2024. [[Paper](https://arxiv.org/abs/2407.15188)]
---
## EEND (End-to-End Neural Diarization)-based
- **BLSTM-EEND**: "End-to-End Neural Speaker Diarization with Permutation-Free Objectives", in *Proc. Interspeech*, 2019. (Hitachi) [[Paper](https://arxiv.org/abs/1909.05952)]
- **SA-EEND (1)**: “End-to-End Neural Speaker Diarization with Self-attention”, in *Proc. ASRU*, 2019. (Hitachi) [[Paper](https://ieeexplore.ieee.org/abstract/document/9003959)] [[Code](https://github.com/hitachi-speech/EEND)] [[Pytorch](https://github.com/Xflick/EEND_PyTorch)] [[Review](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)]
- **SA-EEND (2)**: “End-to-End Neural Diarization: Reformulating Speaker Diarization as Simple Multi-label Classification”, in *arXiv:2003.02966,* 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2003.02966)] [[Review](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)]
- **SC-EEND**: "Neural Speaker Diarization with Speaker-Wise Chain Rule", in *arXiv:2006.01796*, 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2006.01796)] [[Review](https://velog.io/@fbdp1202/SC-EEND-%EB%A6%AC%EB%B7%B0-Neural-Speaker-Diarization-with-Speaker-Wise-Chain-Rule)]
- **EEND-EDA (1)**: “End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors”, in *Proc. Interspeech,* 2020. (Hitachi) [[Paper](https://arxiv.org/abs/2005.09921)] [[Review](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers)] [[Code](https://github.com/butspeechfit/eend)]
- **EEND-EDA (2)**: “Encoder-Decoder Based Attractor Calculation for End-to-End Neural Diarization”, in *IEEE/ACM TASLP,* 2022. (Hitachi) [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9741374)] [[Review](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers)] [[Code](https://github.com/butspeechfit/eend)]
- **CB-EEND**: "End-to-end Neural Diarization: From Transformer to Conformer", in *Proc. Interspeech*, 2021. (Amazon) [[Paper](https://arxiv.org/abs/2106.07167)] [[Review](https://velog.io/@fbdp1202/CB-EEND-%EB%A6%AC%EB%B7%B0-End-to-end-Neural-Diarization-From-Transformer-to-Conformer)]
- **TDCN-SA**: "End-to-End Diarization for Variable Number of Speakers with Local-Global Networks and Discriminative Speaker Embeddings", in *Proc. ICASSP*, 2021. (Google) [[Paper](https://arxiv.org/abs/2105.02096)] [[Review](https://velog.io/@fbdp1202/TDCN-SA-%EB%A6%AC%EB%B7%B0-End-to-End-Diarization-for-Variable-Number-of-Speakers-with-Local-Global-Networks-and-Discriminative-Speaker-Embeddings)]
- "End-to-End Speaker Diarization Conditioned on Speech Activity and Overlap Detection", in *Proc. IEEE SLT*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2106.04078)]
- **EEND-VC (1)**: "Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds", in *Proc. ICASSP*, 2021. (NTT) [[Paper](https://arxiv.org/abs/2010.13366)] [[Review](https://velog.io/@fbdp1202/EEND-vector-clustering-%EB%A6%AC%EB%B7%B0-Integrating-end-to-end-neural-and-clustering-based-diarization-Getting-the-best-of-both-world)] [[Code](https://github.com/nttcslab-sp/EEND-vector-clustering)]
- **EEND-VC (2)**: "Advances in integration of end-to-end neural and clustering-based diarization for real conversational speech", in *Proc. Interspeech*, 2021. (NTT) [[Paper](https://arxiv.org/abs/2105.09040)] [[Review](https://velog.io/@fbdp1202/EEND-vector-clustering-%EB%A6%AC%EB%B7%B0-Integrating-end-to-end-neural-and-clustering-based-diarization-Getting-the-best-of-both-world)] [[Code](https://github.com/nttcslab-sp/EEND-vector-clustering)]
- "Robust End-to-End Speaker Diarization with Conformer and Additive Margin Penalty," in *Proc. Interspeech*, 2021. (Fano Labs) [[Paper](https://www.isca-archive.org/interspeech_2021/leung21_interspeech.html)]
- **EEND-GLA**: "Towards Neural Diarization for Unlimited Numbers of Speakers Using Global and Local Attractors", in *Proc. ASRU*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2107.01545)] [[Reivew](https://velog.io/@fbdp1202/EEND-EDA-Clustering-%EB%A6%AC%EB%B7%B0-Towards-Neural-Diarization-for-Unlimited-Numbers-of-Speakers-using-Global-and-Local-Attractors)] 
- "**DIVE**: End-to-end Speech Diarization via Iterative Speaker Embedding", in *Proc. ICASSP*, 2022. (Google) [[Paper](https://arxiv.org/abs/2105.13802)]
- **RX-EEND**: “Auxiliary Loss of Transformer with Residual Connection for End-to-End Speaker Diarization”, in *Proc. ICASSP,* 2022. (GIST) [[Paper](https://arxiv.org/abs/2110.07116)] [[Review](https://velog.io/@fbdp1202/RX-EEND-%EB%A6%AC%EB%B7%B0-Auxiliary-Loss-of-Transformer-with-Residual-connection-For-End-to-end-Speaker-Diarization)]
- "End-to-end speaker diarization with transformer", in *Proc. arXiv*, 2022. [[Paper](https://arxiv.org/abs/2112.07463)]
- **EEND-VC-iGMM**: "Tight integration of neural and clustering-based diarization through deep unfolding of infinite Gaussian mixture model", in *Proc. ICASSP*, 2022. (NTT) [[Paper](https://arxiv.org/abs/2202.06524)]
- **EDA-RC**: "Robust End-to-end Speaker Diarization with Generic Neural Clustering", in *Proc. Interspeech*, 2022. (SJTU) [[Paper](https://arxiv.org/abs/2204.08164)]
- **EEND-NAA**: "End-to-End Neural Speaker Diarization with an Iterative Refinement of Non-Autoregressive Attention-based Attractors", in *Proc. Interspeech*, 2022. (JHU) [[Paper](https://www.isca-speech.org/archive/interspeech_2022/rybicka22_interspeech.html)] [[Review](https://dongkeon.notion.site/EEND-NAA-13beac879496807eb0bced8de53e91c1?pvs=4)]
- **Graph-PIT**: "Utterance-by-utterance overlap-aware neural diarization with Graph-PIT", in *Proc. Interspeech*, 2022. (NTT) [[Paper](https://arxiv.org/abs/2207.13888)] [[Code](https://github.com/fgnt/graph_pit)]
- "Efficient Transformers for End-to-End Neural Speaker Diarization", in *Proc. IberSPEECH*, 2022. [[Paper](https://www.isca-speech.org/archive/iberspeech_2022/izquierdodelalamo22_iberspeech.html)]
- "Improving Transformer-based End-to-End Speaker Diarization by Assigning Auxiliary Losses to Attention Heads", in *Proc. ICASSP,* 2023. (HU) [[Paper](https://arxiv.org/abs/2303.01192)]
- **EEND-NA**: “Neural Diarization with Non-Autoregressive Intermediate Attractors”, in *Proc. ICASSP,* 2023. (LINE)  [[Paper](https://arxiv.org/abs/2303.06806)]
- **EEND-EDA-SpkAtt**: "Towards End-to-end Speaker Diarization in the Wild", in *arXiv:2211.01299v1,* 2022. [[Paper](https://arxiv.org/abs/2211.01299v1)]
- "**TOLD**: A Novel Two-Stage Overlap-Aware Framework for Speaker Diarization", in *Proc. ICASSP*, 2023. (Alibaba) [[Paper](https://arxiv.org/abs/2303.05397)] [[Code](https://github.com/alibaba-damo-academy/FunASR)]
- **EEND-IAAE**: "End-to-end neural speaker diarization with an iterative adaptive attractor estimation," in *Neural Networks, Elsevier*. [[Paper](https://www.sciencedirect.com/science/article/pii/S089360802300401X)] [[Code](https://github.com/HaoFengyuan/EEND-IAAE)]
- "Improving End-to-End Neural Diarization Using Conversational Summary Representations", in *Proc. Interspeech*, 2023. (Fano Labs) [[Paper](https://arxiv.org/abs/2306.13863)]
- **AED-EEND**: “Attention-based Encoder-Decoder Network for End-to-End Neural Speaker Diarization with Target Speaker Attractor”, in *Proc. Interspeech,* 2023. (SJTU) [[Paper](https://www.isca-speech.org/archive/interspeech_2023/chen23n_interspeech.html)] [[Review](https://www.notion.so/AED-EEND-EE-903d475a735c46218667a25ed45d4e74)]
- "Self-Distillation into Self-Attention Heads for Improving Transformer-based End-to-End Neural Speaker Diarization", in *Proc. Interspeech*, 2023. (HU) [[Paper](https://www.isca-speech.org/archive/interspeech_2023/jeoung23_interspeech.html)]
- "Powerset Multi-class Cross Entropy Loss for Neural Speaker Diarization", in *Proc. Interspeech*, 2023. (Pyannote) [[Paper](https://www.isca-speech.org/archive/interspeech_2023/plaquet23_interspeech.html)] [[Code](https://github.com/FrenchKrab/IS2023-powerset-diarization)]
- "End-to-End Neural Speaker Diarization with Absolute Speaker Loss", in  *Proc. Interspeech*, 2023. (Pyannote) [[Paper](https://www.isca-speech.org/archive/interspeech_2023/wang23g_interspeech.html)]
- "Blueprint Separable Subsampling and Aggregate Feature Conformer-Based End-to-End Neural Diarization", in *Electronics*, 2023. [[Paper](https://www.mdpi.com/2079-9292/12/19/4118)]
- **EEND-TA**: "Transformer Attractors for Robust and Efficient End-to-End Neural Diarization," in *Proc. ASRU,* 2023. (Fano Labs) [[Paper](https://arxiv.org/abs/2312.06253)]
- "Robust End-to-End Diarization with Domain Adaptive Training and Multi-Task Learning," in *Proc. ASRU,* 2023. (Fano Labs) [[Paper](https://arxiv.org/abs/2312.07136)]
- "NTT speaker diarization system for CHiME-7: multi-domain, multi-microphone End-to-end and vector clustering diarization," in *Proc. ICASSP*, 2024. (NTT) [[Paper](https://arxiv.org/abs/2309.12656)]
- **AED-EEND-EE**: "Attention-based Encoder-Decoder End-to-End Neural Diarization with Embedding Enhancer," in *IEEE/ACM TASLP*, 2024. (SJTU) [[Paper](https://arxiv.org/abs/2309.06672)] [[Review](https://www.notion.so/AED-EEND-EE-903d475a735c46218667a25ed45d4e74)]
- "**DiaPer**: End-to-End Neural Diarization with Perceiver-Based Attractors," in *IEEE/ACM TASLP,* 2024. (BUT) [[Paper](https://arxiv.org/abs/2312.04324)] [[Code](https://github.com/BUTSpeechFIT/DiaPer)] [[Review](https://dongkeon.notion.site/2024-DiaPer-Submitted-TASLP-83fbbd4b8e8645d7a1fe7a08069334ea?pvs=4)]
- "**EEND-DEMUX**: End-to-End Neural Speaker Diarization via Demultiplexed Speaker Embeddings," in *Submitted to IEEE SPL,* 2024. (SNU) [[Paper](https://arxiv.org/abs/2312.06065)] [[Review](https://dongkeon.notion.site/2024-EEND-DEMUX-Submitted-SPL-4bfa79521cc74a78a14e5fc148a7c9c1?pvs=4)]
- "**EEND-M2F**: Masked-attention mask transformers for speaker diarization," in *Proc. Interspeech,* 2024. (Fano Labs) [[arXiv](https://arxiv.org/abs/2401.12600)] [[Pub.](https://www.isca-archive.org/interspeech_2024/harkonen24_interspeech.html)] [[Review](https://dongkeon.notion.site/2024-EEND-M2F-arXiv-8bb1ec11cc2c463cab372cd6cec10318?pvs=4)]
- **EEND-NAA (2)**: "End-to-End Neural Speaker Diarization with Non-Autoregressive Attractors", in *IEEE/ACM TASLP*, 2024. (JHU) [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10629182)] [[Review](https://dongkeon.notion.site/EEND-NAA-13beac879496807eb0bced8de53e91c1?pvs=4)]
- "From Modular to End-to-End Speaker Diarization," *Ph.D. thesis*, 2024. (BUT) [[Paper](https://arxiv.org/abs/2407.08752)]
- "On the calibration of powerset speaker diarization models,"  in *Proc. Interspeech,* 2024. (IRIT) [[Paper](https://www.isca-archive.org/interspeech_2024/plaquet24_interspeech.html)] [[Code](https://github.com/FrenchKrab/IS2024-powerset-calibration)]
- **Local-global EEND**: "Speakers Unembedded: Embedding-free Approach to Long-form Neural Diarization," in *Proc. Interspeech,* 2024. (Amazon) [[Paper](https://www.isca-archive.org/interspeech_2024/li24x_interspeech.html)] [[Review](https://dongkeon.notion.site/Local-global-EEND-2b0a3eb900644c31baf0756da0ca4b5d?pvs=4)]
- "Mamba-based Segmentation Model for Speaker Diarization," *Submitted to ICASSP,* 2025. (NTT) [[arXiv](https://arxiv.org/abs/2410.06459)] [[Code](https://github.com/nttcslab-sp/mamba-diarization)]

### Related Speaker information
  - "Do End-to-End Neural Diarization Attractors Need to Encode Speaker Characteristic Information?," in *Proc. Odyssey*, 2024. [[Paper](https://www.isca-archive.org/odyssey_2024/zhang24_odyssey.html)]
  - "Leveraging Speaker Embeddings in End-to-End Neural Diarization for Two-Speaker Scenarios," in *Proc. Odyssey,* 2024. [[Paper](https://www.isca-archive.org/odyssey_2024/alvareztrejos24_odyssey.html)]


### Simulated Dataset
  - **Concat-and-sum**: “End-to-end neuarl speaker diarization with permuation-free objectives”, in *Proc. Interspeech*, 2019. [[Paper](https://arxiv.org/abs/1909.05952)]
  - “From simulated mixtures to simulated conversations as training data for end-to-end neural diarization” , in *Proc. Interspeech*, 2022. (BUT) [[Paper](https://arxiv.org/abs/2204.00890)] [[Code](https://github.com/BUTSpeechFIT/EEND_dataprep)] [[Review](https://velog.io/@dongkeon/2023-Simulated-Conversations-ICASSP)]
  - **Markov selection**: “Improving the naturalness of simulated conversations for end-to-end neural diarization”, in *Proc. Odyssey*, 2022. (Hitachi) [[Paper](https://arxiv.org/abs/2204.11232)]
  - "Multi-Speaker and Wide-Band Simulated Conversations as Training Data for End-to-End Neural Diarization", in *Proc. ICASSP*, 2023. (BUT) [[Paper](https://arxiv.org/abs/2211.06750)] [[Code](https://github.com/BUTSpeechFIT/EEND_dataprep)] [[Review](https://velog.io/@dongkeon/2023-Simulated-Conversations-ICASSP)]
  - **EEND-EDA-SpkAtt**: "Towards End-to-end Speaker Diarization in the Wild", in *arXiv:2211.01299v1,* 2022. [[Paper](https://arxiv.org/abs/2211.01299v1)]
  - "Property-Aware Multi-Speaker Data Simulation: A Probabilistic Modelling Technique for Synthetic Data Generation," in *CHiME-7 Workshop*, 2023. (NVIDIA) [[Paper](https://arxiv.org/abs/2310.12371)]
  - "Enhancing low-latency speaker diarization with spatial dictionary learning," in *Proc. ICASSP*, 2024.  (NTU) [[Paper](https://ieeexplore.ieee.org/document/10446666)] [[Poster](https://sigport.org/sites/default/files/docs/ENHANCING%20LOW-LATENCY%20SPEAKER%20DIARIZATION%20WITH%20SPATIAL%20DICTIONARY%20LEARNING.pdf)]
  - "Improving Neural Diarization through Speaker Attribute Attractors and Local Dependency Modeling," in *Proc. ICASSP*, 2024. (OSU) [[Paper](https://ieeexplore.ieee.org/document/10446213)]
### Post-Processing
  - **EENDasP**: "End-to-End Speaker Diarization as Post-Processing", in *Proc. ICASSP*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2012.10055)] [[Review](https://velog.io/@fbdp1202/EEND-asp-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-as-Post-Processing) [[Code](https://github.com/DongKeon/EENDasP)]
  - **Dover-Lap**: "DOVER-Lap: A Method for Combining Overlap-aware Diarization Outputs", in *Proc. IEEE SLT*, 2021. (JHU) [[Paper](https://arxiv.org/abs/2011.01997)] [[Review](https://velog.io/@fbdp1202/Dover-lap-%EB%A6%AC%EB%B7%B0-A-method-for-combining-overlap-aware-diarization-outputs)] [[Code](https://github.com/desh2608/dover-lap)]
  - "**DiaCorrect**: Error Correction Back-end For Speaker Diarization," in *Proc. ICASSP*, 2024. (BUT) [[Paper](https://arxiv.org/abs/2309.08377)] [[Code](https://github.com/BUTSpeechFIT/diacorrect)]
---
## Using Target Speaker Embedding
- **TS-VAD**: "Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario", in *Proc. Interspeech*, 2020. [[Paper](https://arxiv.org/abs/2005.07272)] [[Code](https://github.com/dodohow1011/TS-VAD)] [[PPT](https://desh2608.github.io/static/ppt/ts-vad.pdf)]
- “The STC system for the CHiME-6 challenge,” in *CHiME Workshop*, 2020. [[Paper](https://www.isca-speech.org/archive/chime_2020/medennikov20_chime.html)]
- **SEND (1)**: "Speaker Embedding-aware Neural Diarization for Flexible Number of Speakers with Textual Information," in *arXiv:2111.13694*, 2021. (Alibaba) [[Paper](https://arxiv.org/abs/2111.13694)]
- **SEND (2)**: "Speaker Embedding-aware Neural Diarization: an Efficient Framework for Overlapping Speech Diarization in Meeting Scenarios," in *arXiv:2203.09767*, 2022 (Alibaba) [[Paper](https://arxiv.org/abs/2203.09767)]
- **MTEAD**: "Multi-target Filter and Detector for Unknown-number Speaker Diarization", in *IEEE SPL*, 2022. [[Paper](https://arxiv.org/abs/2203.16007)]
- **SOND**: "Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis", in *Proc. EMNLP*, 2022. (Alibaba) [[Paper](https://arxiv.org/abs/2211.10243)] [[Code](https://github.com/alibaba-damo-academy/FunASR)]
- **EDA-TS-VAD**: “Target Speaker Voice Activity Detection with Transformers and Its Integration with End-to-End Neural Diarization”, in *Proc. ICASSP*, 2023. (Microsoft) [[Paper](https://arxiv.org/abs/2208.13085)]
- **Seq2Seq-TS-VAD**: “Target-Speaker Voice Activity Detection via Sequence-to-Sequence Prediction”, in *Proc. ICASSP,* 2023. (DKU) [[Paper](https://arxiv.org/abs/2210.16127)] [[Review](https://velog.io/@dongkeon/2023-Seq2Seq-TS-VAD)]
- **QM-TS-VAD**: "Unsupervised Adaptation with Quality-Aware Masking to Improve Target-Speaker Voice Activity Detection for Speaker Diarization", in *Proc. Interspeech,* 2023. (USTC) [[Paper](https://www.isca-speech.org/archive/interspeech_2023/niu23_interspeech.html)]
- "**ANSD-MA-MSE**: Adaptive Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding," in *IEEE/ACM TASLP*, 2023. (USTC) [[Paper](https://ieeexplore.ieee.org/document/10093997)] [[Code](https://github.com/Maokui-He/NSD-MA-MSE/tree/main)]
- **NSD-MS2S**: "Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding with Sequence-to-Sequence Architecture, " in *Proc. ICASSP*, 2024. (USTC) [[Paper](https://arxiv.org/abs/2309.09180)] [[Code](https://github.com/liyunlongaaa/NSD-MS2S)]
- **PET-TSVAD**: "Profile-Error-Tolerant Target-Speaker Voice Activity Detection," in *Proc. ICASSP*, 2024. (Microsoft) [[Paper](https://arxiv.org/abs/2309.12521)]

## Target Speech Diarization 
- **PTSD**: "Prompt-driven Target Speech Diarization," in *Proc. ICASSP*, 2024. (NUS) [[Paper](https://arxiv.org/abs/2310.14823)]

---
## With Separation or Target Speaker Extraction
- "Integration of speech separation, diarization, and recognition for multi-speaker meetings: System description, comparison, and analysis," in *Proc. SLT,* 2021. (JHU) [[Paper](https://arxiv.org/abs/2011.02014)] [[Blog](https://desh2608.github.io/pages/jsalt/)] [[Review](https://dongkeon.notion.site/Integration-of-speech-separation-diarization-and-recognition-for-multi-speaker-meetings-System-de-f1d2574672834743a94c152b536d78b6?pvs=4)]
- **EEND-SS**: "Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers”, in *Proc. SLT,* 2022. (CMU) [[Paper](https://arxiv.org/abs/2203.17068)] [[Review](https://dongkeon.notion.site/EEND-SS-Joint-End-to-End-Neural-Speaker-Diarization-and-Speech-Separation-for-Flexible-Number-of-Sp-32a6a76f796341ca972b3959c1b7311d?pvs=4)]
- "**TS-SEP**: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings", in *IEEE/ACM TASLP*, 2024. [[Paper](https://arxiv.org/abs/2303.03849)]
- "Continuous Target Speech Extraction: Enhancing Personalized Diarization and Extraction on Complex Recordings," in *arXiv:2401.15993*, 2024. (Tencent) [[Paper](https://arxiv.org/abs/2401.15993)] [[Demo](https://herbhezhao.github.io/Continuous-Target-Speech-Extraction/)]
- "**PixIT**: Joint Training of Speaker Diarization and Speech Separation from Real-world Multi-speaker Recordings," in *Proc. Odyssey*, 2024. [[Paper](https://arxiv.org/abs/2403.02288)] [[Code](https://github.com/joonaskalda/PixIT)]
- **MC-EEND**: "Multi-channel Conversational Speaker Separation via Neural Diarization," in *IEEE/ACM TASLP,* 2024. (OSU) [[Paper](https://arxiv.org/abs/2311.08630)]
- "**USED**: Universal Speaker Extraction and Diarization," in *submitted to IEEE/ACM TASLP*, 2024. (CUHK) [[Paper](https://arxiv.org/abs/2309.10674)] [[Demo](https://ajyy.github.io/demo/USED/)] [[Util](https://github.com/msinanyildirim/USED-splits)] [[Review](https://dongkeon.notion.site/USED-Universal-Speaker-Extraction-and-Diarization-dcaf0e22ec334286b188ab5561bdbd27?pvs=4)]
- "Neural Blind Source Separation and Diarization for Distant Speech Recognition," in *Proc. Interspeech*, 2024. (AIST) [[Paper](https://arxiv.org/pdf/2406.08396)]
- "TalTech-IRIT-LIS Speaker and Language Diarization Systems for DISPLACE 2024," in *Proc. Interspeech,* 2024. (Pyannote) [[Paper](https://arxiv.org/abs/2407.12743)]

---
## Multi-Channel
- "Multi-Channel End-to-End Neural Diarization with Distributed Microphones", in *Proc. ICASSP*, 2022. (Hitachi) [[Paper](https://arxiv.org/abs/2110.04694)]
- "Multi-Channel Speaker Diarization Using Spatial Features for Meetings", in *Proc. ICASSP*, 2022. (Tencent) [[Paper](https://ieeexplore.ieee.org/document/9747343)]
- "Mutual Learning of Single- and Multi-Channel End-to-End Neural Diarization," in *Proc. IEEE SLT*, 2023. (Hitachi) [[Paper](https://arxiv.org/abs/2210.03459)]
- "Semi-supervised multi-channel speaker diarization with cross-channel attention", in *Proc. ASRU,* 2023. (USTC) [[Paper](https://arxiv.org/abs/2307.08688)]
- "**UniX-Encoder**: A Universal X-Channel Speech Encoder for Ad-Hoc Microphone Array Speech Processing," in *arXiv:2310.16367*, 2024. (JHU, Tencent) [[Paper](https://arxiv.org/abs/2310.16367)]
- "Channel-Combination Algorithms for Robust Distant Voice Activity and Overlapped Speech Detection," in *IEEE/ACM TASLP,* 2024. [[Paper](https://arxiv.org/abs/2402.08312)]
- "A Spatial Long-Term Iterative Mask Estimation Approach for Multi-Channel Speaker Diarization and Speech Recognition," in *Proc. ICASSP*, 2024. (USTC) [[Paper](https://ieeexplore.ieee.org/document/10446168)]
- **MC-EEND**: "Multi-channel Conversational Speaker Separation via Neural Diarization," in *IEEE/ACM TASLP,* 2024. (OSU) [[Paper](https://arxiv.org/abs/2311.08630)]
- "**ASoBO**: Attentive Beamformer Selection for Distant Speaker Diarization in Meetings," in *Proc. Interspeech*, 2024. (LIUM) [[Paper](https://arxiv.org/abs/2406.03251)]
- "Multi-channel Speaker Counting for EEND-VC-based Speaker Diarization on Multi-domain Conversation," in *Proc. ICASSP,* 2025. (NTT) [[Pub.](https://ieeexplore.ieee.org/abstract/document/10888681)] [[Review](https://dongkeon.notion.site/Multi-channel-Speaker-Counting-1faeac879496809b9075e79580eb9a6e?pvs=4)]

---
## Online
- "Supervised online diarization with sample mean loss for multi-domain data", in *Proc. ICASSP*, 2020 [[Paper](https://arxiv.org/abs/1911.01266)] [[Code](https://github.com/DonkeyShot21/uis-rnn-sml)]
- "Online End-to-End Neural Diarization with Speaker-Tracing Buffer", in *Proc. IEEE SLT*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2006.02616)]
- **BW-EDA-EEND**: "BW-EDA-EEND: Streaming End-to-End Neural Speaker Diarization for a Variable Number of Speakers", in *Proc. Interspeech*, 2021. (Amazon) [[Paper](https://arxiv.org/abs/2011.02678)]
- **FS-EEND**: "Online Streaming End-to-End Neural Diarization Handling Overlapping Speech and Flexible Numbers of Speakers", in *Proc. Interspeech*, 2021. (Hitachi) [[Paper](https://arxiv.org/abs/2101.08473)] [[Reivew](https://velog.io/@fbdp1202/FS-EEND-%EB%A6%AC%EB%B7%B0-Online-end-to-end-diarization-handling-overlapping-speech-and-flexible-numbers-of-speakers)] 
- **Diart**: "Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation", in *Proc. ASRU*, 2021. [[Paper](https://arxiv.org/abs/2109.06483)] [[Code](https://github.com/juanmc2005/diart)]
- "Low-Latency Online Speaker Diarization with Graph-Based Label Generation", in *Proc. Odyssey*, 2022. (DKU) [[Paper](https://arxiv.org/abs/2111.13803)]
- **EEND-GLA**: "Online Neural Diarization of Unlimited Numbers of Speakers Using Global and Local Attractors", in *IEEE/ACM TASLP,* 2022. (Hitachi) [[Paper](https://arxiv.org/abs/2206.02432)]
- **Online TS-VAD**: "Online Target Speaker Voice Activity Detection for Speaker Diarization", in *Proc. Interspeech*, 2022. (DKU) [[Paper](https://arxiv.org/abs/2207.05920)]
- "Absolute decision corrupts absolutely: conservative online speaker diarisation", in *Proc. ICASSP*, 2023. (Naver) [[Paper](https://arxiv.org/abs/2211.04768)]
- "A Reinforcement Learning Framework for Online Speaker Diarization", in *Under Review. NeruIPS*, 2023. (CU) [[Paper](https://arxiv.org/abs/2302.10924)]
- **OTS-VAD**: "End-to-end Online Speaker Diarization with Target Speaker Tracking," in *Submitted IEEE/ACM TASLP,* 2023. (DKU) [[Paper](https://arxiv.org/abs/2310.08696)]
- **FS-EEND**: "Frame-wise streaming end-to-end speaker diarization with non-autoregressive self-attention-based attractors," in *Proc. ICASSP,* 2024. (Hangzhou) [[Paper](https://arxiv.org/abs/2309.13916)] [[Code](https://github.com/Audio-WestlakeU/FS-EEND)]
- "Online speaker diarization of meetings guided by speech separation," in *Proc. ICASSP,* 2024. (LTCI) [[Paper](https://browse.arxiv.org/abs/2402.00067)] [[Code](https://github.com/egruttadauria98/SSpaVAlDo)]
- "Interrelate Training and Clustering for Online Speaker Diarization," in *IEEE/ACM TASLP,* 2024. [[Paper](https://ieeexplore.ieee.org/abstract/document/10418572)]
---
## Clustering-based
- **UIS-RNN**: "Fully Supervised Speaker Diarization" (Google) [[Paper](https://arxiv.org/abs/1810.04719)] [[Code](https://github.com/google/uis-rnn)]
- **DNC**: "Discriminative Neural Clustering for Speaker Diarisation", in *Proc. IEEE SLT*, 2019. [[Paper](https://arxiv.org/abs/1910.09703)] [[Code](https://github.com/FlorianKrey/DNC)] [[Review](https://velog.io/@dongkeon/2019-DNC-SLT)]
- **Pyannote**: "pyannote.audio: neural building blocks for speaker diarization", in *Proc. ICASSP*, 2020. (CNRS) [[Paper](https://arxiv.org/abs/1911.01255)] [[Code](https://github.com/pyannote/pyannote-audio)] [[Video](https://www.youtube.com/watch?v=37R_R82lfwA)]
- **NME-SC**: “Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap”, *IEEE SPL,* 2019. [[Paper](https://arxiv.org/abs/2003.02405)] [[Code](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)]
- **Resegmentation with VB**: “Overlap-Aware Diarization: Resegmentation Using Neural End-to-End Overlapped Speech Detection”, in *Proc. ICASSP*, 2020. [[Paper](https://ieeexplore.ieee.org/document/9053096)]
- **Pyannote 2.0**: "End-to-end speaker segmentation for overlap-aware resegmentation", in *Proc. Interspeech*, 2021. (CNRS) [[Paper](https://arxiv.org/abs/2104.04045)] [[Code](https://github.com/pyannote/pyannote-audio)] [[Video](https://www.youtube.com/watch?v=wDH2rvkjymY)]
- **UMAP-Leiden**: "Reformulating Speaker Diarization as Community Detection With Emphasis On Topological Structure", in *Proc. ICASSP*, 2022. (Alibaba) [[Paper](https://arxiv.org/abs/2204.12112)]
- **SCALE**: "Spectral Clustering-aware Learning of Embeddings for Speaker Diarisation", in *Proc. ICASSP*, 2023. (CAM) [[Paper](https://arxiv.org/abs/2210.13576)]
- **SHARC**: "Supervised Hierarchical Clustering using Graph Neural Networks for Speaker Diarization", in *Proc. ICASSP*, 2023. (IISC) [[Paper](https://arxiv.org/abs/2302.12716)]
- **CDGCN**: "Community Detection Graph Convolutional Network for Overlap-Aware Speaker Diarization," in *Proc. ICASSP*, 2023. (XMU) [[Paper](https://arxiv.org/abs/2306.14530)]
- "**Pyannote.Audio 2.1**: Speaker Diarization Pipeline: Principle, Benchmark and Recipe", in *Proc. Interspeech*, 2023. (CNRS) [[Paper](https://www.isca-speech.org/archive/interspeech_2023/bredin23_interspeech.html)]
- **GADEC**: "Graph attention-based deep embedded clustering for speaker diarization,", in *Speech Communication*, 2023. (NJUPT) [[Paper](https://www.sciencedirect.com/science/article/pii/S0167639323001255)]
- "Overlap-aware End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization," in *submitted to IEEE/ACM TASLP*, 2024. [[Paper](https://arxiv.org/abs/2401.12850)]
- "Apollo's Unheard Voices: Graph Attention Networks for Speaker Diarization and Clustering for Fearless Steps Apollo Collection," in *Proc. ICASSP*, 2024. (UTD) [[Paper](https://ieeexplore.ieee.org/document/10446231)]
- "Multi-View Speaker Embedding Learning for Enhanced Stability and Discriminability," in *Proc. ICASSP*, 2024. (Tsinghua) [[Paper](https://ieeexplore.ieee.org/abstract/document/10448494)]
- "Towards Unsupervised Speaker Diarization System for Multilingual Telephone Calls Using Pre-trained Whisper Model and Mixture of Sparse Autoencoders," in *arXiv:2407.01963*, 2024. [[Paper](https://arxiv.org/abs/2407.01963)]
- "Investigating Confidence Estimation Measures for Speaker Diarization," in *Proc. Interspeech*, 2024. [[Paper](https://arxiv.org/abs/2406.17124)]
- "Once more Diarization: Improving meeting transcription systems through segment-level speaker reassignment," in *Proc. Interspeech*, 2024. (PU) [[Paper](https://arxiv.org/abs/2406.03155)] [[Pub.](https://www.isca-archive.org/interspeech_2024/boeddeker24_interspeech.html)] [[Code](https://github.com/fgnt/speaker_reassignment)]
- **E-SHARC**: "End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization," in *IEEE/ACM TASLP,* 2025. (IISC) [[Paper](https://ieeexplore.ieee.org/abstract/document/10830571/)]

## Embedding (With Clustering)
- "Multi-Scale Speaker Diarization With Neural Affinity Score Fusion", in *Proc. ICASSP*, 2021. (USC) [[Paper](https://arxiv.org/abs/2011.10527)]
- **AA+DR+NS**: "Adapting Speaker Embeddings for Speaker Diarisation", in *Proc. Interspeech*, 2021. (Naver) [[Paper](https://arxiv.org/abs/2104.02879)] [[Review](https://dongkeon.notion.site/2021-AutoEncoder-attention-aggregation-Interspeech-a9a8e6870a17418597fb0dad83d459fc?pvs=4)]
- **GAT+AA**: "Multi-scale speaker embedding-based graph attention networks for speaker diarisation", in *Proc. ICASSP*, 2022. (Naver) [[Paper](https://arxiv.org/abs/2110.03361)]
- **MSDD**: "Multi-scale Speaker Diarization with Dynamic Scale Weighting", in *Proc. Interspeech*, 2022. (NVIDIA) [[Paper](https://arxiv.org/abs/2203.15974)] [[Code](https://github.com/NVIDIA/NeMo)] [[Blog](https://developer.nvidia.com/blog/dynamic-scale-weighting-through-multiscale-speaker-diarization/)]
- "In Search of Strong Embedding Extractors For Speaker Diarization", in *Proc. ICASSP*, 2023. (Naver) [[Paper](https://arxiv.org/abs/2210.14682)] [[Review](https://velog.io/@dongkeon/2023-In-Search-of-Strong-Embedding-Extractors-For-Speaker-Diarization-ICASSP)]
- **PRISM**: "PRISM: Pre-trained Indeterminate Speaker Representation Model for Speaker Diarization and Speaker Verification", in *Proc. Interspeech*, 2022. (Alibaba) [[Paper](https://arxiv.org/abs/2205.07450)]
- **DR-DESA**: "Advancing the dimensionality reduction of speaker embeddings for speaker diarisation: disentangling noise and informing speech activity", in *Proc. ICASSP*, 2023. (Naver) [[Paper](https://arxiv.org/abs/2110.03380)] [[Review](https://dongkeon.notion.site/2023-DR-DESA-ICASSP-c5ab46d215f243b5887b1c5d0b328bb6?pvs=4)]
- **HEE**: "High-resolution embedding extractor for speaker diarisation", in *Proc. ICASSP*, 2023. (Naver)  [[Paper](https://arxiv.org/abs/2211.04060)] [[Review](https://dongkeon.notion.site/2023-HEE-ICASSP-fa7f95d6641744fa90d34ef73d2b8463?pvs=4)]
- "Frame-wise and overlap-robust speaker embeddings for meeting diarization", in *Proc. ICASSP*, 2023. (PU) [[Paper](https://arxiv.org/pdf/2306.00625.pdf)] [[Review](https://dongkeon.notion.site/2023-Frame-wise-and-overlap-robust-speaker-embeddings-ICASSP-2b1878e76b6b45d09b021f00edee036b?pvs=4)]
- "A Teacher-Student approach for extracting informative speaker embeddings from speech mixtures", in *Proc. Interspeech*, 2023. (PU) [[Paper](https://arxiv.org/abs/2306.00634)]
- "Geodesic interpolation of frame-wise speaker embeddings for the diarization of meeting scenarios", in *Proc. ICASSP*, 2024. (PU) [[Paper](https://arxiv.org/abs/2401.03963)] [[Review](https://dongkeon.notion.site/2024-Geodesic-Interpolation-ICASSP-e7ec31aaa1fb49b3aa0f18dead92c5fc?pvs=4)]
- "Speaker Embeddings With Weakly Supervised Voice Activity Detection For Efficient Speaker Diarization," in *Proc. Odyssey*, 2024. (IDLab) [[Paper](https://www.isca-archive.org/odyssey_2024/thienpondt24_odyssey.html)]
- "Efficient Speaker Embedding Extraction Using a Twofold Sliding Window Algorithm for Speaker Diarization," in *Proc. Interspeech,* 2024. (HU) [[Paper](https://www.isca-archive.org/interspeech_2024/choi24d_interspeech.html)]
- "Variable Segment Length and Domain-Adapted Feature Optimization for Speaker Diarization," in *Proc. Interspeech,* 2024. (XMU) [[Paper](https://www.isca-archive.org/interspeech_2024/zhang24b_interspeech.html)] [[Code](https://github.com/xiaoaaa2/Ada-sd)]


## With Speaker Identification
- "Uncertainty Quantification in Machine Learning for Joint Speaker Diarization and Identification, in *Submitted to IEEE/ACM TASLP,* 2024. [[Paper](https://arxiv.org/abs/2312.16763)]

## Speaker Recogniton & Verification
- "Xi-Vector Embedding for Speaker Recognition," in *IEEE, SPL*. (A*STAR) [[Paper](https://arxiv.org/abs/2108.05679)] [[Review](https://dongkeon.notion.site/2021-Xi-Vector-SPL-ce538c87f6d64545acb557a223af3670?pvs=4)]
- "Build a SRE Challenge System: Lessons from VoxSRC 2022 and CNSRC 2022," in *Proc. Interspeech*, 2023. (SJTU) [[Paper](https://www.isca-speech.org/archive/interspeech_2023/chen23m_interspeech.html)]
- **RecXi** "Disentangling Voice and Content with Self-Supervision for Speaker Recognition," in *Proc. NeurIPS,* 2023. (A*STAR) [[Paper](https://arxiv.org/abs/2310.01128)]
- "**ECAPA2**: A Hybrid Neural Network Architecture and Training Strategy for Robust Speaker Embeddings," in *Proc. ASRU,* 2023. (IDLab) [[Paper](https://arxiv.org/abs/2401.08342)] [[Model](https://huggingface.co/Jenthe/ECAPA2)] [[Review](https://dongkeon.notion.site/2023-ECAPA2-ASRU-962943495f2348dab3872e3481bc08a6?pvs=4)]
- "Rethinking Session Variability: Leveraging Session Embeddings for Session Robustness in Speaker Verification," in *Proc. ICASSP*, 2024. (Naver) [[Paper](https://arxiv.org/abs/2309.14741)]
- "Leveraging In-the-Wild Data for Effective Self-Supervised Pretraining in Speaker Recognition," in *Proc. ICASSP*, 2024. (CUHK) [[Paper](https://arxiv.org/abs/2309.11730)]
- "Disentangled Representation Learning for Environment-agnostic Speaker Recognition," in *Proc. Interspeech,* 2024. (KAIST) [[arXiv](https://arxiv.org/abs/2406.14559)] [[Pub.](https://www.isca-archive.org/interspeech_2024/nam24b_interspeech.html)] [[Code](https://github.com/kaistmm/voxceleb-disentangler)]

## Scoring
- **LSTM scoring**: "LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization", in *Proc. Interspeech*, 2019. (DKU) [[Paper](https://arxiv.org/abs/1907.10393)]
- "Self-Attentive Similarity Measurement Strategies in Speaker Diarization", in *Proc. Interspeech*, 2020. (DKU) [[Paper](https://www.isca-speech.org/archive/interspeech_2020/lin20_interspeech.html)]
- “Similarity Measurement of Segment-Level Speaker Embeddings in Speaker Diarization”, *IEEE/ACM TASLP,* 2023. (DKU) [[Paper](https://ieeexplore.ieee.org/document/9849033)]

## Varational Bayes and HMM 
  ### VBx Series
  - "Speaker Diarization based on Bayesian HMM with Eigenvoice Priors", in *Proc. Odyssey*, 2018. (BUT) [[Paper](https://www.isca-speech.org/archive/odyssey_2018/diez18_odyssey.html)]
  - "VB-HMM Speaker Diarization with Enhanced and Refined Segment Representation", in *Proc. Odyssey*, 2018. (Tsinghua) [[Paper](https://www.isca-speech.org/archive_v0/Odyssey_2018/abstracts/53.html)]
  - “Analysis of Speaker Diarization Based on Bayesian HMM With Eigenvoice Priors”, *IEEE/ACM TASLP,* 2019. (BUT) [[Paper](https://ieeexplore.ieee.org/document/8910412)]
  - "BUT System Description for **DIHARD Speech Diarization Challenge 2019**", in *arXiv:1910.08847*, 2019. (BUT) [[Paper](https://arxiv.org/abs/1910.08847)]
  - "Bayesian HMM Based x-Vector Clustering for Speaker Diarization", in *Proc. Interspeech*, 2019. (BUT) [[Paper](https://www.isca-speech.org/archive_v0/Interspeech_2019/abstracts/2813.html)]
  - "Optimizing Bayesian Hmm Based X-Vector Clustering for **the Second Dihard Speech Diarization Challenge**", in *Proc. ICASSP*, 2020. (BUT) [[Paper](https://ieeexplore.ieee.org/document/9053982)]
  - "Analysis of the but Diarization System for **Voxconverse Challenge**", in *Proc. ICASSP*, 2021. (BUT) [[Paper](https://ieeexplore.ieee.org/document/9414315)] [[Code](https://github.com/BUTSpeechFIT/VBx/tree/v1.1_VoxConverse2020)]
  - "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks", in *Computer Speech & Language*, 2022. (BUT) [[Paper](https://arxiv.org/abs/2012.14952)]
  - **MS-VBx**: "Multi-Stream Extension of Variational Bayesian HMM Clustering (MS-VBx) for Combined End-to-End and Vector Clustering-based Diarization", in *Proc. Interspeech*, 2023. (NTT) [[Paper](https://arxiv.org/abs/2305.13580)]
  - **DVBx**: "Discriminative Training of VBx Diarization", in *Proc. ICASSP*, 2024. (BUT) [[Paper](https://arxiv.org/abs/2310.02732)] [[Code](https://github.com/BUTSpeechFIT/DVBx)]

  ### Variational Bayes 
  - "Variational Bayesian methods for audio indexing", in *Proc. ICMI-MLMI*, 2005. [[Paper](https://www.eurecom.fr/fr/publication/1739/download/mm-valefa-050923.pdf)]
  - "Bayesian analysis of speaker diarization with eigenvoice priors", in *CRIM, Montreal, Technical Report*, 2008. [[Paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=36db5cc928d01b13246582d71bde84fabbd24a19)]
  - "Unsupervised Methods for Speaker Diarization: An Integrated and Iterative Approach", *IEEE/ACM TASLP,* 2013. [[Paper](https://ieeexplore.ieee.org/abstract/document/6518171/)]
  - "Diarization resegmentation in the factor analysis subspace", in *Proc. ICASSP*, 2015. [[Paper](https://ieeexplore.ieee.org/abstract/document/7178881)]
  - "Diarization is hard: some experiences and lessons learned for the JHU team in **the inaugural DIHARD challenge**", in *Proc. Interspeech*, 2018. [[Paper](https://www.isca-speech.org/archive/interspeech_2018/sell18_interspeech.html)]

  ### Normalization
  - "Analysis of i-vector length normalization in speaker recognition systems", in *Proc. Interspeech*, 2011. [[Paper](https://www.isca-speech.org/archive/interspeech_2011/garciaromero11_interspeech.html)]

  ### PLDA (Probabilistic Linear Discriminant Analysis)
  - "The speaker partitioning problem", in *Proc. Odyssey*, 2018. [[Paper](https://www.isca-speech.org/archive_open/odyssey_2010/od10_034.html)]
  - "Discriminatively trained probabilistic linear discriminant analysis for speaker verification", in *Proc. ICASSP*, 2021. [[Paper](https://ieeexplore.ieee.org/document/5947437)]
  - "Speaker diarization with plda i-vector scoring and unsupervised calibration", in *Proc. IEEE SLT*, 2014. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7078610)]
  - "Iterative PLDA Adaptation for Speaker Diarization", in *Proc. Interspeech*, 2016. [[Paper](https://www.isca-speech.org/archive/interspeech_2016/lan16_interspeech.html)]
  - "Domain Adaptation of PLDA Models in Broadcast Diarization by Means of Unsupervised Speaker Clustering, in *Proc. Interspeech*, 2017. [[Paper](https://www.isca-speech.org/archive/interspeech_2017/vinals17_interspeech.html)]
  - "Estimation of the Number of Speakers with Variational Bayesian PLDA in **the DIHARD Diarization Challenge**", in *Proc. Interspeech*, 2018. [[Paper](https://www.isca-speech.org/archive/interspeech_2018/vinals18_interspeech.html)]
  - **DCA-PLDA** "A Speaker Verification Backend with Robust Performance across Conditions”, in *Computer & Language*, 2022. [[Paper](https://arxiv.org/pdf/2102.01760.pdf)] [[Code](https://github.com/luferrer/DCA-PLDA)]
  - "Generalized domain adaptation framework for parametric back-end in speaker recognition", in *arXiv:2305.15567*, 2023. [[Paper](https://arxiv.org/abs/2305.15567)]
---
## With ASR
- "Transcribe-to-Diarize: Neural Speaker Diarization for Unlimited Number of Speakers using End-to-End Speaker-Attributed ASR," in *Proc. ICASSP*, 2022. [[Paper](https://arxiv.org/abs/2110.03151)]
- "Tandem Multitask Training of Speaker Diarisation and Speech Recognition for Meeting Transcription", in *Proc. Interspeech*, 2022. [[Paper](https://arxiv.org/abs/2207.03852)]
- "Unified Modeling of Multi-Talker Overlapped Speech Recognition and Diarization with a Sidecar Separator", in *Proc. Interspeech*, 2023. (CUHK) [[Paper](https://arxiv.org/abs/2305.16263)]
- "Multi-resolution Approach to Identification of Spoken Languages and to Improve Overall Language Diarization System using Whisper Model", in *Proc. Interspeech*, 2023.
- "Speaker Diarization for ASR Output with T-vectors: A Sequence Classification Approach", in *Proc. Interspeech*, 2023. [[Paper](https://www.isca-speech.org/archive/interspeech_2023/yousefi23_interspeech.html)]
- "Lexical Speaker Error Correction: Leveraging Language Models for Speaker Diarization Error Correction", in *Proc. Interspeech*, 2023. (Amazon) [[Paper](https://arxiv.org/abs/2306.09313)]
- "Enhancing Speaker Diarization with Large Language Models: A Contextual Beam Search Approach,", in *Proc. ICASSP*, 2024. (NVIDIA) [[Paper](https://arxiv.org/abs/2309.05248)]
- **WEEND**: "Towards Word-Level End-to-End Neural Speaker Diarization with Auxiliary Network," in *arXiv:2309.08489*, 2024. (Google) [[Paper](https://arxiv.org/abs/2309.08489)] [[Supplementary](https://github.com/google/speaker-id/tree/master/publications/WEEND)]
- "One model to rule them all ? Towards End-to-End Joint Speaker Diarization and Speech Recognition", in *Proc. ICASSP*, 2024. (CMU) [[Paper](https://arxiv.org/abs/2310.01688)]
- "Meeting Recognition with Continuous Speech Separation and Transcription-Supported Diarization," in *arXiv:2309.16482*, 2024. (PU) [[Paper](https://arxiv.org/abs/2309.16482)]
- “Joint Inference of Speaker Diarization and ASR with Multi-Stage Information Sharing," in *Proc. ICASSP*, 2024. (DKU) [[Paper](https://sites.duke.edu/dkusmiip/files/2024/03/icassp24_weiqing.pdf)]
- "Multitask Speech Recognition and Speaker Change Detection for Unknown Number of Speakers" in *Proc. ICASSP*, 2024. (Idiap) [[Paper](https://ieeexplore.ieee.org/document/10446130)]
- "A Spatial Long-Term Iterative Mask Estimation Approach for Multi-Channel Speaker Diarization and Speech Recognition," in *Proc. ICASSP*, 2024. (USTC) [[Paper](https://ieeexplore.ieee.org/document/10446168)]
- "On the Success and Limitations of Auxiliary Network Based Word-Level End-to-End Neural Speaker Diarization," in *Proc. Interspeech*, 2024. (Google) [[Paper](https://www.isca-archive.org/interspeech_2024/huang24d_interspeech.html)]
- **Sortformer**: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens, in *Submitted IEEE/ACM TASLP,* 2024. (NVIDIA) [[Paper](https://arxiv.org/abs/2409.06656)]
  ### Speaker-attributed ASR
  - "**SA-Paraformer**: Non-autoregressive End-to-End Speaker-Attributed ASR," in *Proc. ASRU*, 2023. (Alibaba) [[Paper](https://arxiv.org/abs/2310.04863)]
  - "Speaker Mask Transformer for Multi-talker Overlapped Speech Recognition," in *arXiv:2312.10959*, 2024. (NICT) [[Paper](https://arxiv.org/abs/2312.10959)]
  - "On Speaker Attribution with SURT," in *Proc. Odyssey,* 2024. (JHU) [[Paper](https://www.isca-archive.org/odyssey_2024/raj24_odyssey.html)]
  - "Improving Speaker Assignment in Speaker-Attributed ASR for Real Meeting Applications," in *Proc. Odyssey,* 2024. (CNRS) [[Paper](https://www.isca-archive.org/odyssey_2024/cui24_odyssey.html)]
  ### Target Speaker ASR
  - "Target Speaker ASR with Whisper," in *Submitted to ICASSP,* 2025. (BUT) [[Paper](https://arxiv.org/abs/2409.09543)] [[Code(Not yet)](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)]
## Language Diarization
- "End-to-End Spoken Language Diarization with Wav2vec Embeddings", in *Proc. Interspeech*, 2023. [[Paper](https://www.isca-speech.org/archive/interspeech_2023/mishra23_interspeech.html)] [[Code](https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization)]
- "Multi-resolution Approach to Identification of Spoken Languages and To Improve Overall Language Diarization System Using Whisper Model," in *Proc. Interspeech*, 2023. [[Paper](https://www.isca-speech.org/archive/interspeech_2023/vachhani23_interspeech.html)]

## With NLP (LLM)
- "Exploring Speaker-Related Information in Spoken Language Understanding for Better Speaker Diarization", in *Proc. ACL*, 2023. (Alibaba) [[Paper](https://arxiv.org/abs/2305.12927)]
- **MMSCD**, "Encoder-decoder multimodal speaker change detection", in *Proc. Interspeech*, 2023. (Naver) [[Paper](https://arxiv.org/abs/2306.00680)]
- "Aligning Speakers: Evaluating and Visualizing Text-based Diarization Using Efficient Multiple Sequence Alignment,", in *Proc. ICTAI*, 2023. [[Paper](https://arxiv.org/abs/2309.07677)]
- "**DiariST**: Streaming Speech Translation with Speaker Diarization," in *Proc. ICASSP*, 2024. (Microsoft) [[Paper](https://arxiv.org/abs/2309.08007)] [[Code](https://github.com/Mu-Y/DiariST)]
- **JPCP:** "Improving Speaker Diarization using Semantic Information: Joint Pairwise Constraints Propagation," in *arXiv:2309.10456*, 2024. (Alibaba) [[Paper](https://arxiv.org/abs/2309.10456)]
- "**DiarizationLM**: Speaker Diarization Post-Processing with Large Language Models," in *Proc. Interspeech*, 2024. (Google) [[Pub.](https://www.isca-archive.org/interspeech_2024/wang24h_interspeech.html)] [[Paper](https://arxiv.org/abs/2401.03506)] [[Code](https://github.com/google/speaker-id/tree/master/DiarizationLM)] [[Review](https://dongkeon.notion.site/DiarizationLM-1f8eac8794968030839ac145abb0a546?pvs=4)]
- "LLM-based speaker diarization correction: A generalizable approach," in *Submitted to IEEE/ACM TASLP*, 2024. [[Paper](https://arxiv.org/abs/2406.04927)] 
- "AG-LSEC: Audio Grounded Lexical Speaker Error Correction," in *Proc. Interspeech*, 2024. (Amazon) [[Paper](https://arxiv.org/abs/2406.17266)] 


## With Vision
- "Who said that?: Audio-visual speaker diarisation of real-world meetings", in *Proc. Interspeech*, 2019. (Naver) [[Paper](https://arxiv.org/abs/1906.10042)]
- "Self-supervised learning for audio-visual speaker diarization", in *Proc. ICASSP*, 2020. (Tencent) [[Paper](https://arxiv.org/abs/2002.05314)] [[Blog](https://yifan16.github.io/av-spk-diarization/)]
- **AVA-AVD (AVR-Net)**: "AVA-AVD: Audio-Visual Speaker Diarization in the Wild", in *Proc. ACM MM*, 2022. [[Paper](https://arxiv.org/abs/2111.14448)] [[Code](https://github.com/zcxu-eric/AVA-AVD)] [[Video](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3503161.3548027&file=MM22-fp1169.mp4)] 
- "End-to-End Audio-Visual Neural Speaker Diarization", in *Proc. Interspeech*, 2022. (USTC) [[Paper](https://www.isca-speech.org/archive/interspeech_2022/he22c_interspeech.html)] [[Code](https://github.com/mispchallenge/misp2022_baseline/tree/main/track1_AVSD)] [[Review](https://velog.io/@dongkeon/2022-End-to-End-Audio-Visual-Neural-Speaker-Diarization-2022-Interspeech)]
- **DyViSE**: "DyViSE: Dynamic Vision-Guided Speaker Embedding for Audio-Visual Speaker Diarization", in *Proc. MMSP*, 2022. (THU) [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9948860)] [[Code](https://github.com/zaocan666/DyViSE)]
- "Audio-Visual Speaker Diarization in the Framework of Multi-User Human-Robot Interaction", in *Proc. ICASSP*, 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10096295)]
- **STHG**: "Spatial-Temporal Heterogeneous Graph Learning for Advanced Audio-Visual Diarization, in *Proc. CVPR*, 2023. (Intel) [[Paper](https://arxiv.org/abs/2306.10608)]
- "Speaker Diarization of Scripted Audiovisual Content," in *arXiv:2308.02160*, 2024. (Amazon) [[Paper](https://arxiv.org/abs/2308.02160)]
- "Uncertainty-Guided End-to-End Audio-Visual Speaker Diarization for Far-Field Recordings," in *Proc. ACM MM*, 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3581783.3612424)]
- "Joint Training or Not: An Exploration of Pre-trained Speech Models in Audio-Visual Speaker Diarization," in *Springer Computer Science proceedings,* 2023. [[Paper](https://arxiv.org/abs/2312.04131)]
- **EEND-EDA++**: "Late Audio-Visual Fusion for In-The-Wild Speaker Diarization," in *arXiv:2211.01299v2*, 2023. [[Paper](https://arxiv.org/abs/2211.01299v2)]
- "**AFL-Net**: Integrating Audio, Facial, and Lip Modalities with Cross-Attention for Robust Speaker Diarization in the Wild," in *Proc. ICASSP*, 2024. (Tencent) [[Paper](https://arxiv.org/abs/2312.05730)] [[Demos](https://yyk77.github.io/afl_net.github.io/)]
- "**Multichannel AV-wav2vec2**: A Framework for Learning Multichannel Multi-Modal Speech Representation," in *Proc. AAAI,* 2024. (Tencent) [[Paper](https://arxiv.org/abs/2401.03468)]
- "Multi-Input Multi-Output Target-Speaker Voice Activity Detection For Unified, Flexible, and Robust Audio-Visual Speaker Diarization," in *Submitted to IEEE/ACM TASLP*. (DKU) [[Paper](https://arxiv.org/abs/2401.08052)]
- "**3D-Speaker-Toolkit**: An Open Source Toolkit for Multi-modal Speaker Verification and Diarization," in *arXiv:2403.19971*, 2024. (Alibaba) [[Paper](https://arxiv.org/abs/2403.19971)] [[Code](https://github.com/modelscope/3D-Speaker)]
- "Target Speech Diarization with Multimodal Prompts," in *Submitted to IEEE/ACM TASLP*, 2024. (NUS) [[Paper](https://arxiv.org/abs/2406.07198)]
- **MFV-KSD**: "Multi-Stage Face-Voice Association Learning with Keynote Speaker Diarization," in *Submitted to ACM MM,* 2024. [[Paper](https://arxiv.org/abs/2407.17902)] [[Code](https://github.com/TaoRuijie/MFV-KSD)]

## Related Spoofing
- "Spoof Diarization: "What Spoofed When" in Partially Spoofed Audio," in *Proc. Interspeech*, 2024. (IITK) [[Paper](https://arxiv.org/pdf/2406.07816)]


## Related TTS
### Speaker Anonymization
- "A Benchmark for Multi-speaker Anonymization," in *Submitted to IEEE/ACM TASLP*, 2024. (SIT) [[Paper](https://arxiv.org/abs/2407.05608)] [[Code](https://github.com/xiaoxiaomiao323/MSA)]

### Singing Diarization
- "Song Data Cleansing for End-to-End Neural Singer Diarization Using Neural Analysis and Synthesis Framework," in *Proc. Interspeech*, 2024. (LY) [[Paper](https://arxiv.org/abs/2406.16315)]


## With Emotion
- "**Speech Emotion Diarization**: Which Emotion Appears When?," in *Proc. ASRU,* 2023. (Zaion) [[Paper](https://arxiv.org/abs/2306.12991)]
- "**EmoDiarize**: Speaker Diarization and Emotion Identification from Speech Signals using Convolutional Neural Networks," in *arxiv:2310.12851*, 2023. [[Paper](https://arxiv.org/abs/2310.12851)]
- "**ED-TTS**: Multi-scale Emotion Modeling using Cross-domain Emotion Diarization for Emotional Speech Synthesis, in *Proc. ICASSP,* 2024. [[Paper](https://arxiv.org/abs/2401.08166)]
---

## Personal VAD
- "**Personal VAD**: Speaker-Conditioned Voice Activity Detection", in *Proc. Odyssey*, 2020. (Google) [[Paper](https://arxiv.org/abs/1908.04284)]
- "**SVVAD**: Personal Voice Activity Detection for Speaker Verification", in *Proc. Interspeech*, 2023. [[Paper](https://arxiv.org/abs/2305.19581)]

## VAD & OSD & SCD
- "Overlapped Speech Detection in Broadcast Streams Using X-vectors," in *Proc. Interspeech*, 2022. [[Paper](https://www.isca-speech.org/archive/interspeech_2022/mateju22_interspeech.html)]
- "Overlapped speech and gender detection with WavLM pre-trained features,"  in *Proc. Interspeech*, 2022. [[Paper](https://www.isca-speech.org/archive/interspeech_2022/lebourdais22_interspeech.html)]
- "Microphone Array Channel Combination Algorithms for Overlapped Speech Detection,"  in *Proc. Interspeech*, 2022. [[Paper](https://www.isca-speech.org/archive/interspeech_2022/mariotte22_interspeech.html)]
- "Multitask Detection of Speaker Changes, Overlapping Speech and Voice Activity Using wav2vec 2.0," in *Proc. ICASSP*, 2023. [[Paper](https://arxiv.org/abs/2210.14755)] [[Code](https://github.com/mkunes/w2v2_audioFrameClassification)]
- "**Semantic VAD**: Low-Latency Voice Activity Detection for Speech Interaction," in *Proc. Interspeech*, 2023. [[Paper](https://arxiv.org/abs/2305.12450)]
- "Joint speech and overlap detection: a benchmark over multiple audio setup and speech domains," in *arxiv:2307.13012*, 2023. [[Paper](https://arxiv.org/abs/2307.13012)]
- "Advancing the study of Large-Scale Learning in Overlapped Speech Detection," in *arXiv:2308.05987*, 2023. [[Paper](https://arxiv.org/abs/2308.05987)]
- "**USM-SCD**: Multilingual Speaker Change Detection Based on Large Pretrained Foundation Models," in *Proc. ICASSP*, 2024. (Google) [[Paper](https://arxiv.org/abs/2309.08023)]
- "Channel-Combination Algorithms for Robust Distant Voice Activity and Overlapped Speech Detection," in *IEEE/ACM TASLP,* 2024. [[Paper](https://arxiv.org/abs/2402.08312)]
- "Speaker Change Detection with Weighted-sum Knowledge Distillation based on Self-supervised Pre-trained Models," in *Proc. Interspeech,* 2024. [[Paper](https://www.isca-archive.org/interspeech_2024/su24_interspeech.html)]
---
## Dataset
- **Voxconverse**: "Spot the conversation: speaker diarisation in the wild", in *Proc. Interspeech*, 2020. (VGG, Naver) [[Paper](https://arxiv.org/abs/2007.01216)] [[Code](https://github.com/joonson/voxconverse)] [[Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)]
- **MSDWild**: Multi-modal Speaker Diarization Dataset in the Wild, in *Proc. Interspeech,* 2020. [[Paper](https://www.isca-speech.org/archive/interspeech_2022/liu22t_interspeech.html)] [[Dataset](https://github.com/X-LANCE/MSDWILD)]
- "LibriMix: An Open-Source Dataset for Generalizable Speech Separation," in *arXiv:2005.11262*, 2020. [[Paper](https://arxiv.org/abs/2005.11262)] [[Code](https://github.com/JorisCos/LibriMix)]
- **Ego4D**: " Around the World in 3,000 Hours of Egocentric Video," in *Proc. CVPR*, 2022. (Meta) [[Paper](https://arxiv.org/abs/2110.07058)] [[Code](https://github.com/EGO4D/audio-visual)] [[Page](https://ego4d-data.org/docs/benchmarks/av-diarization/)]
- **AliMeeting**: "Summary On The ICASSP 2022 Multi-Channel Multi-Party Meeting Transcription Grand Challenge," in *Proc. ICASSP,* 2022. (Alibaba) [[Paper](https://arxiv.org/abs/2202.03647)] [[Dataset](https://www.openslr.org/119)] [[Code](https://github.com/yufan-aslp/AliMeeting)]
- "**VoxBlink**: X-Large Speaker Verification Dataset on Camera", in *Proc. ICASSP,* 2024. [[Paper](https://arxiv.org/abs/2308.07056)] [[Dataset](https://voxblink.github.io/)]
- "NOTSOFAR-1 Challenge: New Datasets, Baseline, and Tasks for Distant Meeting Transcription," in *arXiv:2401.08887,* 2024. (MS) [[Paper](https://arxiv.org/abs/2401.08887)]
- "A Comparative Analysis of Speaker Diarization Models: Creating a Dataset for German Dialectal Speech," in *Proc. ACL,* 2024. [[Paper](https://aclanthology.org/2024.fieldmatters-1.6/)]
- "Conversations in the wild: Data collection, automatic generation and evaluation," in *Computer Speech & Language,* 2025. [[Paper](https://www.sciencedirect.com/science/article/pii/S0885230824000822)]
- "ALLIES: A Speech Corpus for Segmentation, Speaker Diarization, Speech Recognition and Speaker Change Detection," in *Proc. ACL*, 2024. (LIUM) [[Paper](https://aclanthology.org/2024.lrec-main.67/)]""
---
## Tools
- "Gryannote open-source speaker diarization labeling tool," in *Proc. Interspeech (Show and Tell),* 2024. (IRIT) [[Pub.](https://www.isca-archive.org/interspeech_2024/pages24_interspeech.html)] [[Code](https://github.com/clement-pages/gryannote)]
---
## Self-Supervised
- “Self-supervised Speaker Diarization”, in *Proc. Interspeech,* 2022. [[Paper](https://arxiv.org/abs/2204.04166)]
- **CSDA**: "Continual Self-Supervised Domain Adaptation for End-to-End Speaker Diarization", in *Proc. IEEE SLT*, 2022. (CNRS) [[Paper](https://ieeexplore.ieee.org/document/10023195)] [[Code](https://github.com/juanmc2005/CSDA)]
- **DiariZen**: "Leveraging Self-Supervised Learning for Speaker Diarization," in *Proc. ICASSP," 2025. (BUT) [[Paper](https://arxiv.org/abs/2409.09408)] [[Pub.](https://ieeexplore.ieee.org/abstract/document/10889475)] [[Code](https://github.com/BUTSpeechFIT/DiariZen)] [[Review](https://dongkeon.notion.site/DiarZen-1f8eac87949680439b1ce4aeeb211fa9?pvs=4)]

## Semi-Supervised
- "Active Learning Based Constrained Clustering For Speaker Diarization", in *IEEE/ACM TASLP,* 2017. (UT) [[Paper](https://ieeexplore.ieee.org/abstract/document/8030331)]
---
## Measurement
- **BER:** “Balanced Error Rate For Speaker Diarization”, in *Proc. arXiv:2211.04304,* 2022 [[Paper](https://arxiv.org/abs/2211.04304)] [[Code](https://github.com/X-LANCE/BER)]
---

## Child-Adult
- "Robust Self Supervised Speech Embeddings for Child-Adult Classification in Interactions involving Children with Autism," in *Proc. Interspeech*, 2023. [[Paper](https://www.isca-speech.org/archive/interspeech_2023/lahiri23_interspeech.html)]
- "Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions," in *Proc. Interspeech*, 2024. (USC) [[Paper](https://arxiv.org/abs/2406.07890)]


# Challenge
## VoxSRC (VoxCeleb Speaker Recognition Challenge)
### VosSRC-20 Track4
[[Workshop](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/interspeech2020.html)]
- 1st: **Microsoft** [[Tech Report]](https://arxiv.org/abs/2010.11458) [[Video]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2020/participants/mandalorian.mp4)
- 2nd: **BUT** [[Tech Report](https://arxiv.org/abs/2010.11718)] [[Video]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2020/participants/landini.mp4)
- 3rd: **DKU**  [[Tech Report](https://arxiv.org/abs/2010.12731)]

### VosSRC-21 Track4
[[Workshop](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/interspeech2021.html)]
- 1st: **DKU** [[Tech Report]](https://arxiv.org/abs/2109.02002) [[Video]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/DKU-DukeECE-Lenovo.mp4)
- 2nd: **Bytedance** [[Tech Report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/ByteDance_diarization.pdf)] [[Video]](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/Bytedance_SAMI.mp4)
- 3rd: **Tencent**  [[Tech Report](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/Tencent_diarization.pdf)]

### VoxSRC-22 Track4
[[Paper](https://arxiv.org/abs/2302.10248)] [[Workshop](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/interspeech2022.html)]
- 1st: **DKU** [[Tech Report](https://arxiv.org/abs/2210.01677)] [[slide]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/DKU-DukeECE_slides.pdf) [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/DKU-DukeECE_video.mp4)]
- 2nd: **KristonAI** [[Tech Report](https://arxiv.org/abs/2209.11433)] [[slide]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/voxsrc2022_kristonai_track4.pdf) [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/voxsrc2022_kristonai_track4.mp4)]
- 3rd: **GIST** [[Tech Report](https://arxiv.org/abs/2209.10357)] [[slide]](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/gist_slides.pdf) [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/video_aiter.mp4)] [[Reivew](https://velog.io/@fbdp1202/VoxSRC22-%EB%8C%80%ED%9A%8C-%EC%B0%B8%EA%B0%80-%EB%A6%AC%EB%B7%B0-VoxSRC-Challenge-2022-Task-4)]

### VoxSRC-23 Track4
[[Paper]()] [[Workshop](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/interspeech2023.html)]
- 1st: **DKU** [[Tech Report](https://arxiv.org/abs/2308.07595)] [[Slide](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/slides/dku_track4_slides.pdf)] [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/videos/track4_dku.mp4)]
- 2nd: **KrispAI** [[Tech Report](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/reports/krisp_report.pdf)] [[Slide](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/slides/krispai_slides.pdf)] [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/videos/krispai.mp4)]
- 3rd: **Pyannote** [[Tech Report](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/reports/pyannote_report.pdf)] [[Slide](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/slides/pyannote_slides.pdf)] [[Video](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/videos/pyannote_video.mp4)]
- 4th: **GIST** [[Tech Report](https://arxiv.org/abs/2308.07788)]
- **Wespeaker** [[Tech Report](https://arxiv.org/abs/2306.15161)] 

## M2MeT (Multi-channel Multi-party Meeting Transcription Grand Challenge)
### 2022 M2MeT 
[[Introduction Paper](https://arxiv.org/abs/2110.07393?spm=a3c0i.25445127.6257982940.1.111654811kxLMY&file=2110.07393)] [[Summary Paper](https://arxiv.org/abs/2202.03647?spm=a3c0i.25445127.6257982940.2.111654811kxLMY&file=2202.03647)] [[Dataset-AliMeeting](https://www.openslr.org/119)] [[Code](https://github.com/yufan-aslp/AliMeeting)]
- 1st: **DKU** [[Paper](https://arxiv.org/abs/2202.02687)]
- 2nd: **CUHK-TENCENT** [[Paper](https://arxiv.org/abs/2202.01986)]

## MISP (Multimodal Information Based Speech Processing)
### 2022 MISP Track1
[[Introduction Paper](https://arxiv.org/abs/2303.06326)] [[Page](https://mispchallenge.github.io/mispchallenge2022/)] [[Basline Code](https://github.com/mispchallenge/misp2022_baseline/tree/main)]
- 1st: **WHU-Alibaba** [[Paper](https://mispchallenge.github.io/mispchallenge2022/papers/task1/Track1_WHU-ALIBABA.pdf)] [[Review](https://velog.io/@dongkeon/2023-WHU-Alibaba-MISP-2022)]
- 2nd: **SJTU** [[Paper](https://mispchallenge.github.io/mispchallenge2022/papers/task1/Track1_SJTU.pdf)]
- 3rd: **NPU-ASLP** [[Paper](https://mispchallenge.github.io/mispchallenge2022/papers/task1/Track1_NPU-ASLP.pdf)]

## DIHARD
### 2020 DIHARD III
[[Page](https://dihardchallenge.github.io/dihard3/)] [[Paper](https://arxiv.org/abs/2012.01477)] [[Program](https://dihardchallenge.github.io/dihard3workshop/program)] 
#### Track1
- 1st: **USTC** [[Paper](https://arxiv.org/abs/2103.10661)] [Slides](https://dihardchallenge.github.io/dihard3workshop/slide/The%20USTC-NELSLIP%20Systems%20for%20DIHARD%20III%20Challenge.pdf)] [[Video](https://www.youtube.com/watch?v=ijNPazF8EwU&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=7)]
- 2nd: **Hitachi** [[Paper](https://arxiv.org/abs/2102.01363)] [[Slide](https://dihardchallenge.github.io/dihard3workshop/slide/Hitachi-JHU%20System%20for%20the%20Third%20DIHARD%20Speech%20Diarization%20Challenge.pdf)] [[Video](https://www.youtube.com/watch?v=xKGzrF1YEjQ&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=4)]
- 3rd: **Naver Clova** [[Paper](https://dihardchallenge.github.io/dihard3/system_descriptions/dihard3_system_description_team73.pdf)] [[Slide](https://dihardchallenge.github.io/dihard3workshop/slide/NAVER%20Clova%20Submission%20To%20The%20Third%20DIHARD%20Challenge.pdf)] [[Video](https://www.youtube.com/watch?v=X9F2WPWJIR4&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=2)]

#### Track2
- 1st: **USTC-NELSLIP** [[Paper](https://arxiv.org/abs/2103.10661)] [Slides](https://dihardchallenge.github.io/dihard3workshop/slide/The%20USTC-NELSLIP%20Systems%20for%20DIHARD%20III%20Challenge.pdf)] [[Video](https://www.youtube.com/watch?v=ijNPazF8EwU&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=7)]
- 2nd: **Hitachi** [[Paper](https://arxiv.org/abs/2102.01363)] [[Slide](https://dihardchallenge.github.io/dihard3workshop/slide/Hitachi-JHU%20System%20for%20the%20Third%20DIHARD%20Speech%20Diarization%20Challenge.pdf)] [[Video](https://www.youtube.com/watch?v=xKGzrF1YEjQ&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=4)]
- 3rd: **DKU** [[Paper](https://arxiv.org/abs/2102.03649)] [[Slide](https://dihardchallenge.github.io/dihard3workshop/slide/System%20Description%20for%20Team%20DKU-Duke-Lenovo.pdf)] [[Video](https://www.youtube.com/watch?v=FF5QAm6Jgy8&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=6)]

#### Etc.
"End-to-end speaker diarization system for the third dihard challenge system description," in *DIHARD III Tech. Report*, 2021 

## The DISPLACE Challenge 2023 
- "**The DISPLACE Challenge 2023** - DIarization of SPeaker and LAnguage in Conversational Environments," in *Proc. Interspeech*, 2023. [[Paper](https://www.isca-speech.org/archive/interspeech_2023/baghel23_interspeech.html)] [[Page](https://displace2023.github.io/)]
- "The SpeeD--ZevoTech submission at DISPLACE 2023," in *Proc. Interspeech*, 2023. [[Paper](https://www.isca-speech.org/archive/interspeech_2023/pirlogeanu23_interspeech.html)]

## MERLIon CCS Challenge 2023
- "**MERLIon CCS Challenge**: A English-Mandarin code-switching child-directed speech corpus for language identification and diarization," in *Proc. Interspeech*, 2023. [[Paper](https://www.isca-speech.org/archive/interspeech_2023/baghel23_interspeech.html)] [[Page](https://sites.google.com/view/merlion-ccs-challenge/)]

## CHiME-6
[[Overview](https://chimechallenge.github.io/chime6/overview.html)] [[Paper](https://arxiv.org/abs/2004.09249)]

## ICMC-ASR Grand Challenge (ICASSP2024)
- "ICMC-ASR: The ICASSP 2024 In-Car Multi-Channel Automatic Speech Recognition Challenge," 2023. [[Paper](https://arxiv.org/abs/2401.03473)]
- "The NUS-HLT System for ICASSP2024 ICMC-ASR Grand Challenge," in *Technical Report*, 2023. [[Paper](https://arxiv.org/abs/2312.16002)]

## The Second DISPLACE
- "The Second DISPLACE Challenge : DIarization of SPeaker and LAnguage in Conversational Environments," in *Proc. Interspeech*, 2024. [[Paper](https://arxiv.org/abs/2406.09494)]"

## CHiME-8
- "The CHiME-8 DASR Challenge for Generalizable and Array Agnostic Distant Automatic Speech Recognition and Diarization," 2024. [[Paper](https://arxiv.org/abs/2407.16447)]


# Other Awesome-list 
https://github.com/wq2012/awesome-diarization

https://github.com/xyxCalvin/awesome-speaker-diarization
