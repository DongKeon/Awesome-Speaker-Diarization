# 🎤 Awesome Speaker Diarization [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> 325+ papers · 61 code repos · Updated Mar 2026
>
> 📄 Paper · 💻 Code · 📝 Review · 🎬 Video/Demo · 📊 Slides · 🔗 Other


📚 [EEND](#eend) · 🎯 [TS-VAD](#ts-vad) · 🔗 [Clustering](#clustering) · ⚡ [Online](#online) · 🗣️ [ASR](#asr) · 👁️ [Vision](#vision) · 📊 [Dataset](#dataset) · 🏆 [Challenge](#challenge)

---

## Table of Contents

**Core Diarization Methods**
- [📚 EEND (End-to-End)](#eend) (52) · [🎯 TS-VAD](#ts-vad) (17) · [🔗 Clustering](#clustering) (21) · [🧩 Embedding](#embedding) (14) · [📐 VBx & HMM](#vbx-hmm) (24) · [📊 Scoring](#scoring) (3)

**Architecture Extensions**
- [⚡ Online](#online) (16) · [📡 Multi-Channel](#multi-channel) (14) · [🔀 Separation / Extraction](#separation) (12)

**Cross-Modal & Integration**
- [🗣️ With ASR](#asr) (27) · [👁️ With Vision](#vision) (21) · [💬 With NLP / LLM](#nlp-llm) (10) · [🌐 Language](#language) (2) · [😊 Emotion](#emotion) (3)

**Related Tasks**
- [📈 VAD / OSD / SCD](#vad-osd-scd) (10) · [🔊 Speaker Recognition](#speaker-rec) (7) · [🎙️ Personal VAD](#personal-vad) (2) · [🛡️ Spoofing](#spoofing) · [🔉 TTS](#tts) · [👶 Child-Adult](#child-adult) (2)

**Resources & Training**
- [📊 Dataset](#dataset) (11) · [🛠️ Tools](#tools) · [📝 Reviews](#reviews) (3) · [📏 Measurement](#measurement) (3) · [🔄 Self-Supervised](#self-supervised) (5) · [🔃 Semi-Supervised](#semi-supervised)

**[🏆 Challenge](#challenge)** — VoxSRC · M2MeT · MISP · DIHARD · DISPLACE · CHiME


---

<a id="overview"></a>

## 📖 Overview — 1 paper

<details open>
<summary><b>2020 (1 paper)</b></summary>

- **DIHARD Keynote Session:** The yellow brick road of diarization, challenges and other neural paths [📊](https://dihardchallenge.github.io/dihard3workshop/slide/The%20yellow%20brick%20road%20of%20diarization,%20challenges%20and%20other%20neural%20paths.pdf) [🎬](https://www.youtube.com/watch?v=_usbos-SJlg&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=10)

</details>


---

<a id="reviews"></a>

## 📝 Reviews — 3 papers

<details open>
<summary><b>🔥 2023-2024 (3 papers)</b></summary>

- "Overview of Speaker Modeling and Its Applications: From the Lens of Deep Speaker Representation Learning," in *Submitted to IEEE/ACM TASLP*, 2024. [📄](https://arxiv.org/abs/2407.15188)
- “A review of speaker diarization: Recent advances with deep learning”, in *Computer Speech & Language, Volume 72,* 2023. (USC) [📄](https://arxiv.org/abs/2101.09624)
- "An Experimental Review of Speaker Diarization methods with application to Two-Speaker Conversational Telephone Speech recordings", in *Computer Speech & Language,* 2023. [📄](https://arxiv.org/abs/2305.18074)

</details>


---

<a id="eend"></a>

## 📚 EEND (End-to-End Neural Diarization)-based — 52 papers

<details open>
<summary><b>🔥 2025 (9 papers)</b></summary>

- "Mamba-based Segmentation Model for Speaker Diarization," *Proc. ICASSP,* 2025. (NTT) [📄](https://arxiv.org/abs/2410.06459) [💻](https://github.com/nttcslab-sp/mamba-diarization)
- **LS-EEND**: "Long-Form Streaming End-to-End Neural Diarization with Online Attractor Extraction," in *IEEE/ACM TASLP,* 2025. (Westlake) [📄](https://arxiv.org/abs/2410.06670) [💻](https://github.com/Audio-WestlakeU/FS-EEND)
- "Pushing the Limits of End-to-End Diarization," in *Proc. Interspeech,* 2025. [📄](https://www.isca-archive.org/interspeech_2025/broughton25_interspeech.pdf)
- **O-EENC-SD**: "Efficient Online End-to-End Neural Clustering for Speaker Diarization," in *arXiv:2512.15229,* 2025. [📄](https://arxiv.org/abs/2512.15229)
- **VBx-EEND-VC**: "VBx for End-to-End Neural and Clustering-based Diarization," in *arXiv:2510.19572,* 2025. (BUT) [📄](https://arxiv.org/abs/2510.19572)
- "Improving Neural Diarization through Speaker Attribute Attractors and Local Dependency Modeling," in *arXiv:2506.05593,* 2025. (OSU) [📄](https://arxiv.org/abs/2506.05593)
- **DLF-EEND**: "Dynamic Layer Fusion for End-to-End Speaker Diarization," in *Proc. Interspeech,* 2025. [📄](https://www.isca-archive.org/interspeech_2025/kim25l_interspeech.pdf)
- "End-to-End Diarization utilizing Attractor Deep Clustering," in *Proc. Interspeech,* 2025. (JHU, OSU) [📄](https://arxiv.org/abs/2506.11090)
- "Pretraining Multi-Speaker Identification for Neural Speaker Diarization," in *Proc. Interspeech,* 2025. (NTT) [📄](https://arxiv.org/abs/2505.24545)

</details>

<details>
<summary><b>2024 (9 papers)</b></summary>

- "NTT speaker diarization system for CHiME-7: multi-domain, multi-microphone End-to-end and vector clustering diarization," in *Proc. ICASSP*, 2024. (NTT) [📄](https://arxiv.org/abs/2309.12656)
- **AED-EEND-EE**: "Attention-based Encoder-Decoder End-to-End Neural Diarization with Embedding Enhancer," in *IEEE/ACM TASLP*, 2024. (SJTU) [📄](https://arxiv.org/abs/2309.06672) [📝](https://www.notion.so/AED-EEND-EE-903d475a735c46218667a25ed45d4e74)
- "**DiaPer**: End-to-End Neural Diarization with Perceiver-Based Attractors," in *IEEE/ACM TASLP,* 2024. (BUT) [📄](https://arxiv.org/abs/2312.04324) [💻](https://github.com/BUTSpeechFIT/DiaPer) [📝](https://dongkeon.notion.site/2024-DiaPer-Submitted-TASLP-83fbbd4b8e8645d7a1fe7a08069334ea?pvs=4)
- "**EEND-DEMUX**: End-to-End Neural Speaker Diarization via Demultiplexed Speaker Embeddings," in *Submitted to IEEE SPL,* 2024. (SNU) [📄](https://arxiv.org/abs/2312.06065) [📝](https://dongkeon.notion.site/2024-EEND-DEMUX-Submitted-SPL-4bfa79521cc74a78a14e5fc148a7c9c1?pvs=4)
- "**EEND-M2F**: Masked-attention mask transformers for speaker diarization," in *Proc. Interspeech,* 2024. (Fano Labs) [📄](https://arxiv.org/abs/2401.12600) [📄](https://www.isca-archive.org/interspeech_2024/harkonen24_interspeech.html) [📝](https://dongkeon.notion.site/2024-EEND-M2F-arXiv-8bb1ec11cc2c463cab372cd6cec10318?pvs=4)
- **EEND-NAA (2)**: "End-to-End Neural Speaker Diarization with Non-Autoregressive Attractors", in *IEEE/ACM TASLP*, 2024. (JHU) [📄](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10629182) [📝](https://dongkeon.notion.site/EEND-NAA-13beac879496807eb0bced8de53e91c1?pvs=4)
- "From Modular to End-to-End Speaker Diarization," *Ph.D. thesis*, 2024. (BUT) [📄](https://arxiv.org/abs/2407.08752)
- "On the calibration of powerset speaker diarization models,"  in *Proc. Interspeech,* 2024. (IRIT) [📄](https://arxiv.org/abs/2409.15885) [📄](https://www.isca-archive.org/interspeech_2024/plaquet24_interspeech.html) [💻](https://github.com/FrenchKrab/IS2024-powerset-calibration) [📝](https://dongkeon.notion.site/PSE-calibration-1faeac8794968000b5bfd6c900638bba?pvs=4)
- **Local-global EEND**: "Speakers Unembedded: Embedding-free Approach to Long-form Neural Diarization," in *Proc. Interspeech,* 2024. (Amazon) [📄](https://www.isca-archive.org/interspeech_2024/li24x_interspeech.html) [📝](https://dongkeon.notion.site/Local-global-EEND-2b0a3eb900644c31baf0756da0ca4b5d?pvs=4)

</details>

<details>
<summary><b>2023 (11 papers)</b></summary>

- "Improving Transformer-based End-to-End Speaker Diarization by Assigning Auxiliary Losses to Attention Heads", in *Proc. ICASSP,* 2023. (HU) [📄](https://arxiv.org/abs/2303.01192)
- **EEND-NA**: “Neural Diarization with Non-Autoregressive Intermediate Attractors”, in *Proc. ICASSP,* 2023. (LINE)  [📄](https://arxiv.org/abs/2303.06806)
- "**TOLD**: A Novel Two-Stage Overlap-Aware Framework for Speaker Diarization", in *Proc. ICASSP*, 2023. (Alibaba) [📄](https://arxiv.org/abs/2303.05397) [💻](https://github.com/alibaba-damo-academy/FunASR)
- "Improving End-to-End Neural Diarization Using Conversational Summary Representations", in *Proc. Interspeech*, 2023. (Fano Labs) [📄](https://arxiv.org/abs/2306.13863)
- **AED-EEND**: “Attention-based Encoder-Decoder Network for End-to-End Neural Speaker Diarization with Target Speaker Attractor”, in *Proc. Interspeech,* 2023. (SJTU) [📄](https://www.isca-speech.org/archive/interspeech_2023/chen23n_interspeech.html) [📝](https://www.notion.so/AED-EEND-EE-903d475a735c46218667a25ed45d4e74)
- "Self-Distillation into Self-Attention Heads for Improving Transformer-based End-to-End Neural Speaker Diarization", in *Proc. Interspeech*, 2023. (HU) [📄](https://www.isca-speech.org/archive/interspeech_2023/jeoung23_interspeech.html)
- "Powerset Multi-class Cross Entropy Loss for Neural Speaker Diarization", in *Proc. Interspeech*, 2023. (Pyannote) [📄](https://www.isca-speech.org/archive/interspeech_2023/plaquet23_interspeech.html) [💻](https://github.com/FrenchKrab/IS2023-powerset-diarization)
- "End-to-End Neural Speaker Diarization with Absolute Speaker Loss", in  *Proc. Interspeech*, 2023. (Pyannote) [📄](https://www.isca-speech.org/archive/interspeech_2023/wang23g_interspeech.html)
- "Blueprint Separable Subsampling and Aggregate Feature Conformer-Based End-to-End Neural Diarization", in *Electronics*, 2023. [📄](https://www.mdpi.com/2079-9292/12/19/4118)
- **EEND-TA**: "Transformer Attractors for Robust and Efficient End-to-End Neural Diarization," in *Proc. ASRU,* 2023. (Fano Labs) [📄](https://arxiv.org/abs/2312.06253)
- "Robust End-to-End Diarization with Domain Adaptive Training and Multi-Task Learning," in *Proc. ASRU,* 2023. (Fano Labs) [📄](https://arxiv.org/abs/2312.07136)

</details>

<details>
<summary><b>2022 (10 papers)</b></summary>

- **EEND-EDA (2)**: “Encoder-Decoder Based Attractor Calculation for End-to-End Neural Diarization”, in *IEEE/ACM TASLP,* 2022. (Hitachi) [📄](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9741374) [📝](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers) [💻](https://github.com/butspeechfit/eend)
- "**DIVE**: End-to-end Speech Diarization via Iterative Speaker Embedding", in *Proc. ICASSP*, 2022. (Google) [📄](https://arxiv.org/abs/2105.13802)
- **RX-EEND**: “Auxiliary Loss of Transformer with Residual Connection for End-to-End Speaker Diarization”, in *Proc. ICASSP,* 2022. (GIST) [📄](https://arxiv.org/abs/2110.07116) [📝](https://velog.io/@fbdp1202/RX-EEND-%EB%A6%AC%EB%B7%B0-Auxiliary-Loss-of-Transformer-with-Residual-connection-For-End-to-end-Speaker-Diarization)
- "End-to-end speaker diarization with transformer", in *Proc. arXiv*, 2022. [📄](https://arxiv.org/abs/2112.07463)
- **EEND-VC-iGMM**: "Tight integration of neural and clustering-based diarization through deep unfolding of infinite Gaussian mixture model", in *Proc. ICASSP*, 2022. (NTT) [📄](https://arxiv.org/abs/2202.06524)
- **EDA-RC**: "Robust End-to-end Speaker Diarization with Generic Neural Clustering", in *Proc. Interspeech*, 2022. (SJTU) [📄](https://arxiv.org/abs/2204.08164)
- **EEND-NAA**: "End-to-End Neural Speaker Diarization with an Iterative Refinement of Non-Autoregressive Attention-based Attractors", in *Proc. Interspeech*, 2022. (JHU) [📄](https://www.isca-speech.org/archive/interspeech_2022/rybicka22_interspeech.html) [📝](https://dongkeon.notion.site/EEND-NAA-13beac879496807eb0bced8de53e91c1?pvs=4)
- **Graph-PIT**: "Utterance-by-utterance overlap-aware neural diarization with Graph-PIT", in *Proc. Interspeech*, 2022. (NTT) [📄](https://arxiv.org/abs/2207.13888) [💻](https://github.com/fgnt/graph_pit)
- "Efficient Transformers for End-to-End Neural Speaker Diarization", in *Proc. IberSPEECH*, 2022. [📄](https://www.isca-speech.org/archive/iberspeech_2022/izquierdodelalamo22_iberspeech.html)
- **EEND-EDA-SpkAtt**: "Towards End-to-end Speaker Diarization in the Wild", in *arXiv:2211.01299v1,* 2022. [📄](https://arxiv.org/abs/2211.01299v1)

</details>

<details>
<summary><b>2021 (7 papers)</b></summary>

- **CB-EEND**: "End-to-end Neural Diarization: From Transformer to Conformer", in *Proc. Interspeech*, 2021. (Amazon) [📄](https://arxiv.org/abs/2106.07167) [📝](https://velog.io/@fbdp1202/CB-EEND-%EB%A6%AC%EB%B7%B0-End-to-end-Neural-Diarization-From-Transformer-to-Conformer)
- **TDCN-SA**: "End-to-End Diarization for Variable Number of Speakers with Local-Global Networks and Discriminative Speaker Embeddings", in *Proc. ICASSP*, 2021. (Google) [📄](https://arxiv.org/abs/2105.02096) [📝](https://velog.io/@fbdp1202/TDCN-SA-%EB%A6%AC%EB%B7%B0-End-to-End-Diarization-for-Variable-Number-of-Speakers-with-Local-Global-Networks-and-Discriminative-Speaker-Embeddings)
- "End-to-End Speaker Diarization Conditioned on Speech Activity and Overlap Detection", in *Proc. IEEE SLT*, 2021. (Hitachi) [📄](https://arxiv.org/abs/2106.04078)
- **EEND-VC (1)**: "Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds", in *Proc. ICASSP*, 2021. (NTT) [📄](https://arxiv.org/abs/2010.13366) [📝](https://velog.io/@fbdp1202/EEND-vector-clustering-%EB%A6%AC%EB%B7%B0-Integrating-end-to-end-neural-and-clustering-based-diarization-Getting-the-best-of-both-world) [💻](https://github.com/nttcslab-sp/EEND-vector-clustering)
- **EEND-VC (2)**: "Advances in integration of end-to-end neural and clustering-based diarization for real conversational speech", in *Proc. Interspeech*, 2021. (NTT) [📄](https://arxiv.org/abs/2105.09040) [📝](https://velog.io/@fbdp1202/EEND-vector-clustering-%EB%A6%AC%EB%B7%B0-Integrating-end-to-end-neural-and-clustering-based-diarization-Getting-the-best-of-both-world) [💻](https://github.com/nttcslab-sp/EEND-vector-clustering)
- "Robust End-to-End Speaker Diarization with Conformer and Additive Margin Penalty," in *Proc. Interspeech*, 2021. (Fano Labs) [📄](https://www.isca-archive.org/interspeech_2021/leung21_interspeech.html)
- **EEND-GLA**: "Towards Neural Diarization for Unlimited Numbers of Speakers Using Global and Local Attractors", in *Proc. ASRU*, 2021. (Hitachi) [📄](https://arxiv.org/abs/2107.01545) [📝](https://velog.io/@fbdp1202/EEND-EDA-Clustering-%EB%A6%AC%EB%B7%B0-Towards-Neural-Diarization-for-Unlimited-Numbers-of-Speakers-using-Global-and-Local-Attractors)

</details>

<details>
<summary><b>2020 (3 papers)</b></summary>

- **SA-EEND (2)**: “End-to-End Neural Diarization: Reformulating Speaker Diarization as Simple Multi-label Classification”, in *arXiv:2003.02966,* 2020. (Hitachi) [📄](https://arxiv.org/abs/2003.02966) [📝](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)
- **SC-EEND**: "Neural Speaker Diarization with Speaker-Wise Chain Rule", in *arXiv:2006.01796*, 2020. (Hitachi) [📄](https://arxiv.org/abs/2006.01796) [📝](https://velog.io/@fbdp1202/SC-EEND-%EB%A6%AC%EB%B7%B0-Neural-Speaker-Diarization-with-Speaker-Wise-Chain-Rule)
- **EEND-EDA (1)**: “End-to-End Speaker Diarization for an Unknown Number of Speakers with Encoder-Decoder Based Attractors”, in *Proc. Interspeech,* 2020. (Hitachi) [📄](https://arxiv.org/abs/2005.09921) [📝](https://velog.io/@fbdp1202/EEND-EDA-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-for-an-Unknown-Number-of-Speakers) [💻](https://github.com/butspeechfit/eend)

</details>

<details>
<summary><b>2023 (1 paper)</b></summary>

- **EEND-IAAE**: "End-to-end neural speaker diarization with an iterative adaptive attractor estimation," in *Neural Networks, Elsevier*. [📄](https://www.sciencedirect.com/science/article/pii/S089360802300401X) [💻](https://github.com/HaoFengyuan/EEND-IAAE)

</details>

<details>
<summary><b>2019 (2 papers)</b></summary>

- **BLSTM-EEND**: "End-to-End Neural Speaker Diarization with Permutation-Free Objectives", in *Proc. Interspeech*, 2019. (Hitachi) [📄](https://arxiv.org/abs/1909.05952)
- **SA-EEND (1)**: “End-to-End Neural Speaker Diarization with Self-attention”, in *Proc. ASRU*, 2019. (Hitachi) [📄](https://ieeexplore.ieee.org/abstract/document/9003959) [💻](https://github.com/hitachi-speech/EEND) [💻](https://github.com/Xflick/EEND_PyTorch) [📝](https://velog.io/@fbdp1202/SA-EEND-%EB%A6%AC%EB%B7%B0-End-to-End-Neural-Speaker-Diarization-with-Self-Attention)

</details>

<a id="eend-speaker-info"></a>

### 📌 Related Speaker information — 2 papers

<details open>
<summary><b>🔥 2024 (2 papers)</b></summary>

  - "Do End-to-End Neural Diarization Attractors Need to Encode Speaker Characteristic Information?," in *Proc. Odyssey*, 2024. [📄](https://www.isca-archive.org/odyssey_2024/zhang24_odyssey.html)
  - "Leveraging Speaker Embeddings in End-to-End Neural Diarization for Two-Speaker Scenarios," in *Proc. Odyssey,* 2024. [📄](https://www.isca-archive.org/odyssey_2024/alvareztrejos24_odyssey.html)

</details>

<a id="eend-simulated"></a>

### 📊 Simulated Dataset — 8 papers

<details open>
<summary><b>🔥 2023-2024 (4 papers)</b></summary>

  - "Enhancing low-latency speaker diarization with spatial dictionary learning," in *Proc. ICASSP*, 2024.  (NTU) [📄](https://ieeexplore.ieee.org/document/10446666) [📊](https://sigport.org/sites/default/files/docs/ENHANCING%20LOW-LATENCY%20SPEAKER%20DIARIZATION%20WITH%20SPATIAL%20DICTIONARY%20LEARNING.pdf)
  - "Improving Neural Diarization through Speaker Attribute Attractors and Local Dependency Modeling," in *Proc. ICASSP*, 2024. (OSU) [📄](https://ieeexplore.ieee.org/document/10446213)
  - "Multi-Speaker and Wide-Band Simulated Conversations as Training Data for End-to-End Neural Diarization", in *Proc. ICASSP*, 2023. (BUT) [📄](https://arxiv.org/abs/2211.06750) [💻](https://github.com/BUTSpeechFIT/EEND_dataprep) [📝](https://velog.io/@dongkeon/2023-Simulated-Conversations-ICASSP)
  - "Property-Aware Multi-Speaker Data Simulation: A Probabilistic Modelling Technique for Synthetic Data Generation," in *CHiME-7 Workshop*, 2023. (NVIDIA) [📄](https://arxiv.org/abs/2310.12371)

</details>

<details>
<summary><b>2022 (3 papers)</b></summary>

  - “From simulated mixtures to simulated conversations as training data for end-to-end neural diarization” , in *Proc. Interspeech*, 2022. (BUT) [📄](https://arxiv.org/abs/2204.00890) [💻](https://github.com/BUTSpeechFIT/EEND_dataprep) [📝](https://velog.io/@dongkeon/2023-Simulated-Conversations-ICASSP)
  - **Markov selection**: “Improving the naturalness of simulated conversations for end-to-end neural diarization”, in *Proc. Odyssey*, 2022. (Hitachi) [📄](https://arxiv.org/abs/2204.11232)
  - **EEND-EDA-SpkAtt**: "Towards End-to-end Speaker Diarization in the Wild", in *arXiv:2211.01299v1,* 2022. [📄](https://arxiv.org/abs/2211.01299v1)

</details>

<details>
<summary><b>2019 (1 paper)</b></summary>

  - **Concat-and-sum**: “End-to-end neuarl speaker diarization with permuation-free objectives”, in *Proc. Interspeech*, 2019. [📄](https://arxiv.org/abs/1909.05952)

</details>

<a id="eend-post"></a>

### 📌 Post-Processing — 3 papers

<details open>
<summary><b>🔥 2021-2024 (3 papers)</b></summary>

  - "**DiaCorrect**: Error Correction Back-end For Speaker Diarization," in *Proc. ICASSP*, 2024. (BUT) [📄](https://arxiv.org/abs/2309.08377) [💻](https://github.com/BUTSpeechFIT/diacorrect)
  - **EENDasP**: "End-to-End Speaker Diarization as Post-Processing", in *Proc. ICASSP*, 2021. (Hitachi) [📄](https://arxiv.org/abs/2012.10055) [📝](https://velog.io/@fbdp1202/EEND-asp-%EB%A6%AC%EB%B7%B0-End-to-End-Speaker-Diarization-as-Post-Processing) [💻](https://github.com/DongKeon/EENDasP)
  - **Dover-Lap**: "DOVER-Lap: A Method for Combining Overlap-aware Diarization Outputs", in *Proc. IEEE SLT*, 2021. (JHU) [📄](https://arxiv.org/abs/2011.01997) [📝](https://velog.io/@fbdp1202/Dover-lap-%EB%A6%AC%EB%B7%B0-A-method-for-combining-overlap-aware-diarization-outputs) [💻](https://github.com/desh2608/dover-lap)

</details>


---

<a id="ts-vad"></a>

## 🎯 Using Target Speaker Embedding — 17 papers

<details open>
<summary><b>🔥 2025 (4 papers)</b></summary>

- "Noise-Robust Target-Speaker Voice Activity Detection Through Self-Supervised Pretraining," in *Proc. ICASSP,* 2025. [📄](https://arxiv.org/abs/2501.03184)
- **MIMO-TSVAD**: "Multi-Input Multi-Output Target-Speaker Voice Activity Detection For Unified, Flexible, and Robust Audio-Visual Speaker Diarization," in *IEEE/ACM TASLP,* 2025. (DKU) [📄](https://arxiv.org/abs/2401.08052)
- "Mitigating Non-Target Speaker Bias in Guided Speaker Embedding," in *Proc. Interspeech,* 2025. (NTT) [📄](https://arxiv.org/abs/2506.12500)
- "Diarization-Guided Multi-Speaker Embeddings," in *Proc. Interspeech,* 2025. (Pyannote) [📄](https://www.isca-archive.org/interspeech_2025/kalda25_interspeech.pdf)

</details>

<details>
<summary><b>2024 (3 papers)</b></summary>

- **NSD-MS2S**: "Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding with Sequence-to-Sequence Architecture, " in *Proc. ICASSP*, 2024. (USTC) [📄](https://arxiv.org/abs/2309.09180) [💻](https://github.com/liyunlongaaa/NSD-MS2S)
- **PET-TSVAD**: "Profile-Error-Tolerant Target-Speaker Voice Activity Detection," in *Proc. ICASSP*, 2024. (Microsoft) [📄](https://arxiv.org/abs/2309.12521)
- **Flow-TSVAD**: "Target-Speaker Voice Activity Detection via Latent Flow Matching," in *arXiv:2409.04859,* 2024. (DKU) [📄](https://arxiv.org/abs/2409.04859)

</details>

<details>
<summary><b>2023 (4 papers)</b></summary>

- **EDA-TS-VAD**: “Target Speaker Voice Activity Detection with Transformers and Its Integration with End-to-End Neural Diarization”, in *Proc. ICASSP*, 2023. (Microsoft) [📄](https://arxiv.org/abs/2208.13085)
- **Seq2Seq-TS-VAD**: “Target-Speaker Voice Activity Detection via Sequence-to-Sequence Prediction”, in *Proc. ICASSP,* 2023. (DKU) [📄](https://arxiv.org/abs/2210.16127) [📝](https://velog.io/@dongkeon/2023-Seq2Seq-TS-VAD)
- **QM-TS-VAD**: "Unsupervised Adaptation with Quality-Aware Masking to Improve Target-Speaker Voice Activity Detection for Speaker Diarization", in *Proc. Interspeech,* 2023. (USTC) [📄](https://www.isca-speech.org/archive/interspeech_2023/niu23_interspeech.html)
- "**ANSD-MA-MSE**: Adaptive Neural Speaker Diarization Using Memory-Aware Multi-Speaker Embedding," in *IEEE/ACM TASLP*, 2023. (USTC) [📄](https://ieeexplore.ieee.org/document/10093997) [💻](https://github.com/Maokui-He/NSD-MA-MSE/tree/main)

</details>

<details>
<summary><b>2022 (3 papers)</b></summary>

- **SEND (2)**: "Speaker Embedding-aware Neural Diarization: an Efficient Framework for Overlapping Speech Diarization in Meeting Scenarios," in *arXiv:2203.09767*, 2022 (Alibaba) [📄](https://arxiv.org/abs/2203.09767)
- **MTEAD**: "Multi-target Filter and Detector for Unknown-number Speaker Diarization", in *IEEE SPL*, 2022. [📄](https://arxiv.org/abs/2203.16007)
- **SOND**: "Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis", in *Proc. EMNLP*, 2022. (Alibaba) [📄](https://arxiv.org/abs/2211.10243) [💻](https://github.com/alibaba-damo-academy/FunASR)

</details>

<details>
<summary><b>2020-2021 (3 papers)</b></summary>

- **SEND (1)**: "Speaker Embedding-aware Neural Diarization for Flexible Number of Speakers with Textual Information," in *arXiv:2111.13694*, 2021. (Alibaba) [📄](https://arxiv.org/abs/2111.13694)
- **TS-VAD**: "Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario", in *Proc. Interspeech*, 2020. [📄](https://arxiv.org/abs/2005.07272) [💻](https://github.com/dodohow1011/TS-VAD) [📊](https://desh2608.github.io/static/ppt/ts-vad.pdf)
- “The STC system for the CHiME-6 challenge,” in *CHiME Workshop*, 2020. [📄](https://www.isca-speech.org/archive/chime_2020/medennikov20_chime.html)

</details>


---

<a id="target-speech"></a>

## 🎯 Target Speech Diarization — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- **PTSD**: "Prompt-driven Target Speech Diarization," in *Proc. ICASSP*, 2024. (NUS) [📄](https://arxiv.org/abs/2310.14823)

</details>


---

<a id="separation"></a>

## 🔀 With Separation or Target Speaker Extraction — 12 papers

<details open>
<summary><b>🔥 2025 (3 papers)</b></summary>

- "Robust Target Speaker Diarization and Separation via Augmented Speaker Embedding Sampling," in *Proc. Interspeech,* 2025. [📄](https://arxiv.org/abs/2508.06393)
- **S2SND**: "Sequence-to-Sequence Neural Diarization with Automatic Speaker Detection and Representation," in *IEEE/ACM TASLP,* 2025. (DKU) [📄](https://arxiv.org/abs/2411.13849)
- "Exploring Speaker Diarization with Mixture of Experts," in *arXiv:2506.14750,* 2025. (USTC) [📄](https://arxiv.org/abs/2506.14750)

</details>

<details>
<summary><b>2024 (7 papers)</b></summary>

- "**TS-SEP**: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings", in *IEEE/ACM TASLP*, 2024. [📄](https://arxiv.org/abs/2303.03849)
- "Continuous Target Speech Extraction: Enhancing Personalized Diarization and Extraction on Complex Recordings," in *arXiv:2401.15993*, 2024. (Tencent) [📄](https://arxiv.org/abs/2401.15993) [🎬](https://herbhezhao.github.io/Continuous-Target-Speech-Extraction/)
- "**PixIT**: Joint Training of Speaker Diarization and Speech Separation from Real-world Multi-speaker Recordings," in *Proc. Odyssey*, 2024. [📄](https://arxiv.org/abs/2403.02288) [💻](https://github.com/joonaskalda/PixIT)
- **MC-EEND**: "Multi-channel Conversational Speaker Separation via Neural Diarization," in *IEEE/ACM TASLP,* 2024. (OSU) [📄](https://arxiv.org/abs/2311.08630)
- "**USED**: Universal Speaker Extraction and Diarization," in *submitted to IEEE/ACM TASLP*, 2024. (CUHK) [📄](https://arxiv.org/abs/2309.10674) [🎬](https://ajyy.github.io/demo/USED/) [🔗](https://github.com/msinanyildirim/USED-splits) [📝](https://dongkeon.notion.site/USED-Universal-Speaker-Extraction-and-Diarization-dcaf0e22ec334286b188ab5561bdbd27?pvs=4)
- "Neural Blind Source Separation and Diarization for Distant Speech Recognition," in *Proc. Interspeech*, 2024. (AIST) [📄](https://arxiv.org/pdf/2406.08396)
- "TalTech-IRIT-LIS Speaker and Language Diarization Systems for DISPLACE 2024," in *Proc. Interspeech,* 2024. (Pyannote) [📄](https://arxiv.org/abs/2407.12743)

</details>

<details>
<summary><b>2021-2022 (2 papers)</b></summary>

- **EEND-SS**: "Joint End-to-End Neural Speaker Diarization and Speech Separation for Flexible Number of Speakers”, in *Proc. SLT,* 2022. (CMU) [📄](https://arxiv.org/abs/2203.17068) [📝](https://dongkeon.notion.site/EEND-SS-Joint-End-to-End-Neural-Speaker-Diarization-and-Speech-Separation-for-Flexible-Number-of-Sp-32a6a76f796341ca972b3959c1b7311d?pvs=4)
- "Integration of speech separation, diarization, and recognition for multi-speaker meetings: System description, comparison, and analysis," in *Proc. SLT,* 2021. (JHU) [📄](https://arxiv.org/abs/2011.02014) [🔗](https://desh2608.github.io/pages/jsalt/) [📝](https://dongkeon.notion.site/Integration-of-speech-separation-diarization-and-recognition-for-multi-speaker-meetings-System-de-f1d2574672834743a94c152b536d78b6?pvs=4)

</details>


---

<a id="multi-channel"></a>

## 📡 Multi-Channel — 14 papers

<details open>
<summary><b>🔥 2025 (5 papers)</b></summary>

- "Multi-channel Speaker Counting for EEND-VC-based Speaker Diarization on Multi-domain Conversation," in *Proc. ICASSP,* 2025. (NTT) [📄](https://ieeexplore.ieee.org/abstract/document/10888681) [📝](https://dongkeon.notion.site/Multi-channel-Speaker-Counting-1faeac879496809b9075e79580eb9a6e?pvs=4)
- "Spatially Aware Self-Supervised Models for Multi-Channel Neural Speaker Diarization," in *arXiv:2510.14551,* 2025. (BUT) [📄](https://arxiv.org/abs/2510.14551) [💻](https://github.com/BUTSpeechFIT/DiariZen)
- "Multi-Channel Sequence-to-Sequence Neural Diarization for The MISP 2025 Challenge," in *arXiv:2505.16387,* 2025. [📄](https://arxiv.org/abs/2505.16387)
- "Incorporating Spatial Cues in Modular Speaker Diarization for Multi-channel Multi-party Meetings," in *Proc. ICASSP,* 2025. [📄](https://arxiv.org/abs/2409.16803)
- "Spatio-Spectral Diarization of Meetings by Combining TDOA-based Segmentation and Speaker Embedding-based Clustering," in *Proc. Interspeech,* 2025. [📄](https://arxiv.org/abs/2506.16228)

</details>

<details>
<summary><b>2024 (5 papers)</b></summary>

- "**UniX-Encoder**: A Universal X-Channel Speech Encoder for Ad-Hoc Microphone Array Speech Processing," in *arXiv:2310.16367*, 2024. (JHU, Tencent) [📄](https://arxiv.org/abs/2310.16367)
- "Channel-Combination Algorithms for Robust Distant Voice Activity and Overlapped Speech Detection," in *IEEE/ACM TASLP,* 2024. [📄](https://arxiv.org/abs/2402.08312)
- "A Spatial Long-Term Iterative Mask Estimation Approach for Multi-Channel Speaker Diarization and Speech Recognition," in *Proc. ICASSP*, 2024. (USTC) [📄](https://ieeexplore.ieee.org/document/10446168)
- **MC-EEND**: "Multi-channel Conversational Speaker Separation via Neural Diarization," in *IEEE/ACM TASLP,* 2024. (OSU) [📄](https://arxiv.org/abs/2311.08630)
- "**ASoBO**: Attentive Beamformer Selection for Distant Speaker Diarization in Meetings," in *Proc. Interspeech*, 2024. (LIUM) [📄](https://arxiv.org/abs/2406.03251)

</details>

<details>
<summary><b>2022-2023 (4 papers)</b></summary>

- "Mutual Learning of Single- and Multi-Channel End-to-End Neural Diarization," in *Proc. IEEE SLT*, 2023. (Hitachi) [📄](https://arxiv.org/abs/2210.03459)
- "Semi-supervised multi-channel speaker diarization with cross-channel attention", in *Proc. ASRU,* 2023. (USTC) [📄](https://arxiv.org/abs/2307.08688)
- "Multi-Channel End-to-End Neural Diarization with Distributed Microphones", in *Proc. ICASSP*, 2022. (Hitachi) [📄](https://arxiv.org/abs/2110.04694)
- "Multi-Channel Speaker Diarization Using Spatial Features for Meetings", in *Proc. ICASSP*, 2022. (Tencent) [📄](https://ieeexplore.ieee.org/document/9747343)

</details>


---

<a id="online"></a>

## ⚡ Online — 16 papers

<details open>
<summary><b>🔥 2024-2025 (5 papers)</b></summary>

- **SCDiar**: "A Streaming Diarization System based on Speaker Change Detection and Speech Recognition," in *Proc. ICASSP,* 2025. [📄](https://arxiv.org/abs/2501.16641)
- **Streaming Sortformer**: "Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering," in *Proc. Interspeech,* 2025. (NVIDIA) [📄](https://arxiv.org/abs/2507.18446)
- **FS-EEND**: "Frame-wise streaming end-to-end speaker diarization with non-autoregressive self-attention-based attractors," in *Proc. ICASSP,* 2024. (Hangzhou) [📄](https://arxiv.org/abs/2309.13916) [💻](https://github.com/Audio-WestlakeU/FS-EEND)
- "Online speaker diarization of meetings guided by speech separation," in *Proc. ICASSP,* 2024. (LTCI) [📄](https://browse.arxiv.org/abs/2402.00067) [💻](https://github.com/egruttadauria98/SSpaVAlDo)
- "Interrelate Training and Clustering for Online Speaker Diarization," in *IEEE/ACM TASLP,* 2024. [📄](https://ieeexplore.ieee.org/abstract/document/10418572)

</details>

<details>
<summary><b>2023 (3 papers)</b></summary>

- "Absolute decision corrupts absolutely: conservative online speaker diarisation", in *Proc. ICASSP*, 2023. (Naver) [📄](https://arxiv.org/abs/2211.04768)
- "A Reinforcement Learning Framework for Online Speaker Diarization", in *Under Review. NeruIPS*, 2023. (CU) [📄](https://arxiv.org/abs/2302.10924)
- **OTS-VAD**: "End-to-end Online Speaker Diarization with Target Speaker Tracking," in *Submitted IEEE/ACM TASLP,* 2023. (DKU) [📄](https://arxiv.org/abs/2310.08696)

</details>

<details>
<summary><b>2022 (3 papers)</b></summary>

- "Low-Latency Online Speaker Diarization with Graph-Based Label Generation", in *Proc. Odyssey*, 2022. (DKU) [📄](https://arxiv.org/abs/2111.13803)
- **EEND-GLA**: "Online Neural Diarization of Unlimited Numbers of Speakers Using Global and Local Attractors", in *IEEE/ACM TASLP,* 2022. (Hitachi) [📄](https://arxiv.org/abs/2206.02432)
- **Online TS-VAD**: "Online Target Speaker Voice Activity Detection for Speaker Diarization", in *Proc. Interspeech*, 2022. (DKU) [📄](https://arxiv.org/abs/2207.05920)

</details>

<details>
<summary><b>2021 (4 papers)</b></summary>

- "Online End-to-End Neural Diarization with Speaker-Tracing Buffer", in *Proc. IEEE SLT*, 2021. (Hitachi) [📄](https://arxiv.org/abs/2006.02616)
- **BW-EDA-EEND**: "BW-EDA-EEND: Streaming End-to-End Neural Speaker Diarization for a Variable Number of Speakers", in *Proc. Interspeech*, 2021. (Amazon) [📄](https://arxiv.org/abs/2011.02678)
- **FS-EEND**: "Online Streaming End-to-End Neural Diarization Handling Overlapping Speech and Flexible Numbers of Speakers", in *Proc. Interspeech*, 2021. (Hitachi) [📄](https://arxiv.org/abs/2101.08473) [📝](https://velog.io/@fbdp1202/FS-EEND-%EB%A6%AC%EB%B7%B0-Online-end-to-end-diarization-handling-overlapping-speech-and-flexible-numbers-of-speakers)
- **Diart**: "Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation", in *Proc. ASRU*, 2021. [📄](https://arxiv.org/abs/2109.06483) [💻](https://github.com/juanmc2005/diart)

</details>

<details>
<summary><b>2020 (1 paper)</b></summary>

- "Supervised online diarization with sample mean loss for multi-domain data", in *Proc. ICASSP*, 2020 [📄](https://arxiv.org/abs/1911.01266) [💻](https://github.com/DonkeyShot21/uis-rnn-sml)

</details>


---

<a id="clustering"></a>

## 🔗 Clustering-based — 21 papers

<details open>
<summary><b>🔥 2025 (3 papers)</b></summary>

- **E-SHARC**: "End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization," in *IEEE/ACM TASLP,* 2025. (IISC) [📄](https://ieeexplore.ieee.org/abstract/document/10830571/)
- **Pyannote Community-1**: "pyannote.audio 4.0 with community-1 open-source diarization model," 2025. [🔗](https://www.pyannote.ai/blog/community-1) [💻](https://github.com/pyannote/pyannote-audio)
- "Speaker Diarization with Overlapping Community Detection Using Graph Attention Networks and Label Propagation Algorithm," in *Proc. Interspeech,* 2025. [📄](https://arxiv.org/abs/2506.02610)

</details>

<details>
<summary><b>2024 (6 papers)</b></summary>

- "Overlap-aware End-to-End Supervised Hierarchical Graph Clustering for Speaker Diarization," in *submitted to IEEE/ACM TASLP*, 2024. [📄](https://arxiv.org/abs/2401.12850)
- "Apollo's Unheard Voices: Graph Attention Networks for Speaker Diarization and Clustering for Fearless Steps Apollo Collection," in *Proc. ICASSP*, 2024. (UTD) [📄](https://ieeexplore.ieee.org/document/10446231)
- "Multi-View Speaker Embedding Learning for Enhanced Stability and Discriminability," in *Proc. ICASSP*, 2024. (Tsinghua) [📄](https://ieeexplore.ieee.org/abstract/document/10448494)
- "Towards Unsupervised Speaker Diarization System for Multilingual Telephone Calls Using Pre-trained Whisper Model and Mixture of Sparse Autoencoders," in *arXiv:2407.01963*, 2024. [📄](https://arxiv.org/abs/2407.01963)
- "Investigating Confidence Estimation Measures for Speaker Diarization," in *Proc. Interspeech*, 2024. [📄](https://arxiv.org/abs/2406.17124)
- "Once more Diarization: Improving meeting transcription systems through segment-level speaker reassignment," in *Proc. Interspeech*, 2024. (PU) [📄](https://arxiv.org/abs/2406.03155) [📄](https://www.isca-archive.org/interspeech_2024/boeddeker24_interspeech.html) [💻](https://github.com/fgnt/speaker_reassignment)

</details>

<details>
<summary><b>2023 (5 papers)</b></summary>

- **SCALE**: "Spectral Clustering-aware Learning of Embeddings for Speaker Diarisation", in *Proc. ICASSP*, 2023. (CAM) [📄](https://arxiv.org/abs/2210.13576)
- **SHARC**: "Supervised Hierarchical Clustering using Graph Neural Networks for Speaker Diarization", in *Proc. ICASSP*, 2023. (IISC) [📄](https://arxiv.org/abs/2302.12716)
- **CDGCN**: "Community Detection Graph Convolutional Network for Overlap-Aware Speaker Diarization," in *Proc. ICASSP*, 2023. (XMU) [📄](https://arxiv.org/abs/2306.14530)
- "**Pyannote.Audio 2.1**: Speaker Diarization Pipeline: Principle, Benchmark and Recipe", in *Proc. Interspeech*, 2023. (CNRS) [📄](https://www.isca-speech.org/archive/interspeech_2023/bredin23_interspeech.html)
- **GADEC**: "Graph attention-based deep embedded clustering for speaker diarization,", in *Speech Communication*, 2023. (NJUPT) [📄](https://www.sciencedirect.com/science/article/pii/S0167639323001255)

</details>

<details>
<summary><b>2020-2022 (4 papers)</b></summary>

- **UMAP-Leiden**: "Reformulating Speaker Diarization as Community Detection With Emphasis On Topological Structure", in *Proc. ICASSP*, 2022. (Alibaba) [📄](https://arxiv.org/abs/2204.12112)
- **Pyannote 2.0**: "End-to-end speaker segmentation for overlap-aware resegmentation", in *Proc. Interspeech*, 2021. (CNRS) [📄](https://arxiv.org/abs/2104.04045) [💻](https://github.com/pyannote/pyannote-audio) [🎬](https://www.youtube.com/watch?v=wDH2rvkjymY)
- **Pyannote**: "pyannote.audio: neural building blocks for speaker diarization", in *Proc. ICASSP*, 2020. (CNRS) [📄](https://arxiv.org/abs/1911.01255) [💻](https://github.com/pyannote/pyannote-audio) [🎬](https://www.youtube.com/watch?v=37R_R82lfwA)
- **Resegmentation with VB**: “Overlap-Aware Diarization: Resegmentation Using Neural End-to-End Overlapped Speech Detection”, in *Proc. ICASSP*, 2020. [📄](https://ieeexplore.ieee.org/document/9053096)

</details>

<details>
<summary><b>2018 (1 paper)</b></summary>

- **UIS-RNN**: "Fully Supervised Speaker Diarization" (Google) [📄](https://arxiv.org/abs/1810.04719) [💻](https://github.com/google/uis-rnn)

</details>

<details>
<summary><b>2019 (2 papers)</b></summary>

- **DNC**: "Discriminative Neural Clustering for Speaker Diarisation", in *Proc. IEEE SLT*, 2019. [📄](https://arxiv.org/abs/1910.09703) [💻](https://github.com/FlorianKrey/DNC) [📝](https://velog.io/@dongkeon/2019-DNC-SLT)
- **NME-SC**: “Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap”, *IEEE SPL,* 2019. [📄](https://arxiv.org/abs/2003.02405) [💻](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)

</details>


---

<a id="embedding"></a>

## 🧩 Embedding (With Clustering) — 14 papers

<details open>
<summary><b>🔥 2024 (4 papers)</b></summary>

- "Geodesic interpolation of frame-wise speaker embeddings for the diarization of meeting scenarios", in *Proc. ICASSP*, 2024. (PU) [📄](https://arxiv.org/abs/2401.03963) [📝](https://dongkeon.notion.site/2024-Geodesic-Interpolation-ICASSP-e7ec31aaa1fb49b3aa0f18dead92c5fc?pvs=4)
- "Speaker Embeddings With Weakly Supervised Voice Activity Detection For Efficient Speaker Diarization," in *Proc. Odyssey*, 2024. (IDLab) [📄](https://www.isca-archive.org/odyssey_2024/thienpondt24_odyssey.html)
- "Efficient Speaker Embedding Extraction Using a Twofold Sliding Window Algorithm for Speaker Diarization," in *Proc. Interspeech,* 2024. (HU) [📄](https://www.isca-archive.org/interspeech_2024/choi24d_interspeech.html)
- "Variable Segment Length and Domain-Adapted Feature Optimization for Speaker Diarization," in *Proc. Interspeech,* 2024. (XMU) [📄](https://www.isca-archive.org/interspeech_2024/zhang24b_interspeech.html) [💻](https://github.com/xiaoaaa2/Ada-sd)

</details>

<details>
<summary><b>2023 (5 papers)</b></summary>

- "In Search of Strong Embedding Extractors For Speaker Diarization", in *Proc. ICASSP*, 2023. (Naver) [📄](https://arxiv.org/abs/2210.14682) [📝](https://velog.io/@dongkeon/2023-In-Search-of-Strong-Embedding-Extractors-For-Speaker-Diarization-ICASSP)
- **DR-DESA**: "Advancing the dimensionality reduction of speaker embeddings for speaker diarisation: disentangling noise and informing speech activity", in *Proc. ICASSP*, 2023. (Naver) [📄](https://arxiv.org/abs/2110.03380) [📝](https://dongkeon.notion.site/2023-DR-DESA-ICASSP-c5ab46d215f243b5887b1c5d0b328bb6?pvs=4)
- **HEE**: "High-resolution embedding extractor for speaker diarisation", in *Proc. ICASSP*, 2023. (Naver)  [📄](https://arxiv.org/abs/2211.04060) [📝](https://dongkeon.notion.site/2023-HEE-ICASSP-fa7f95d6641744fa90d34ef73d2b8463?pvs=4)
- "Frame-wise and overlap-robust speaker embeddings for meeting diarization", in *Proc. ICASSP*, 2023. (PU) [📄](https://arxiv.org/pdf/2306.00625.pdf) [📝](https://dongkeon.notion.site/2023-Frame-wise-and-overlap-robust-speaker-embeddings-ICASSP-2b1878e76b6b45d09b021f00edee036b?pvs=4)
- "A Teacher-Student approach for extracting informative speaker embeddings from speech mixtures", in *Proc. Interspeech*, 2023. (PU) [📄](https://arxiv.org/abs/2306.00634)

</details>

<details>
<summary><b>2022 (3 papers)</b></summary>

- **GAT+AA**: "Multi-scale speaker embedding-based graph attention networks for speaker diarisation", in *Proc. ICASSP*, 2022. (Naver) [📄](https://arxiv.org/abs/2110.03361)
- **MSDD**: "Multi-scale Speaker Diarization with Dynamic Scale Weighting", in *Proc. Interspeech*, 2022. (NVIDIA) [📄](https://arxiv.org/abs/2203.15974) [💻](https://github.com/NVIDIA/NeMo) [🔗](https://developer.nvidia.com/blog/dynamic-scale-weighting-through-multiscale-speaker-diarization/)
- **PRISM**: "PRISM: Pre-trained Indeterminate Speaker Representation Model for Speaker Diarization and Speaker Verification", in *Proc. Interspeech*, 2022. (Alibaba) [📄](https://arxiv.org/abs/2205.07450)

</details>

<details>
<summary><b>2021 (2 papers)</b></summary>

- "Multi-Scale Speaker Diarization With Neural Affinity Score Fusion", in *Proc. ICASSP*, 2021. (USC) [📄](https://arxiv.org/abs/2011.10527)
- **AA+DR+NS**: "Adapting Speaker Embeddings for Speaker Diarisation", in *Proc. Interspeech*, 2021. (Naver) [📄](https://arxiv.org/abs/2104.02879) [📝](https://dongkeon.notion.site/2021-AutoEncoder-attention-aggregation-Interspeech-a9a8e6870a17418597fb0dad83d459fc?pvs=4)

</details>


---

<a id="speaker-id"></a>

## 🪪 With Speaker Identification — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- "Uncertainty Quantification in Machine Learning for Joint Speaker Diarization and Identification, in *Submitted to IEEE/ACM TASLP,* 2024. [📄](https://arxiv.org/abs/2312.16763)

</details>


---

<a id="speaker-rec"></a>

## 🔊 Speaker Recognition & Verification — 7 papers

<details open>
<summary><b>🔥 2024 (3 papers)</b></summary>

- "Rethinking Session Variability: Leveraging Session Embeddings for Session Robustness in Speaker Verification," in *Proc. ICASSP*, 2024. (Naver) [📄](https://arxiv.org/abs/2309.14741)
- "Leveraging In-the-Wild Data for Effective Self-Supervised Pretraining in Speaker Recognition," in *Proc. ICASSP*, 2024. (CUHK) [📄](https://arxiv.org/abs/2309.11730)
- "Disentangled Representation Learning for Environment-agnostic Speaker Recognition," in *Proc. Interspeech,* 2024. (KAIST) [📄](https://arxiv.org/abs/2406.14559) [📄](https://www.isca-archive.org/interspeech_2024/nam24b_interspeech.html) [💻](https://github.com/kaistmm/voxceleb-disentangler)

</details>

<details>
<summary><b>2023 (3 papers)</b></summary>

- "Build a SRE Challenge System: Lessons from VoxSRC 2022 and CNSRC 2022," in *Proc. Interspeech*, 2023. (SJTU) [📄](https://www.isca-speech.org/archive/interspeech_2023/chen23m_interspeech.html)
- **RecXi** "Disentangling Voice and Content with Self-Supervision for Speaker Recognition," in *Proc. NeurIPS,* 2023. (A*STAR) [📄](https://arxiv.org/abs/2310.01128)
- "**ECAPA2**: A Hybrid Neural Network Architecture and Training Strategy for Robust Speaker Embeddings," in *Proc. ASRU,* 2023. (IDLab) [📄](https://arxiv.org/abs/2401.08342) [🔗](https://huggingface.co/Jenthe/ECAPA2) [📝](https://dongkeon.notion.site/2023-ECAPA2-ASRU-962943495f2348dab3872e3481bc08a6?pvs=4)

</details>

<details>
<summary><b>2021 (1 paper)</b></summary>

- "Xi-Vector Embedding for Speaker Recognition," in *IEEE, SPL*. (A*STAR) [📄](https://arxiv.org/abs/2108.05679) [📝](https://dongkeon.notion.site/2021-Xi-Vector-SPL-ce538c87f6d64545acb557a223af3670?pvs=4)

</details>


---

<a id="scoring"></a>

## 📊 Scoring — 3 papers

<details open>
<summary><b>🔥 2019-2023 (3 papers)</b></summary>

- “Similarity Measurement of Segment-Level Speaker Embeddings in Speaker Diarization”, *IEEE/ACM TASLP,* 2023. (DKU) [📄](https://ieeexplore.ieee.org/document/9849033)
- "Self-Attentive Similarity Measurement Strategies in Speaker Diarization", in *Proc. Interspeech*, 2020. (DKU) [📄](https://www.isca-speech.org/archive/interspeech_2020/lin20_interspeech.html)
- **LSTM scoring**: "LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization", in *Proc. Interspeech*, 2019. (DKU) [📄](https://arxiv.org/abs/1907.10393)

</details>


---

<a id="vbx-hmm"></a>

## 📐 Variational Bayes and HMM — 24 papers

<details open>
<summary><b>🔥 2023-2024 (3 papers)</b></summary>

  - **DVBx**: "Discriminative Training of VBx Diarization", in *Proc. ICASSP*, 2024. (BUT) [📄](https://arxiv.org/abs/2310.02732) [💻](https://github.com/BUTSpeechFIT/DVBx)
  - **MS-VBx**: "Multi-Stream Extension of Variational Bayesian HMM Clustering (MS-VBx) for Combined End-to-End and Vector Clustering-based Diarization", in *Proc. Interspeech*, 2023. (NTT) [📄](https://arxiv.org/abs/2305.13580)
  - "Generalized domain adaptation framework for parametric back-end in speaker recognition", in *arXiv:2305.15567*, 2023. [📄](https://arxiv.org/abs/2305.15567)

</details>

<details>
<summary><b>2021-2022 (4 papers)</b></summary>

  - "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks", in *Computer Speech & Language*, 2022. (BUT) [📄](https://arxiv.org/abs/2012.14952)
  - **DCA-PLDA** "A Speaker Verification Backend with Robust Performance across Conditions”, in *Computer & Language*, 2022. [📄](https://arxiv.org/pdf/2102.01760.pdf) [💻](https://github.com/luferrer/DCA-PLDA)
  - "Analysis of the but Diarization System for **Voxconverse Challenge**", in *Proc. ICASSP*, 2021. (BUT) [📄](https://ieeexplore.ieee.org/document/9414315) [💻](https://github.com/BUTSpeechFIT/VBx/tree/v1.1_VoxConverse2020)
  - "Discriminatively trained probabilistic linear discriminant analysis for speaker verification", in *Proc. ICASSP*, 2021. [📄](https://ieeexplore.ieee.org/document/5947437)

</details>

<details>
<summary><b>2019-2020 (4 papers)</b></summary>

  - "Optimizing Bayesian Hmm Based X-Vector Clustering for **the Second Dihard Speech Diarization Challenge**", in *Proc. ICASSP*, 2020. (BUT) [📄](https://ieeexplore.ieee.org/document/9053982)
  - “Analysis of Speaker Diarization Based on Bayesian HMM With Eigenvoice Priors”, *IEEE/ACM TASLP,* 2019. (BUT) [📄](https://ieeexplore.ieee.org/document/8910412)
  - "BUT System Description for **DIHARD Speech Diarization Challenge 2019**", in *arXiv:1910.08847*, 2019. (BUT) [📄](https://arxiv.org/abs/1910.08847)
  - "Bayesian HMM Based x-Vector Clustering for Speaker Diarization", in *Proc. Interspeech*, 2019. (BUT) [📄](https://www.isca-speech.org/archive_v0/Interspeech_2019/abstracts/2813.html)

</details>

<details>
<summary><b>2018 (5 papers)</b></summary>

  - "Speaker Diarization based on Bayesian HMM with Eigenvoice Priors", in *Proc. Odyssey*, 2018. (BUT) [📄](https://www.isca-speech.org/archive/odyssey_2018/diez18_odyssey.html)
  - "VB-HMM Speaker Diarization with Enhanced and Refined Segment Representation", in *Proc. Odyssey*, 2018. (Tsinghua) [📄](https://www.isca-speech.org/archive_v0/Odyssey_2018/abstracts/53.html)
  - "Diarization is hard: some experiences and lessons learned for the JHU team in **the inaugural DIHARD challenge**", in *Proc. Interspeech*, 2018. [📄](https://www.isca-speech.org/archive/interspeech_2018/sell18_interspeech.html)
  - "The speaker partitioning problem", in *Proc. Odyssey*, 2018. [📄](https://www.isca-speech.org/archive_open/odyssey_2010/od10_034.html)
  - "Estimation of the Number of Speakers with Variational Bayesian PLDA in **the DIHARD Diarization Challenge**", in *Proc. Interspeech*, 2018. [📄](https://www.isca-speech.org/archive/interspeech_2018/vinals18_interspeech.html)

</details>

<details>
<summary><b>2015-2017 (3 papers)</b></summary>

  - "Domain Adaptation of PLDA Models in Broadcast Diarization by Means of Unsupervised Speaker Clustering, in *Proc. Interspeech*, 2017. [📄](https://www.isca-speech.org/archive/interspeech_2017/vinals17_interspeech.html)
  - "Iterative PLDA Adaptation for Speaker Diarization", in *Proc. Interspeech*, 2016. [📄](https://www.isca-speech.org/archive/interspeech_2016/lan16_interspeech.html)
  - "Diarization resegmentation in the factor analysis subspace", in *Proc. ICASSP*, 2015. [📄](https://ieeexplore.ieee.org/abstract/document/7178881)

</details>

<details>
<summary><b>2011-2014 (3 papers)</b></summary>

  - "Speaker diarization with plda i-vector scoring and unsupervised calibration", in *Proc. IEEE SLT*, 2014. [📄](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7078610)
  - "Unsupervised Methods for Speaker Diarization: An Integrated and Iterative Approach", *IEEE/ACM TASLP,* 2013. [📄](https://ieeexplore.ieee.org/abstract/document/6518171/)
  - "Analysis of i-vector length normalization in speaker recognition systems", in *Proc. Interspeech*, 2011. [📄](https://www.isca-speech.org/archive/interspeech_2011/garciaromero11_interspeech.html)

</details>

<details>
<summary><b>2005-2008 (2 papers)</b></summary>

  - "Bayesian analysis of speaker diarization with eigenvoice priors", in *CRIM, Montreal, Technical Report*, 2008. [📄](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=36db5cc928d01b13246582d71bde84fabbd24a19)
  - "Variational Bayesian methods for audio indexing", in *Proc. ICMI-MLMI*, 2005. [📄](https://www.eurecom.fr/fr/publication/1739/download/mm-valefa-050923.pdf)

</details>


---

<a id="asr"></a>

## 🗣️ With ASR — 27 papers

<details open>
<summary><b>🔥 2025-2026 (8 papers)</b></summary>

- **SE-DiCoW**: "Self-Enrolled Diarization-Conditioned Whisper," in *arXiv:2601.19194,* 2026. (BUT) [📄](https://arxiv.org/abs/2601.19194)
- **TagSpeech**: "End-to-End Multi-Speaker ASR and Diarization with Fine-Grained Temporal Grounding," in *arXiv:2601.06896,* 2026. [📄](https://arxiv.org/abs/2601.06896)
- **DiCoW**: "Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition," in *Proc. ICASSP,* 2025. (BUT) [📄](https://arxiv.org/abs/2501.00114) [💻](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)
- "Adapting Diarization-Conditioned Whisper for End-to-End Multi-Talker Speech Recognition," in *arXiv:2510.03723,* 2025. (BUT) [📄](https://arxiv.org/abs/2510.03723)
- "Diarization-Aware Multi-Speaker Automatic Speech Recognition via Large Language Models," in *arXiv:2506.05796,* 2025. (DKU) [📄](https://arxiv.org/abs/2506.05796)
- "Language Modelling for Speaker Diarization in Telephonic Interviews," in *arXiv:2501.17893,* 2025. [📄](https://arxiv.org/abs/2501.17893)
- **SC-SOT**: "Conditioning the Decoder on Diarized Speaker Information for End-to-End Overlapped Speech Recognition," in *Proc. Interspeech,* 2025. [📄](https://arxiv.org/abs/2506.12672)
  - "Target Speaker ASR with Whisper," in *Submitted to ICASSP,* 2025. (BUT) [📄](https://arxiv.org/abs/2409.09543) [💻](https://github.com/BUTSpeechFIT/TS-ASR-Whisper)

</details>

<details>
<summary><b>2024 (12 papers)</b></summary>

- "Enhancing Speaker Diarization with Large Language Models: A Contextual Beam Search Approach,", in *Proc. ICASSP*, 2024. (NVIDIA) [📄](https://arxiv.org/abs/2309.05248)
- **WEEND**: "Towards Word-Level End-to-End Neural Speaker Diarization with Auxiliary Network," in *arXiv:2309.08489*, 2024. (Google) [📄](https://arxiv.org/abs/2309.08489) [🔗](https://github.com/google/speaker-id/tree/master/publications/WEEND)
- "One model to rule them all ? Towards End-to-End Joint Speaker Diarization and Speech Recognition", in *Proc. ICASSP*, 2024. (CMU) [📄](https://arxiv.org/abs/2310.01688)
- "Meeting Recognition with Continuous Speech Separation and Transcription-Supported Diarization," in *arXiv:2309.16482*, 2024. (PU) [📄](https://arxiv.org/abs/2309.16482)
- “Joint Inference of Speaker Diarization and ASR with Multi-Stage Information Sharing," in *Proc. ICASSP*, 2024. (DKU) [📄](https://sites.duke.edu/dkusmiip/files/2024/03/icassp24_weiqing.pdf)
- "Multitask Speech Recognition and Speaker Change Detection for Unknown Number of Speakers" in *Proc. ICASSP*, 2024. (Idiap) [📄](https://ieeexplore.ieee.org/document/10446130)
- "A Spatial Long-Term Iterative Mask Estimation Approach for Multi-Channel Speaker Diarization and Speech Recognition," in *Proc. ICASSP*, 2024. (USTC) [📄](https://ieeexplore.ieee.org/document/10446168)
- "On the Success and Limitations of Auxiliary Network Based Word-Level End-to-End Neural Speaker Diarization," in *Proc. Interspeech*, 2024. (Google) [📄](https://www.isca-archive.org/interspeech_2024/huang24d_interspeech.html)
- **Sortformer**: "Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens," in *Submitted IEEE/ACM TASLP,* 2024. (NVIDIA) [📄](https://arxiv.org/abs/2409.06656)
  - "Speaker Mask Transformer for Multi-talker Overlapped Speech Recognition," in *arXiv:2312.10959*, 2024. (NICT) [📄](https://arxiv.org/abs/2312.10959)
  - "On Speaker Attribution with SURT," in *Proc. Odyssey,* 2024. (JHU) [📄](https://www.isca-archive.org/odyssey_2024/raj24_odyssey.html)
  - "Improving Speaker Assignment in Speaker-Attributed ASR for Real Meeting Applications," in *Proc. Odyssey,* 2024. (CNRS) [📄](https://www.isca-archive.org/odyssey_2024/cui24_odyssey.html)

</details>

<details>
<summary><b>2023 (5 papers)</b></summary>

- "Unified Modeling of Multi-Talker Overlapped Speech Recognition and Diarization with a Sidecar Separator", in *Proc. Interspeech*, 2023. (CUHK) [📄](https://arxiv.org/abs/2305.16263)
- "Multi-resolution Approach to Identification of Spoken Languages and to Improve Overall Language Diarization System using Whisper Model", in *Proc. Interspeech*, 2023.
- "Speaker Diarization for ASR Output with T-vectors: A Sequence Classification Approach", in *Proc. Interspeech*, 2023. [📄](https://www.isca-speech.org/archive/interspeech_2023/yousefi23_interspeech.html)
- "Lexical Speaker Error Correction: Leveraging Language Models for Speaker Diarization Error Correction", in *Proc. Interspeech*, 2023. (Amazon) [📄](https://arxiv.org/abs/2306.09313)
  - "**SA-Paraformer**: Non-autoregressive End-to-End Speaker-Attributed ASR," in *Proc. ASRU*, 2023. (Alibaba) [📄](https://arxiv.org/abs/2310.04863)

</details>

<details>
<summary><b>2022 (2 papers)</b></summary>

- "Transcribe-to-Diarize: Neural Speaker Diarization for Unlimited Number of Speakers using End-to-End Speaker-Attributed ASR," in *Proc. ICASSP*, 2022. [📄](https://arxiv.org/abs/2110.03151)
- "Tandem Multitask Training of Speaker Diarisation and Speech Recognition for Meeting Transcription", in *Proc. Interspeech*, 2022. [📄](https://arxiv.org/abs/2207.03852)

</details>


---

<a id="language"></a>

## 🌐 Language Diarization — 2 papers

<details open>
<summary><b>🔥 2023 (2 papers)</b></summary>

- "End-to-End Spoken Language Diarization with Wav2vec Embeddings", in *Proc. Interspeech*, 2023. [📄](https://www.isca-speech.org/archive/interspeech_2023/mishra23_interspeech.html) [💻](https://github.com/jagabandhumishra/W2V-E2E-Language-Diarization)
- "Multi-resolution Approach to Identification of Spoken Languages and To Improve Overall Language Diarization System Using Whisper Model," in *Proc. Interspeech*, 2023. [📄](https://www.isca-speech.org/archive/interspeech_2023/vachhani23_interspeech.html)

</details>


---

<a id="nlp-llm"></a>

## 💬 With NLP (LLM) — 10 papers

<details open>
<summary><b>🔥 2024-2025 (7 papers)</b></summary>

- **SpeakerLM**: "End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models," in *arXiv:2508.06372,* 2025. [📄](https://arxiv.org/abs/2508.06372)
- "Interactive Real-Time Speaker Diarization Correction with Human Feedback," in *arXiv:2509.18377,* 2025. [📄](https://arxiv.org/abs/2509.18377)
- "**DiariST**: Streaming Speech Translation with Speaker Diarization," in *Proc. ICASSP*, 2024. (Microsoft) [📄](https://arxiv.org/abs/2309.08007) [💻](https://github.com/Mu-Y/DiariST)
- **JPCP:** "Improving Speaker Diarization using Semantic Information: Joint Pairwise Constraints Propagation," in *arXiv:2309.10456*, 2024. (Alibaba) [📄](https://arxiv.org/abs/2309.10456)
- "**DiarizationLM**: Speaker Diarization Post-Processing with Large Language Models," in *Proc. Interspeech*, 2024. (Google) [📄](https://www.isca-archive.org/interspeech_2024/wang24h_interspeech.html) [📄](https://arxiv.org/abs/2401.03506) [💻](https://github.com/google/speaker-id/tree/master/DiarizationLM) [📝](https://dongkeon.notion.site/DiarizationLM-1f8eac8794968030839ac145abb0a546?pvs=4)
- "LLM-based speaker diarization correction: A generalizable approach," in *Submitted to IEEE/ACM TASLP*, 2024. [📄](https://arxiv.org/abs/2406.04927)
- "AG-LSEC: Audio Grounded Lexical Speaker Error Correction," in *Proc. Interspeech*, 2024. (Amazon) [📄](https://arxiv.org/abs/2406.17266)

</details>

<details>
<summary><b>2023 (3 papers)</b></summary>

- "Exploring Speaker-Related Information in Spoken Language Understanding for Better Speaker Diarization", in *Proc. ACL*, 2023. (Alibaba) [📄](https://arxiv.org/abs/2305.12927)
- **MMSCD**, "Encoder-decoder multimodal speaker change detection", in *Proc. Interspeech*, 2023. (Naver) [📄](https://arxiv.org/abs/2306.00680)
- "Aligning Speakers: Evaluating and Visualizing Text-based Diarization Using Efficient Multiple Sequence Alignment,", in *Proc. ICTAI*, 2023. [📄](https://arxiv.org/abs/2309.07677)

</details>


---

<a id="vision"></a>

## 👁️ With Vision — 21 papers

<details open>
<summary><b>🔥 2025-2026 (4 papers)</b></summary>

- **CineSRD**: "Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization," in *arXiv:2603.16966,* 2026. [📄](https://arxiv.org/abs/2603.16966)
- "Integrating Audio, Visual, and Semantic Information for Enhanced Multimodal Speaker Diarization," in *Proc. ACL,* 2025. [📄](https://arxiv.org/abs/2408.12102)
- "Cross-Attention and Self-Attention for Audio-visual Speaker Diarization," in *arXiv:2506.02621,* 2025. [📄](https://arxiv.org/abs/2506.02621)
- "Count Your Speakers! Multitask Learning for Multimodal Speaker Diarization," in *Proc. Interspeech,* 2025. [📄](https://www.isca-archive.org/interspeech_2025/singh25_interspeech.pdf)

</details>

<details>
<summary><b>2024 (6 papers)</b></summary>

- "Speaker Diarization of Scripted Audiovisual Content," in *arXiv:2308.02160*, 2024. (Amazon) [📄](https://arxiv.org/abs/2308.02160)
- "**AFL-Net**: Integrating Audio, Facial, and Lip Modalities with Cross-Attention for Robust Speaker Diarization in the Wild," in *Proc. ICASSP*, 2024. (Tencent) [📄](https://arxiv.org/abs/2312.05730) [🎬](https://yyk77.github.io/afl_net.github.io/)
- "**Multichannel AV-wav2vec2**: A Framework for Learning Multichannel Multi-Modal Speech Representation," in *Proc. AAAI,* 2024. (Tencent) [📄](https://arxiv.org/abs/2401.03468)
- "**3D-Speaker-Toolkit**: An Open Source Toolkit for Multi-modal Speaker Verification and Diarization," in *arXiv:2403.19971*, 2024. (Alibaba) [📄](https://arxiv.org/abs/2403.19971) [💻](https://github.com/modelscope/3D-Speaker)
- "Target Speech Diarization with Multimodal Prompts," in *Submitted to IEEE/ACM TASLP*, 2024. (NUS) [📄](https://arxiv.org/abs/2406.07198)
- **MFV-KSD**: "Multi-Stage Face-Voice Association Learning with Keynote Speaker Diarization," in *Submitted to ACM MM,* 2024. [📄](https://arxiv.org/abs/2407.17902) [💻](https://github.com/TaoRuijie/MFV-KSD)

</details>

<details>
<summary><b>2023 (5 papers)</b></summary>

- "Audio-Visual Speaker Diarization in the Framework of Multi-User Human-Robot Interaction", in *Proc. ICASSP*, 2023. [📄](https://ieeexplore.ieee.org/abstract/document/10096295)
- **STHG**: "Spatial-Temporal Heterogeneous Graph Learning for Advanced Audio-Visual Diarization, in *Proc. CVPR*, 2023. (Intel) [📄](https://arxiv.org/abs/2306.10608)
- "Uncertainty-Guided End-to-End Audio-Visual Speaker Diarization for Far-Field Recordings," in *Proc. ACM MM*, 2023. [📄](https://dl.acm.org/doi/abs/10.1145/3581783.3612424)
- "Joint Training or Not: An Exploration of Pre-trained Speech Models in Audio-Visual Speaker Diarization," in *Springer Computer Science proceedings,* 2023. [📄](https://arxiv.org/abs/2312.04131)
- **EEND-EDA++**: "Late Audio-Visual Fusion for In-The-Wild Speaker Diarization," in *arXiv:2211.01299v2*, 2023. [📄](https://arxiv.org/abs/2211.01299v2)

</details>

<details>
<summary><b>2022 (3 papers)</b></summary>

- **AVA-AVD (AVR-Net)**: "AVA-AVD: Audio-Visual Speaker Diarization in the Wild", in *Proc. ACM MM*, 2022. [📄](https://arxiv.org/abs/2111.14448) [💻](https://github.com/zcxu-eric/AVA-AVD) [🎬](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3503161.3548027&file=MM22-fp1169.mp4)
- "End-to-End Audio-Visual Neural Speaker Diarization", in *Proc. Interspeech*, 2022. (USTC) [📄](https://www.isca-speech.org/archive/interspeech_2022/he22c_interspeech.html) [💻](https://github.com/mispchallenge/misp2022_baseline/tree/main/track1_AVSD) [📝](https://velog.io/@dongkeon/2022-End-to-End-Audio-Visual-Neural-Speaker-Diarization-2022-Interspeech)
- **DyViSE**: "DyViSE: Dynamic Vision-Guided Speaker Embedding for Audio-Visual Speaker Diarization", in *Proc. MMSP*, 2022. (THU) [📄](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9948860) [💻](https://github.com/zaocan666/DyViSE)

</details>

<details>
<summary><b>2024 (1 paper)</b></summary>

- "Multi-Input Multi-Output Target-Speaker Voice Activity Detection For Unified, Flexible, and Robust Audio-Visual Speaker Diarization," in *Submitted to IEEE/ACM TASLP*. (DKU) [📄](https://arxiv.org/abs/2401.08052)

</details>

<details>
<summary><b>2019-2020 (2 papers)</b></summary>

- "Self-supervised learning for audio-visual speaker diarization", in *Proc. ICASSP*, 2020. (Tencent) [📄](https://arxiv.org/abs/2002.05314) [🔗](https://yifan16.github.io/av-spk-diarization/)
- "Who said that?: Audio-visual speaker diarisation of real-world meetings", in *Proc. Interspeech*, 2019. (Naver) [📄](https://arxiv.org/abs/1906.10042)

</details>


---

<a id="spoofing"></a>

## 🛡️ Related Spoofing — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- "Spoof Diarization: "What Spoofed When" in Partially Spoofed Audio," in *Proc. Interspeech*, 2024. (IITK) [📄](https://arxiv.org/pdf/2406.07816)

</details>


---

<a id="tts"></a>

## 🔉 Related TTS

<a id="anonymization"></a>

### 📌 Speaker Anonymization — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- "A Benchmark for Multi-speaker Anonymization," in *Submitted to IEEE/ACM TASLP*, 2024. (SIT) [📄](https://arxiv.org/abs/2407.05608) [💻](https://github.com/xiaoxiaomiao323/MSA)

</details>

<a id="singing"></a>

### 📌 Singing Diarization — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- "Song Data Cleansing for End-to-End Neural Singer Diarization Using Neural Analysis and Synthesis Framework," in *Proc. Interspeech*, 2024. (LY) [📄](https://arxiv.org/abs/2406.16315)

</details>


---

<a id="emotion"></a>

## 😊 With Emotion — 3 papers

<details open>
<summary><b>🔥 2023-2024 (3 papers)</b></summary>

- "**ED-TTS**: Multi-scale Emotion Modeling using Cross-domain Emotion Diarization for Emotional Speech Synthesis, in *Proc. ICASSP,* 2024. [📄](https://arxiv.org/abs/2401.08166)
- "**Speech Emotion Diarization**: Which Emotion Appears When?," in *Proc. ASRU,* 2023. (Zaion) [📄](https://arxiv.org/abs/2306.12991)
- "**EmoDiarize**: Speaker Diarization and Emotion Identification from Speech Signals using Convolutional Neural Networks," in *arxiv:2310.12851*, 2023. [📄](https://arxiv.org/abs/2310.12851)

</details>


---

<a id="personal-vad"></a>

## 🎙️ Personal VAD — 2 papers

<details open>
<summary><b>🔥 2020-2023 (2 papers)</b></summary>

- "**SVVAD**: Personal Voice Activity Detection for Speaker Verification", in *Proc. Interspeech*, 2023. [📄](https://arxiv.org/abs/2305.19581)
- "**Personal VAD**: Speaker-Conditioned Voice Activity Detection", in *Proc. Odyssey*, 2020. (Google) [📄](https://arxiv.org/abs/1908.04284)

</details>


---

<a id="vad-osd-scd"></a>

## 📈 VAD & OSD & SCD — 10 papers

<details open>
<summary><b>🔥 2024 (3 papers)</b></summary>

- "**USM-SCD**: Multilingual Speaker Change Detection Based on Large Pretrained Foundation Models," in *Proc. ICASSP*, 2024. (Google) [📄](https://arxiv.org/abs/2309.08023)
- "Channel-Combination Algorithms for Robust Distant Voice Activity and Overlapped Speech Detection," in *IEEE/ACM TASLP,* 2024. [📄](https://arxiv.org/abs/2402.08312)
- "Speaker Change Detection with Weighted-sum Knowledge Distillation based on Self-supervised Pre-trained Models," in *Proc. Interspeech,* 2024. [📄](https://www.isca-archive.org/interspeech_2024/su24_interspeech.html)

</details>

<details>
<summary><b>2023 (4 papers)</b></summary>

- "Multitask Detection of Speaker Changes, Overlapping Speech and Voice Activity Using wav2vec 2.0," in *Proc. ICASSP*, 2023. [📄](https://arxiv.org/abs/2210.14755) [💻](https://github.com/mkunes/w2v2_audioFrameClassification)
- "**Semantic VAD**: Low-Latency Voice Activity Detection for Speech Interaction," in *Proc. Interspeech*, 2023. [📄](https://arxiv.org/abs/2305.12450)
- "Joint speech and overlap detection: a benchmark over multiple audio setup and speech domains," in *arxiv:2307.13012*, 2023. [📄](https://arxiv.org/abs/2307.13012)
- "Advancing the study of Large-Scale Learning in Overlapped Speech Detection," in *arXiv:2308.05987*, 2023. [📄](https://arxiv.org/abs/2308.05987)

</details>

<details>
<summary><b>2022 (3 papers)</b></summary>

- "Overlapped Speech Detection in Broadcast Streams Using X-vectors," in *Proc. Interspeech*, 2022. [📄](https://www.isca-speech.org/archive/interspeech_2022/mateju22_interspeech.html)
- "Overlapped speech and gender detection with WavLM pre-trained features,"  in *Proc. Interspeech*, 2022. [📄](https://www.isca-speech.org/archive/interspeech_2022/lebourdais22_interspeech.html)
- "Microphone Array Channel Combination Algorithms for Overlapped Speech Detection,"  in *Proc. Interspeech*, 2022. [📄](https://www.isca-speech.org/archive/interspeech_2022/mariotte22_interspeech.html)

</details>


---

<a id="dataset"></a>

## 📊 Dataset — 11 papers

<details open>
<summary><b>🔥 2024-2025 (6 papers)</b></summary>

- "Conversations in the wild: Data collection, automatic generation and evaluation," in *Computer Speech & Language,* 2025. [📄](https://www.sciencedirect.com/science/article/pii/S0885230824000822)
- **M3SD**: "Multi-modal, Multi-scenario and Multi-language Speaker Diarization Dataset," in *arXiv:2506.14427,* 2025. [📄](https://arxiv.org/abs/2506.14427)
- "**VoxBlink**: X-Large Speaker Verification Dataset on Camera", in *Proc. ICASSP,* 2024. [📄](https://arxiv.org/abs/2308.07056) [🔗](https://voxblink.github.io/)
- "NOTSOFAR-1 Challenge: New Datasets, Baseline, and Tasks for Distant Meeting Transcription," in *arXiv:2401.08887,* 2024. (MS) [📄](https://arxiv.org/abs/2401.08887)
- "A Comparative Analysis of Speaker Diarization Models: Creating a Dataset for German Dialectal Speech," in *Proc. ACL,* 2024. [📄](https://aclanthology.org/2024.fieldmatters-1.6/)
- "ALLIES: A Speech Corpus for Segmentation, Speaker Diarization, Speech Recognition and Speaker Change Detection," in *Proc. ACL*, 2024. (LIUM) [📄](https://aclanthology.org/2024.lrec-main.67/)

</details>

<details>
<summary><b>2020-2022 (5 papers)</b></summary>

- **Ego4D**: " Around the World in 3,000 Hours of Egocentric Video," in *Proc. CVPR*, 2022. (Meta) [📄](https://arxiv.org/abs/2110.07058) [💻](https://github.com/EGO4D/audio-visual) [🔗](https://ego4d-data.org/docs/benchmarks/av-diarization/)
- **AliMeeting**: "Summary On The ICASSP 2022 Multi-Channel Multi-Party Meeting Transcription Grand Challenge," in *Proc. ICASSP,* 2022. (Alibaba) [📄](https://arxiv.org/abs/2202.03647) [🔗](https://www.openslr.org/119) [💻](https://github.com/yufan-aslp/AliMeeting)
- **Voxconverse**: "Spot the conversation: speaker diarisation in the wild", in *Proc. Interspeech*, 2020. (VGG, Naver) [📄](https://arxiv.org/abs/2007.01216) [💻](https://github.com/joonson/voxconverse) [🔗](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)
- **MSDWild**: Multi-modal Speaker Diarization Dataset in the Wild, in *Proc. Interspeech,* 2020. [📄](https://www.isca-speech.org/archive/interspeech_2022/liu22t_interspeech.html) [🔗](https://github.com/X-LANCE/MSDWILD)
- "LibriMix: An Open-Source Dataset for Generalizable Speech Separation," in *arXiv:2005.11262*, 2020. [📄](https://arxiv.org/abs/2005.11262) [💻](https://github.com/JorisCos/LibriMix)

</details>


---

<a id="tools"></a>

## 🛠️ Tools — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- "Gryannote open-source speaker diarization labeling tool," in *Proc. Interspeech (Show and Tell),* 2024. (IRIT) [📄](https://www.isca-archive.org/interspeech_2024/pages24_interspeech.html) [💻](https://github.com/clement-pages/gryannote)

</details>


---

<a id="self-supervised"></a>

## 🔄 Self-Supervised — 5 papers

<details open>
<summary><b>🔥 2025 (3 papers)</b></summary>

- **DiariZen**: "Leveraging Self-Supervised Learning for Speaker Diarization," in *Proc. ICASSP,*" 2025. (BUT) [📄](https://arxiv.org/abs/2409.09408) [📄](https://ieeexplore.ieee.org/abstract/document/10889475) [💻](https://github.com/BUTSpeechFIT/DiariZen) [📝](https://dongkeon.notion.site/DiarZen-1f8eac87949680439b1ce4aeeb211fa9?pvs=4)
- "Efficient and Generalizable Speaker Diarization via Structured Pruning of Self-Supervised Models," in *arXiv:2506.18623,* 2025. (BUT) [📄](https://arxiv.org/abs/2506.18623) [💻](https://github.com/BUTSpeechFIT/DiariZen)
- "Fine-tune Before Structured Pruning: Towards Compact and Accurate Self-Supervised Models for Speaker Diarization," in *arXiv:2505.24111,* 2025. [📄](https://arxiv.org/abs/2505.24111)

</details>

<details>
<summary><b>2022 (2 papers)</b></summary>

- “Self-supervised Speaker Diarization”, in *Proc. Interspeech,* 2022. [📄](https://arxiv.org/abs/2204.04166)
- **CSDA**: "Continual Self-Supervised Domain Adaptation for End-to-End Speaker Diarization", in *Proc. IEEE SLT*, 2022. (CNRS) [📄](https://ieeexplore.ieee.org/document/10023195) [💻](https://github.com/juanmc2005/CSDA)

</details>


---

<a id="semi-supervised"></a>

## 🔃 Semi-Supervised — 1 paper

<details open>
<summary><b>🔥 2017 (1 paper)</b></summary>

- "Active Learning Based Constrained Clustering For Speaker Diarization", in *IEEE/ACM TASLP,* 2017. (UT) [📄](https://ieeexplore.ieee.org/abstract/document/8030331)

</details>


---

<a id="measurement"></a>

## 📏 Measurement — 3 papers

<details open>
<summary><b>🔥 2022-2025 (3 papers)</b></summary>

- **SDBench**: “A Comprehensive Benchmark Suite for Speaker Diarization,” in *Proc. Interspeech,* 2025. [📄](https://arxiv.org/abs/2507.16136)
- “Benchmarking Diarization Models,” in *arXiv:2509.26177,* 2025. [📄](https://arxiv.org/abs/2509.26177)
- **BER:** “Balanced Error Rate For Speaker Diarization”, in *Proc. arXiv:2211.04304,* 2022 [📄](https://arxiv.org/abs/2211.04304) [💻](https://github.com/X-LANCE/BER)

</details>


---

<a id="child-adult"></a>

## 👶 Child-Adult — 2 papers

<details open>
<summary><b>🔥 2023-2024 (2 papers)</b></summary>

- "Exploring Speech Foundation Models for Speaker Diarization in Child-Adult Dyadic Interactions," in *Proc. Interspeech*, 2024. (USC) [📄](https://arxiv.org/abs/2406.07890)
- "Robust Self Supervised Speech Embeddings for Child-Adult Classification in Interactions involving Children with Autism," in *Proc. Interspeech*, 2023. [📄](https://www.isca-speech.org/archive/interspeech_2023/lahiri23_interspeech.html)

</details>

<a id="challenge"></a>

# 🏆 Challenge


---

<a id="voxsrc"></a>

## 🔊 VoxSRC (VoxCeleb Speaker Recognition Challenge)

### 📌 VoxSRC-20 Track4 — 3 papers

<details open>
<summary><b>Unknown (3 papers)</b></summary>

- 🥇 1st: **Microsoft** [📄](https://arxiv.org/abs/2010.11458) [🎬](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2020/participants/mandalorian.mp4)
- 🥈 2nd: **BUT** [📄](https://arxiv.org/abs/2010.11718) [🎬](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2020/participants/landini.mp4)
- 🥉 3rd: **DKU**  [📄](https://arxiv.org/abs/2010.12731)

</details>

### 📌 VoxSRC-21 Track4 — 3 papers

<details open>
<summary><b>Unknown (3 papers)</b></summary>

- 🥇 1st: **DKU** [📄](https://arxiv.org/abs/2109.02002) [🎬](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/DKU-DukeECE-Lenovo.mp4)
- 🥈 2nd: **Bytedance** [📄](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/ByteDance_diarization.pdf) [🎬](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/participants/Bytedance_SAMI.mp4)
- 🥉 3rd: **Tencent**  [📄](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/reports/Tencent_diarization.pdf)

</details>

### 📌 VoxSRC-22 Track4 — 3 papers

<details open>
<summary><b>Unknown (3 papers)</b></summary>

- 🥇 1st: **DKU** [📄](https://arxiv.org/abs/2210.01677) [📊](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/DKU-DukeECE_slides.pdf) [🎬](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/DKU-DukeECE_video.mp4)
- 🥈 2nd: **KristonAI** [📄](https://arxiv.org/abs/2209.11433) [📊](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/voxsrc2022_kristonai_track4.pdf) [🎬](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/voxsrc2022_kristonai_track4.mp4)
- 🥉 3rd: **GIST** [📄](https://arxiv.org/abs/2209.10357) [📊](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/slides/gist_slides.pdf) [🎬](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2022/videos/video_aiter.mp4) [📝](https://velog.io/@fbdp1202/VoxSRC22-%EB%8C%80%ED%9A%8C-%EC%B0%B8%EA%B0%80-%EB%A6%AC%EB%B7%B0-VoxSRC-Challenge-2022-Task-4)

</details>

### 📌 VoxSRC-23 Track4 — 5 papers

<details open>
<summary><b>Unknown (5 papers)</b></summary>

- 🥇 1st: **DKU** [📄](https://arxiv.org/abs/2308.07595) [📊](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/slides/dku_track4_slides.pdf) [🎬](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/videos/track4_dku.mp4)
- 🥈 2nd: **KrispAI** [📄](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/reports/krisp_report.pdf) [📊](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/slides/krispai_slides.pdf) [🎬](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/videos/krispai.mp4)
- 🥉 3rd: **Pyannote** [📄](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/reports/pyannote_report.pdf) [📊](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/slides/pyannote_slides.pdf) [🎬](http://mm.kaist.ac.kr/datasets/voxceleb/voxsrc/data_workshop_2023/videos/pyannote_video.mp4)
- 4th: **GIST** [📄](https://arxiv.org/abs/2308.07788)
- **Wespeaker** [📄](https://arxiv.org/abs/2306.15161)

</details>


---

<a id="m2met"></a>

## 📡 M2MeT (Multi-channel Multi-party Meeting Transcription Grand Challenge)

### 📌 2022 M2MeT — 2 papers

<details open>
<summary><b>Unknown (2 papers)</b></summary>

- 🥇 1st: **DKU** [📄](https://arxiv.org/abs/2202.02687)
- 🥈 2nd: **CUHK-TENCENT** [📄](https://arxiv.org/abs/2202.01986)

</details>


---

<a id="misp"></a>

## 📌 MISP (Multimodal Information Based Speech Processing)

### 📌 2022 MISP Track1 — 3 papers

<details open>
<summary><b>Unknown (3 papers)</b></summary>

- 🥇 1st: **WHU-Alibaba** [📄](https://mispchallenge.github.io/mispchallenge2022/papers/task1/Track1_WHU-ALIBABA.pdf) [📝](https://velog.io/@dongkeon/2023-WHU-Alibaba-MISP-2022)
- 🥈 2nd: **SJTU** [📄](https://mispchallenge.github.io/mispchallenge2022/papers/task1/Track1_SJTU.pdf)
- 🥉 3rd: **NPU-ASLP** [📄](https://mispchallenge.github.io/mispchallenge2022/papers/task1/Track1_NPU-ASLP.pdf)

</details>


---

<a id="dihard"></a>

## 📌 DIHARD

### 📌 2020 DIHARD III

#### 📌 Track1 — 3 papers

<details open>
<summary><b>Unknown (3 papers)</b></summary>

- 🥇 1st: **USTC** [📄](https://arxiv.org/abs/2103.10661) [📊](https://dihardchallenge.github.io/dihard3workshop/slide/The%20USTC-NELSLIP%20Systems%20for%20DIHARD%20III%20Challenge.pdf) [🎬](https://www.youtube.com/watch?v=ijNPazF8EwU&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=7)
- 🥈 2nd: **Hitachi** [📄](https://arxiv.org/abs/2102.01363) [📊](https://dihardchallenge.github.io/dihard3workshop/slide/Hitachi-JHU%20System%20for%20the%20Third%20DIHARD%20Speech%20Diarization%20Challenge.pdf) [🎬](https://www.youtube.com/watch?v=xKGzrF1YEjQ&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=4)
- 🥉 3rd: **Naver Clova** [📄](https://dihardchallenge.github.io/dihard3/system_descriptions/dihard3_system_description_team73.pdf) [📊](https://dihardchallenge.github.io/dihard3workshop/slide/NAVER%20Clova%20Submission%20To%20The%20Third%20DIHARD%20Challenge.pdf) [🎬](https://www.youtube.com/watch?v=X9F2WPWJIR4&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=2)

</details>

#### 📌 Track2 — 3 papers

<details open>
<summary><b>Unknown (3 papers)</b></summary>

- 🥇 1st: **USTC-NELSLIP** [📄](https://arxiv.org/abs/2103.10661) [📊](https://dihardchallenge.github.io/dihard3workshop/slide/The%20USTC-NELSLIP%20Systems%20for%20DIHARD%20III%20Challenge.pdf) [🎬](https://www.youtube.com/watch?v=ijNPazF8EwU&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=7)
- 🥈 2nd: **Hitachi** [📄](https://arxiv.org/abs/2102.01363) [📊](https://dihardchallenge.github.io/dihard3workshop/slide/Hitachi-JHU%20System%20for%20the%20Third%20DIHARD%20Speech%20Diarization%20Challenge.pdf) [🎬](https://www.youtube.com/watch?v=xKGzrF1YEjQ&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=4)
- 🥉 3rd: **DKU** [📄](https://arxiv.org/abs/2102.03649) [📊](https://dihardchallenge.github.io/dihard3workshop/slide/System%20Description%20for%20Team%20DKU-Duke-Lenovo.pdf) [🎬](https://www.youtube.com/watch?v=FF5QAm6Jgy8&list=PLK8w8IgaxTVrf1DBMajNytq87bZi183El&index=6)

</details>

#### 📌 Etc.


---

<a id="displace-2023"></a>

## 🏆 The DISPLACE Challenge 2023 — 2 papers

<details open>
<summary><b>🔥 2023 (2 papers)</b></summary>

- "**The DISPLACE Challenge 2023** - DIarization of SPeaker and LAnguage in Conversational Environments," in *Proc. Interspeech*, 2023. [📄](https://www.isca-speech.org/archive/interspeech_2023/baghel23_interspeech.html) [🔗](https://displace2023.github.io/)
- "The SpeeD--ZevoTech submission at DISPLACE 2023," in *Proc. Interspeech*, 2023. [📄](https://www.isca-speech.org/archive/interspeech_2023/pirlogeanu23_interspeech.html)

</details>


---

<a id="merlion"></a>

## 🏆 MERLIon CCS Challenge 2023 — 1 paper

<details open>
<summary><b>🔥 2023 (1 paper)</b></summary>

- "**MERLIon CCS Challenge**: A English-Mandarin code-switching child-directed speech corpus for language identification and diarization," in *Proc. Interspeech*, 2023. [📄](https://www.isca-speech.org/archive/interspeech_2023/baghel23_interspeech.html) [🔗](https://sites.google.com/view/merlion-ccs-challenge/)

</details>


---

<a id="chime-6"></a>

## 📌 CHiME-6


---

<a id="icmc-asr"></a>

## 🏆 ICMC-ASR Grand Challenge (ICASSP2024) — 2 papers

<details open>
<summary><b>🔥 2023-2024 (2 papers)</b></summary>

- "ICMC-ASR: The ICASSP 2024 In-Car Multi-Channel Automatic Speech Recognition Challenge," 2023. [📄](https://arxiv.org/abs/2401.03473)
- "The NUS-HLT System for ICASSP2024 ICMC-ASR Grand Challenge," in *Technical Report*, 2023. [📄](https://arxiv.org/abs/2312.16002)

</details>


---

<a id="displace-2"></a>

## 📌 The Second DISPLACE — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- "The Second DISPLACE Challenge : DIarization of SPeaker and LAnguage in Conversational Environments," in *Proc. Interspeech*, 2024. [📄](https://arxiv.org/abs/2406.09494)

</details>


---

<a id="chime-8"></a>

## 📌 CHiME-8 — 1 paper

<details open>
<summary><b>🔥 2024 (1 paper)</b></summary>

- "The CHiME-8 DASR Challenge for Generalizable and Array Agnostic Distant Automatic Speech Recognition and Diarization," 2024. [📄](https://arxiv.org/abs/2407.16447)

</details>


---

<a id="misp-2025"></a>

## 📌 MISP 2025 (Interspeech 2025) — 2 papers

<details open>
<summary><b>🔥 2025 (2 papers)</b></summary>

- "The Multimodal Information Based Speech Processing (MISP) 2025 Challenge: Audio-Visual Diarization and Recognition," 2025. [📄](https://arxiv.org/abs/2505.13971)
- "Overlap-Adaptive Hybrid Speaker Diarization and ASR-Aware Observation Addition for MISP 2025 Challenge," in *Proc. Interspeech,* 2025. [📄](https://arxiv.org/abs/2505.22013)

</details>


---

<a id="other-awesome"></a>

# 🔗 Other Awesome Lists

- [wq2012/awesome-diarization](https://github.com/wq2012/awesome-diarization)
- [jim-schwoebel/voice_datasets](https://github.com/jim-schwoebel/voice_datasets)
