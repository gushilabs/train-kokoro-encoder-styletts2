# train-kokoro-encoder-styletts2

An experimental open-source attempt to retrain the missing **Kokoro TTS** encoders using a modified **StyleTTS2** framework to enable custom voice cloning.

## 📌 Project Purpose
While Kokoro TTS is based on StyleTTS2, its official release only included the decoder weights. This project is an experimental attempt to retrain the missing encoders to explore custom voice cloning and generate new voice packs.

##  Current State
The training pipeline runs successfully, but initial zero-shot voice cloning results are sub-optimal. According to the original author of Kokoro TTS, high-quality voice cloning may not be achievable under these conditions due to limited training data and time.

However, the pipeline has successfully trained a voice pack encoder. The resulting voice packs are fully functional and compatible with the official Kokoro V1 release model. This repository is open-sourced to share these findings and artifacts with the community.

## 🙏 Background & Credits
Initially, I attempted to use the vanilla [StyleTTS2](https://github.com/yl4579/StyleTTS2) training code to train the Kokoro encoders, but the codebase crashed frequently during adaptation. 

During my research, I discovered [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch), an excellent repository that patched the StyleTTS2 code to successfully skip diffuser sampling. I modified and adapted the training logic from the `semidark` repository to get this specific Kokoro encoder training pipeline working. 

Massive credit and thanks to the author of `semidark/kokoro-deutsch` for the patched StyleTTS2 baseline!
