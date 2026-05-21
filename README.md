# train-kokoro-encoder-styletts2

An experimental open-source attempt to retrain the missing **Kokoro TTS** encoders using a modified **StyleTTS2** framework to enable custom voice cloning.

## 📌 Project Purpose
While Kokoro TTS is built on the StyleTTS2 architecture, the official public release only included the decoder weights. This repository is an empirical effort to reconstruct and retrain the missing style and text encoders. 

>  The training pipeline successfully runs, but initial audio synthesis results are sub-optimal. The current hypothesis is that independently training the encoders without the decoders might be insufficient, and joint training is likely required. This is open-sourced to share findings.

## 🙏 Background & Credits
Initially, I attempted to use the vanilla [StyleTTS2](https://github.com/yl4579/StyleTTS2) training code to train the Kokoro encoders, but the codebase crashed frequently during adaptation. 

During my research, I discovered [semidark/kokoro-deutsch](https://github.com/semidark/kokoro-deutsch), an excellent repository that patched the StyleTTS2 code to successfully skip diffuser sampling. I modified and adapted the training logic from the `semidark` repository to get this specific Kokoro encoder training pipeline working. 

Massive credit and thanks to the author of `semidark/kokoro-deutsch` for the patched StyleTTS2 baseline!
