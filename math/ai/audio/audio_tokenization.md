# Audio Tokenization and Mapping to phonemes /Words/Chars

## Traditional Solution: Acoustic + CTC + Lexicon

Before modern LLMs and Attention mechanisms existed, audio/text processes need diff models in diff phases.

### The ASR Pipeline (Audio -> Text)

0. Audio waveforms are split to *frames*
    Typically, 20 ms one frame, 50 frames per seconds.
1. The Acoustic Model (AM)
    Analyze the raw audio wave and guess what basic sounds (phonemes) or letters are being spoken.
    Output overlapping character guesses (K-K-K-[blank]-A-A-[blank]-T-T)
2. The CTC Aligner
    Aligns the hundreds of audio frames into a neat, discrete sequence of characters or phonemes: "C - A - T".
3. The Language Model (LM) & Lexicon

### The TTS Pipeline (Text -> Audio)

1. Grapheme-to-Phoneme (G2P)
    G2P model converts the English spelling into phonetic codes
2. Acoustic Synthesizer to generate *Mel-Spectrogram*
    A mel-spectrogram is essentially a 2D mathematical image or heat map of sound frequencies over time.
    The Acoustic Synthesizer decides the pitch, length, and volume of the word.
3. Vocal decode
    Translate the mel-spectrogram into standard waveform that hardware can play

## Attention Mechanism and Audio Tokenization

Instead of relying strictly on phonemes (the sounds of letters), modern models use Neural Audio Codecs.
The system passes the raw waveform through an audio encoder which compresses the sound into discrete, finite integers (tokens) using Vector Quantization (VQ).

* Semantic Codebooks
* Residual/Acoustic Codebooks: acoustic details