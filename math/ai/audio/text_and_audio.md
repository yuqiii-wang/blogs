# Text and Audio

## General Process and Terminologies

### Audio to Text (A2T) / Automatic Speech Recognition (ASR)

Process:

1. Noise reduction
2. Spectrogram Extraction
3. Acoustic feature extraction to identify basic sound units called phonemes
4. A language model takes the sequence of identified phonemes and figures out the most probable sequence of words, e.g., identify to/too/two by grammar.

### Text to Audio (T2A) / Text-to-Speech (TTS)

Process:

1. Text Preprocessing (Normalization): The model cleans the input text, expanding abbreviations ("Dr." -> "Doctor")
2. Linguistic Feature Generation: The model converts the text into a sequence of linguistic features (phonemes, stress, intonation)
3. Waveform Generation: A vocoder (voice coder) takes these linguistic features and synthesizes the final audio waveform.

### Audio Tasks 

| Tasks                                | Model Architecture Type             | Comments                                                                                                                                    |
| :----------------------------------- | :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| Audio-to-Text (Transcription)        | Encoder-Decoder (or Encoder-only)   | -                                                                                                                                           |
| Text-to-Audio (Speech Generation)    | Decoder-Only / Language Model-based | It treats audio generation like predicting the next "audio token" in a sequence; To synthesize intelligible, natural-sounding human speech. |
| Text-to-Audio (General Sounds)       | Diffusion Models                    | To create any non-speech audio/foley, including sound effects, ambient noise, and music.                                                    |
| Audio-to-Audio (Translation/Editing) | Encoder-Decoder                     | -                                                                                                                                           |
| Understanding Audio (Classification) | Encoder-Only                        | outputs a classification or embedding that can be used for search (embedding similarity) or analysis.                                       |

where speech generation (TTS) vs general sound:

|                   | speech generation (TTS)                                    | general sound                                                                            |
| :---------------- | :--------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| Description       | To synthesize intelligible, natural-sounding human speech. | To create any non-speech audio/foley, including sound effects, ambient noise, and music. |
| Prompt and output | "Hello, how are you today?" -> "Good, thank you."          | "A dog barking in the distance on a windy day." -> [waveform: dog barking]               |
