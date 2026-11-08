# Echo of Recitation (صدى التلاوة): Tajweed Rule Classification from Audio

**9th of 36 teams** on the final (private) leaderboard of the Olo Academy Kaggle competition, macro F1 **0.81**.

## The task
Each audio clip is a short Quran recitation containing a *noon sakinah* or *tanween*. The goal is to predict which tajweed rule applies:

| Rule | Next letter | Train clips |
|---|---|---|
| Izhar (إظهار) | ء هـ ع ح غ خ | 193 |
| Idgham (إدغام) | ي ر م ل و ن | 292 |
| Iqlab (إقلاب) | ب | 120 |
| Ikhfa (إخفاء) | the other 15 letters | 249 |

- 854 training clips from 4 reciters; 397 test clips from **4 reciters never seen in training**
- 16 kHz WAV audio; metric: macro F1

The main difficulty: a model that learns the reciters' voices will not generalize to new reciters.

## Approach
Instead of classifying the sound directly, the pipeline turns the audio into text and applies the tajweed rules to the text, which does not depend on who is reciting.

1. **Speech-to-text:** a Quran-tuned Whisper model (`tarteel-ai/whisper-base-ar-quran`) produces vowelled Arabic text, including sukun and tanween marks.
2. **Rule engine:** finds each noon sakinah / tanween in the text, looks at the following letter and scores evidence for each rule. It also handles silent alif/ya after tanween, the small-meem Iqlab mark, and meem + ب.
3. **Text classifier:** rule features + character n-grams (TF-IDF) → Logistic Regression with balanced class weights, as a fallback when the rules find nothing.
4. **Rule-first prediction:** when the rule engine finds evidence, use it; otherwise use the classifier.
5. **Hard clips:** test clips where the rule found nothing are transcribed again with a larger model (`openai/whisper-large-v3`) and passed through the rules a second time.

## Results
| Step | Result |
|---|---|
| Rules only, no training (train set) | macro F1 0.94 |
| Rules + text Logistic Regression (5-fold CV) | macro F1 0.976 |
| Test clips resolved directly by the rules | 79% |
| **Final submission, private leaderboard** | **0.81** (9th of 36) |

Submissions improved step by step: 0.76 → 0.80 → 0.81 on the private leaderboard.
Cross-validation is higher than the leaderboard because the training clips come from only 4 reciters, while the test reciters are new. Transcription errors on unfamiliar voices are the main source of mistakes.

## How to run
1. Open the notebook on Kaggle and attach the competition data.
2. Settings: **Accelerator = GPU T4**, **Internet = On** (the Whisper models download from Hugging Face).
3. Run all cells; the notebook writes `submission.csv`.

The competition data is not included in this repository.

## Files
- `echo_of_recitation_tajweed.ipynb`: full pipeline
- `requirements.txt`: Python packages

## Tools
Python, Hugging Face Transformers (Whisper), PyTorch, librosa, scikit-learn, pandas, NumPy, Kaggle Notebooks (GPU)
