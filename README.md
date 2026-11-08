# Echo of Recitation (صدى التلاوة)

> Hear a Quran recitation clip and name the **tajweed rule** for *noon sakinah / tanween*: Izhar, Idgham, Iqlab or Ikhfa. The catch: the test reciters are **voices the model has never heard**.

🏆 **9th of 36 teams** · Olo Academy Kaggle competition (AIEP bootcamp, Olo Academy × MCIT)
📈 **Macro F1 0.81** on the final leaderboard · 0.976 in 5-fold CV

## The idea
A model that learns *voices* fails on new reciters. So instead of classifying sound, this solution **turns the audio into text** and reads the rule from the letter that follows the noon or tanween, exactly the way tajweed is taught.

```
audio ──► Whisper (Quran-tuned) ──► vowelled text ──► tajweed rule engine ──► rule
                                                        │ no evidence?
                                                        └──► text classifier / Whisper-large-v3 retry
```

| Next letter | Rule |
|---|---|
| ء هـ ع ح غ خ | Izhar |
| ي ر م ل و ن | Idgham |
| ب | Iqlab |
| the other 15 letters | Ikhfa |

## Results
| Step | Macro F1 |
|---|---|
| Rule engine alone, no training (train set) | 0.94 |
| Rules + character n-gram Logistic Regression (5-fold CV) | 0.976 |
| **Final leaderboard (unseen reciters)** | **0.81** |

Submissions improved 0.76 → 0.80 → 0.81. The gap from CV comes from transcription errors on new voices.

## Run it
1. Open `echo_of_recitation_tajweed.ipynb` on Kaggle and attach the competition data.
2. Settings: **GPU T4**, **Internet on**. Run all cells.

**Stack:** Python · Whisper (Hugging Face Transformers) · PyTorch · librosa · scikit-learn

Competition data is not included.

**Sarah Al-Bahesh** · [LinkedIn](https://www.linkedin.com/in/sarah-albahesh) · [GitHub](https://github.com/saravii7)
