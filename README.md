# MFN
Multistage Fusion with Forget Gate for Multimodal Summarization in Open-Domain Videos
## DATA
You can download the extracted ASR transcription from [google drive](https://drive.google.com/drive/folders/1A2jWqVbr-q_6UK7VBWTsn9fi_JpLX1PY?usp=sharing):
## START
Please go to folder MF_RNN or MF_transformer to start the experiment.

## Evaluation
Please use the [nmtpytorch](https://github.com/srvk/how2-dataset) evaluation library suggested by the How2 Challenge, which includes BLEU (1, 2, 3, 4), ROUGE-L, METEOR, and CIDEr evaluation metrics. 

As an alternative, [nlg-eval](https://github.com/Maluuba/nlg-eval) evaluation library can obtain the same evaluation score as nmtpytorch.

In addition, a [ROUGE](https://github.com/neural-dialogue-metrics/rouge) evaluation library can be used to calculate the ROUGE series (ROUGE-N, ROUGE-L, ROUGE-W) score.

