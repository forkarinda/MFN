# MFN
Multistage Fusion with Forget Gate for Multimodal Summarization in Open-Domain Videos
## FRAMEWORK
it was implemented on RNN-based and transformer-based architectures, respectively
## START
Please go to folder MF_RNN or MF_transformer to start the experiment.
## DATA
The ASR transcription extracted by google-speech-v2 is available here.
## Evaluation
We follow the evaluation metrics provided by the how2 dataset, [nmtpytorch](https://github.com/srvk/how2-dataset), including BLEU{1-4}, ROUGE-L, METEOR, CIDEr.

As an alternative, [nlg-eval](https://github.com/Maluuba/nlg-eval) can obtain the same evaluation score as nmtpytorch.

In addition, [here](https://github.com/neural-dialogue-metrics/rouge) can be used to calculate the score of ROUGE series (such as ROUGE-N, ROUGE-L, ROUGE-W)
