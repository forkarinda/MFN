# Preparing
Put the complete How2 summarization dataset in the "how2data" folder according to the format of the demo data in the "how2data" folder.


# Training
You can run the following commands for training:

```
python nmtpy.py train --config configs/mfn_rnn.conf 
#python nmtpy.py train --config configs/asr_mfn_rnn.conf  ##for ASR transcript model
```


# Prediction
You can run the following commands for prediction:

```
python nmtpy.py translate model/mfn/mfn_rnn/your_checkpoint_file -o output
#python nmtpy.py translate model/asr_mfn/asr_mfn_rnn/your_checkpoint_file -o output
```
