# Preparing
Make sure that you put the complete How2 summarization dataset in the "how2data" folder according to the format of the demo data in the "how2data" folder, then run the following commands:

```
python preprocess.py --train-prefix ../how2data/text/sum_train --valid-prefix ../how2data/text/sum_cv --test-prefix ../how2data/text/sum_devtest --dest-dir data-bin/how2
#python preprocess.py --train-prefix ../how2data/text/sum_asr_train --valid-prefix ../how2data/text/sum_asr_cv --test-prefix ../how2data/text/sum_asr_devtest --dest-dir data-bin/how2asr
```

# Training
You can run the following commands for training:

```
python train.py --data data-bin/how2 --train_video_file ../how2data/text/sum_train/tr_action.txt --val_video_file ../how2data/text/sum_cv/cv_action.txt --save-dir checkpoints/mfn 
#python train.py --data data-bin/how2asr --train_video_file ../how2data/text/sum_asr_train/tr_action.txt --val_video_file ../how2data/text/sum_asr_cv/cv_action.txt --save-dir checkpoints/asr_mfn 
```


# Prediction
You can run the following commands for prediction:

```
python generate.py --data data-bin/how2 --video_file ../how2data/text/sum_devtest/dete_action.txt --checkpoint-path checkpoints/mfn/your_checkpoint_file.pt > mfn.out
#python generate.py --data data-bin/how2asr --video_file ../how2data/text/sum_asr_devtest/dete_action.txt --checkpoint-path checkpoints/asr_mfn/your_checkpoint_file.pt > asr_mfn.out
```

Then you can run the following command to convert the output into the format that is convenient for evaluation:

```
grep ^H mfn.out | cut -f2- | sed -r 's/'$(echo -e "\033")'\[[0-9]{1,2}(;([0-9]{1,2})?)?[mK]//g' > mfn.sys
#grep ^H asr_mfn.out | cut -f2- | sed -r 's/'$(echo -e "\033")'\[[0-9]{1,2}(;([0-9]{1,2})?)?[mK]//g' > asr_mfn.sys
```





