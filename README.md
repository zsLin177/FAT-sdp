# Fast and Accurate Transition-based SDP

It is the source code of CCKS 2021 “基于转移的快速精准的语义依存图分析”.

## training

```shell
python -m supar.cmds.transition_based_semantic_dependency train -b \
                            -p exp/dual-k10-do0.2-acti-PAS-s2/model \
                            -d 5 \
                            --batch-size 3000 \
                            --decode_mode dual \
                            --seed 2 \
                            --dynamic \
                            --pro 0.2 \
                            --k 10 \
                            --train data/sdp/PAS/train.conllu \
                            --dev data/sdp/PAS/dev.conllu \
                            --test data/sdp/PAS/test.conllu \
                            --feat tag,char,lemma
```

* with "--dynamic" means training with dynamic
* --k 10 means dynamic training after 10 epochs
* --pro 0.2 means the probability to not take predicted actions
* -d 5 means using the 5th gpu

## evaluate

```shell
python -m supar.cmds.transition_based_semantic_dependency evaluate --data data/sdp/PAS_OOD/test.conllu \
                        -p exp/dual-k10-do0.2-acti-PAS-s2/model \
                        --decode_mode dual \
                        --batch_decode \
                        -d 5
```

* --batch_decode means using our batch decoding

## predict

```shell
python -m supar.cmds.transition_based_semantic_dependency predict --data data/sdp/PAS_OOD/test.conllu \
                        -p exp/dual-k10-do0.2-acti-PAS-s2/model \
                        --pred PAS_OOD_pred.conllu
                        --decode_mode dual \
                        -d 5
```

* --pred means saving the predicted results to the file PAS_OOD_pred.conllu

