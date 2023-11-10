# Results - notes for graph-mclcl
## General running notes
There are usually two stages: pretraining with our contrastive scheme which selects the two augmentations,
and then 
## Naming convention for results
dataset_name/run_name.txt
## Naming convention for a run
trainingScheme_eEpochs_fValidationFolds_aug1_aug2_mNumberOfMclIters \
For example, pretraining_e100_f3_maskN_mcl_m10.txt stands for:\
pretraining stage, with 100 epochs, a K-fold validation where k=3, using the
augmentations of maskN, mcl, with the mcl parmater of 10 iterations.