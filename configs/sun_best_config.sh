bash kge_run.sh train <MODEL> <DATASET> <GPU_ID> <SEED> <BATCH_SIZE> <NEGATIVE_SAMPLE_SIZE> <HIDDEN_DIM> <GAMMA> <ALPHA> <LEARNING_RATE> <MAX_STEPS> <TEST_BATCH_SIZE> <ADDITIONAL_PARAMS>
:: << 'END'
    <MODEL>: Name of the embedding model (e.g., RotatE, pRotatE, TransE, ComplEx, DistMult).
    <DATASET>: Name of the dataset (e.g., FB15k, FB15k-237, wn18, wn18rr, countries_S1, YAGO3-10).
    <GPU_ID>: ID of the GPU to use (e.g., 0).
    <SEED>: Random seed for reproducibility (e.g., 0).
    <BATCH_SIZE>: Batch size for training (e.g., 256, 64, 1024).
    <NEGATIVE_SAMPLE_SIZE>: Number of negative samples (e.g., 1000, 500, 2000).
    <HIDDEN_DIM>: Dimensionality of the hidden layers (e.g., 1024, 512).
    <GAMMA>: Margin value for the loss function (e.g., 24.0, 9.0, 12.0, 0.1).
    <ALPHA>: Weight for the regularization term (e.g., 1.0, 0.5).
    <LEARNING_RATE>: Learning rate for optimization (e.g., 0.0001, 0.00005).
    <MAX_STEPS>: Maximum number of training steps (e.g., 150000, 100000, 80000, 40000).
    <TEST_BATCH_SIZE>: Batch size for testing (e.g., 16, 8, 4).
    <ADDITIONAL_PARAMS>: Additional parameters/flags (e.g., -de, -dr, --countries, -r 0.000002).
END

# Best Configuration for RotatE
#
bash kge_run.sh train RotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 -de
bash kge_run.sh train RotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16 -de
bash kge_run.sh train RotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8 -de
bash kge_run.sh train RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8 -de
bash kge_run.sh train RotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash kge_run.sh train RotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries 
bash kge_run.sh train RotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de --countries
bash kge_run.sh train RotatE YAGO3-10 0 0 1024 400 500 24.0 1.0 0.0002 100000 4 -de
#
# Best Configuration for pRotatE
#
bash kge_run.sh train pRotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash kge_run.sh train pRotatE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash kge_run.sh train pRotatE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash kge_run.sh train pRotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash kge_run.sh train pRotatE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash kge_run.sh train pRotatE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash kge_run.sh train pRotatE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# pRotatE example with Autoencoder (14th value should be True or False)
# Notice that at 15th argument we got the lambda value for the autoencoder
# Notice that at 16th argument we got the autoencoder hidden dimension
#
bash kge_run.sh train pRotatE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16 True 0.5 50
#
# Best Configuration for TransE
# 
bash kge_run.sh train TransE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash kge_run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash kge_run.sh train TransE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash kge_run.sh train TransE wn18rr 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
bash kge_run.sh train TransE countries_S1 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash kge_run.sh train TransE countries_S2 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
bash kge_run.sh train TransE countries_S3 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 --countries
#
# Best Configuration for ComplEx
# 
bash kge_run.sh train ComplEx FB15k 0 0 1024 256 1000 500.0 1.0 0.001 150000 16 -de -dr -r 0.000002
bash kge_run.sh train ComplEx FB15k-237 0 0 1024 256 1000 200.0 1.0 0.001 100000 16 -de -dr -r 0.00001
bash kge_run.sh train ComplEx wn18 0 0 512 1024 500 200.0 1.0 0.001 80000 8 -de -dr -r 0.00001
bash kge_run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.002 80000 8 -de -dr -r 0.000005
bash kge_run.sh train ComplEx countries_S1 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash kge_run.sh train ComplEx countries_S2 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
bash kge_run.sh train ComplEx countries_S3 0 0 512 64 1000 1.0 1.0 0.000002 40000 8 -de -dr -r 0.0005 --countries
#
# Best Configuration for DistMult
# 
bash kge_run.sh train DistMult FB15k 0 0 1024 256 2000 500.0 1.0 0.001 150000 16 -r 0.000002
bash kge_run.sh train DistMult FB15k-237 0 0 1024 256 2000 200.0 1.0 0.001 100000 16 -r 0.00001
bash kge_run.sh train DistMult wn18 0 0 512 1024 1000 200.0 1.0 0.001 80000 8 -r 0.00001
bash kge_run.sh train DistMult wn18rr 0 0 512 1024 1000 200.0 1.0 0.002 80000 8 -r 0.000005
bash kge_run.sh train DistMult countries_S1 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
bash kge_run.sh train DistMult countries_S2 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
bash kge_run.sh train DistMult countries_S3 0 0 512 64 2000 1.0 1.0 0.000002 40000 8 -r 0.0005 --countries
#
