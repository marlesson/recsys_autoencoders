# /bin/bash

# mlflow run . -P activation=selu -P batch=64 -P dropout=0.8 -P epochs=100 -P layers='[512,256,512]' -P lr=0.0001 -P name=auto_enc_content -P reg=0.01

# mlflow run . -P activation=selu -P batch=64 -P dropout=0.8 -P epochs=100 -P layers='[512,256,512]' -P lr=0.0001 -P name=auto_enc -P reg=0.01

mlflow run . -P activation=selu -P batch=64 -P dropout=0.8 -P epochs=100 -P factors=256 -P lr=0.0001 -P name=cdae -P reg=0.0001

mlflow run . -P activation=selu -P batch=64 -P dropout=0.8 -P epochs=100 -P factors=1000 -P lr=0.0001 -P name=cdae -P reg=0.0001