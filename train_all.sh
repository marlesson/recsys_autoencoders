# /bin/bash

# 1. Popularity Model
mlflow run . -e popularity_train

# 2. CDAE - Collaborative Denoising Auto-Encoders for Top-N Recommender Systems
mlflow run . -P activation=selu -P batch=64 -P dropout=0.8 -P epochs=50 -P factors=500 -P lr=0.0001 -P name=cdae -P reg=0.0001

# 3. Deep AutoEncoder for Collaborative Filtering
mlflow run . -P activation=selu -P batch=64 -P dropout=0.8 -P epochs=50 -P layers='[512,256,512]' -P lr=0.0001 -P name=auto_enc_content -P reg=0.01

# 4. Deep AutoEncoder for Collaborative Filtering With Content Information
mlflow run . -P activation=selu -P batch=64 -P dropout=0.8 -P epochs=50 -P layers='[512,256,512]' -P lr=0.0001 -P name=auto_enc -P reg=0.01

