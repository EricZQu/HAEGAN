# Step 1: Train HJTAE

## Training the model

The configuration file is located in `../config.py`.

Training scipt:

```
python vae_train.py --train ../data/moses-processed --vocab ../data/vocab.txt --save_dir ../logs/
```

## Sample the embedding

After training, use the following to sample the hyperbolic emebdding: (please change)

```
python gen_embedding.py --model-path ../logs/xxx/model.iter-xxxx --config-path ../logs/xxx/config.json --save_dir ../2_HWGAN/Embed/
```