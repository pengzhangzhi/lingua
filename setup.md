


#### Env

```
bash setup/create_env.sh
```


### dataset 
```
 python setup/download_prepare_hf_data.py   dclm_baseline_1.0_10prct 32 --data_dir /network/scratch/a/alexander.tong/lingua/data --nchunks 1

```

#### tokenizer
```cmd
 python setup/download_tokenizer.py llama3 /network/scratch/a/alexander.tong/lingua/tokenizer/ --api_key 

 ```


