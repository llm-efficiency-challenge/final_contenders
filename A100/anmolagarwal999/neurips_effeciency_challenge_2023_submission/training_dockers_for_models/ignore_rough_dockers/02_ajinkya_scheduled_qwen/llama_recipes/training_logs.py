2023-10-10 06:16:15.851687: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Token will not been saved to git credential helper. Pass `add_to_git_credential=True` if you want to set the git credential as well.
Token is valid (permission: write).
Your token has been saved to /home/anmol/.cache/huggingface/token
Login successful

Loading checkpoint shards:   0%|          | 0/2 [00:00<?,
 ?it/s]
Loading checkpoint shards:  50%|█████     | 1/2 [00:01<00:01,
  1.60s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:02<00:00,
  1.08s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:02<00:00,
  1.16s/it]
/home/anmol/anaconda3/envs/wizard_coder/lib/python3.8/site-packages/peft/utils/other.py:133: FutureWarning: prepare_model_for_int8_training is deprecated and will be removed in a future version. Use prepare_model_for_kbit_training instead.
  warnings.warn(
/home/anmol/anaconda3/envs/wizard_coder/lib/python3.8/site-packages/torch/cuda/memory.py:329: FutureWarning: torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats,
 which resets /all/ peak memory stats.
  warnings.warn(
--> Model meta-llama/Llama-2-7b-hf

--> meta-llama/Llama-2-7b-hf has 262.41024 Million params

trainable params: 4,
194,
304 || all params: 6,
742,
609,
920 || trainable%: 0.06220594176090199
INSIDE INIT FUNCTION
--> Training Set Length = 1994
INSIDE INIT FUNCTION
--> Validation Set Length = 200

Training Epoch: 0:   0%|[34m          [0m| 0/249 [00:00<?,
 ?it/s]/home/anmol/anaconda3/envs/wizard_coder/lib/python3.8/site-packages/torch/utils/checkpoint.py:429: UserWarning: torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly. The default value of use_reentrant will be updated to be False in the future. To maintain current behavior,
 pass use_reentrant=True. It is recommended that you use use_reentrant=False. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
/home/anmol/anaconda3/envs/wizard_coder/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:322: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")

Training Epoch: 0:   0%|[34m          [0m| 0/249 [00:15<?,
 ?it/s]
Training Epoch: 0/3,
 step 0/249 completed (loss: 2.1333298683166504):   0%|[34m          [0m| 0/249 [00:15<?,
 ?it/s]
Training Epoch: 0/3,
 step 0/249 completed (loss: 2.1333298683166504):   0%|[34m          [0m| 1/249 [00:27<50:12,
 12.15s/it]
Training Epoch: 0/3,
 step 1/249 completed (loss: 3.4118824005126953):   0%|[34m          [0m| 1/249 [00:27<50:12,
 12.15s/it]
Training Epoch: 0/3,
 step 1/249 completed (loss: 3.4118824005126953):   1%|[34m          [0m| 3/249 [00:39<31:37,
  7.71s/it]
Training Epoch: 0/3,
 step 2/249 completed (loss: 2.866913318634033):   1%|[34m          [0m| 3/249 [00:39<31:37,
  7.71s/it] 
Training Epoch: 0/3,
 step 2/249 completed (loss: 2.866913318634033):   2%|[34m▏         [0m| 6/249 [00:51<22:18,
  5.51s/it]
Training Epoch: 0/3,
 step 3/249 completed (loss: 2.138425588607788):   2%|[34m▏         [0m| 6/249 [00:52<22:18,
  5.51s/it]
Training Epoch: 0/3,
 step 3/249 completed (loss: 2.138425588607788):   4%|[34m▍         [0m| 10/249 [01:04<16:44,
  4.20s/it]
Training Epoch: 0/3,
 step 4/249 completed (loss: 2.1441726684570312):   4%|[34m▍         [0m| 10/249 [01:04<16:44,
  4.20s/it]
Training Epoch: 0/3,
 step 4/249 completed (loss: 2.1441726684570312):   6%|[34m▌         [0m| 15/249 [01:16<13:04,
  3.35s/it]
Training Epoch: 0/3,
 step 5/249 completed (loss: 2.4053032398223877):   6%|[34m▌         [0m| 15/249 [01:16<13:04,
  3.35s/it]
Training Epoch: 0/3,
 step 5/249 completed (loss: 2.4053032398223877):   8%|[34m▊         [0m| 21/249 [01:28<10:29,
  2.76s/it]
Training Epoch: 0/3,
 step 6/249 completed (loss: 2.6290998458862305):   8%|[34m▊         [0m| 21/249 [01:29<10:29,
  2.76s/it]
Training Epoch: 0/3,
 step 6/249 completed (loss: 2.6290998458862305):  11%|[34m█         [0m| 28/249 [01:41<08:34,
  2.33s/it]
Training Epoch: 0/3,
 step 7/249 completed (loss: 0.9014130234718323):  11%|[34m█         [0m| 28/249 [01:41<08:34,
  2.33s/it]
Training Epoch: 0/3,
 step 7/249 completed (loss: 0.9014130234718323):  14%|[34m█▍        [0m| 36/249 [01:53<07:07,
  2.01s/it]
Training Epoch: 0/3,
 step 8/249 completed (loss: 1.4169257879257202):  14%|[34m█▍        [0m| 36/249 [01:53<07:07,
  2.01s/it]
Training Epoch: 0/3,
 step 8/249 completed (loss: 1.4169257879257202):  18%|[34m█▊        [0m| 45/249 [02:05<05:57,
  1.75s/it]
Training Epoch: 0/3,
 step 9/249 completed (loss: 0.9314072728157043):  18%|[34m█▊        [0m| 45/249 [02:06<05:57,
  1.75s/it]
Training Epoch: 0/3,
 step 9/249 completed (loss: 0.9314072728157043):  22%|[34m██▏       [0m| 55/249 [02:18<05:01,
  1.55s/it]
Training Epoch: 0/3,
 step 10/249 completed (loss: 1.1570155620574951):  22%|[34m██▏       [0m| 55/249 [02:18<05:01,
  1.55s/it]
Training Epoch: 0/3,
 step 10/249 completed (loss: 1.1570155620574951):  27%|[34m██▋       [0m| 66/249 [02:30<04:14,
  1.39s/it]
Training Epoch: 0/3,
 step 11/249 completed (loss: 0.6948252320289612):  27%|[34m██▋       [0m| 66/249 [02:30<04:14,
  1.39s/it]
Training Epoch: 0/3,
 step 11/249 completed (loss: 0.6948252320289612):  31%|[34m███▏      [0m| 78/249 [02:43<03:35,
  1.26s/it]
Training Epoch: 0/3,
 step 12/249 completed (loss: 0.80927574634552):  31%|[34m███▏      [0m| 78/249 [02:43<03:35,
  1.26s/it]  
Training Epoch: 0/3,
 step 12/249 completed (loss: 0.80927574634552):  37%|[34m███▋      [0m| 91/249 [02:55<03:01,
  1.15s/it]
Training Epoch: 0/3,
 step 13/249 completed (loss: 0.6976915001869202):  37%|[34m███▋      [0m| 91/249 [02:55<03:01,
  1.15s/it]
Training Epoch: 0/3,
 step 13/249 completed (loss: 0.6976915001869202):  42%|[34m████▏     [0m| 105/249 [03:07<02:31,
  1.05s/it]
Training Epoch: 0/3,
 step 14/249 completed (loss: 0.5221677422523499):  42%|[34m████▏     [0m| 105/249 [03:08<02:31,
  1.05s/it]
Training Epoch: 0/3,
 step 14/249 completed (loss: 0.5221677422523499):  48%|[34m████▊     [0m| 120/249 [03:20<02:05,
  1.03it/s]
Training Epoch: 0/3,
 step 15/249 completed (loss: 0.33695852756500244):  48%|[34m████▊     [0m| 120/249 [03:20<02:05,
  1.03it/s]
Training Epoch: 0/3,
 step 15/249 completed (loss: 0.33695852756500244):  55%|[34m█████▍    [0m| 136/249 [03:32<01:42,
  1.11it/s]
Training Epoch: 0/3,
 step 16/249 completed (loss: 0.329599529504776):  55%|[34m█████▍    [0m| 136/249 [03:32<01:42,
  1.11it/s]  
Training Epoch: 0/3,
 step 16/249 completed (loss: 0.329599529504776):  61%|[34m██████▏   [0m| 153/249 [03:45<01:20,
  1.19it/s]
Training Epoch: 0/3,
 step 17/249 completed (loss: 0.2877921164035797):  61%|[34m██████▏   [0m| 153/249 [03:45<01:20,
  1.19it/s]
Training Epoch: 0/3,
 step 17/249 completed (loss: 0.2877921164035797):  69%|[34m██████▊   [0m| 171/249 [03:57<01:01,
  1.27it/s]
Training Epoch: 0/3,
 step 18/249 completed (loss: 0.5064311027526855):  69%|[34m██████▊   [0m| 171/249 [03:57<01:01,
  1.27it/s]
Training Epoch: 0/3,
 step 18/249 completed (loss: 0.5064311027526855):  76%|[34m███████▋  [0m| 190/249 [04:09<00:43,
  1.34it/s]
Training Epoch: 0/3,
 step 19/249 completed (loss: 0.21894046664237976):  76%|[34m███████▋  [0m| 190/249 [04:10<00:43,
  1.34it/s]
Training Epoch: 0/3,
 step 19/249 completed (loss: 0.21894046664237976):  84%|[34m████████▍ [0m| 210/249 [04:22<00:27,
  1.43it/s]
Training Epoch: 0/3,
 step 20/249 completed (loss: 0.37807559967041016):  84%|[34m████████▍ [0m| 210/249 [04:22<00:27,
  1.43it/s]
Training Epoch: 0/3,
 step 20/249 completed (loss: 0.37807559967041016):  93%|[34m█████████▎[0m| 231/249 [04:34<00:11,
  1.51it/s]
Training Epoch: 0/3,
 step 21/249 completed (loss: 0.25385192036628723):  93%|[34m█████████▎[0m| 231/249 [04:34<00:11,
  1.51it/s]
Training Epoch: 0/3,
 step 21/249 completed (loss: 0.25385192036628723): : 253it [04:47,
  1.59it/s]                       
Training Epoch: 0/3,
 step 22/249 completed (loss: 0.23273351788520813): : 253it [04:47,
  1.59it/s]
Training Epoch: 0/3,
 step 22/249 completed (loss: 0.23273351788520813): : 276it [04:59,
  1.67it/s]
Training Epoch: 0/3,
 step 23/249 completed (loss: 0.302875816822052): : 276it [04:59,
  1.67it/s]  
Training Epoch: 0/3,
 step 23/249 completed (loss: 0.302875816822052): : 300it [05:11,
  1.75it/s]
Training Epoch: 0/3,
 step 24/249 completed (loss: 0.2855057120323181): : 300it [05:12,
  1.75it/s]
Training Epoch: 0/3,
 step 24/249 completed (loss: 0.2855057120323181): : 325it [05:24,
  1.83it/s]
Training Epoch: 0/3,
 step 25/249 completed (loss: 0.3093566298484802): : 325it [05:24,
  1.83it/s]
Training Epoch: 0/3,
 step 25/249 completed (loss: 0.3093566298484802): : 351it [05:36,
  1.91it/s]
Training Epoch: 0/3,
 step 26/249 completed (loss: 0.22448644042015076): : 351it [05:36,
  1.91it/s]
Training Epoch: 0/3,
 step 26/249 completed (loss: 0.22448644042015076): : 378it [05:49,
  1.99it/s]
Training Epoch: 0/3,
 step 27/249 completed (loss: 0.3216976523399353): : 378it [05:49,
  1.99it/s] 
Training Epoch: 0/3,
 step 27/249 completed (loss: 0.3216976523399353): : 406it [06:01,
  2.07it/s]
Training Epoch: 0/3,
 step 28/249 completed (loss: 0.1909751296043396): : 406it [06:01,
  2.07it/s]
Training Epoch: 0/3,
 step 28/249 completed (loss: 0.1909751296043396): : 435it [06:14,
  2.15it/s]
Training Epoch: 0/3,
 step 29/249 completed (loss: 0.28569403290748596): : 435it [06:14,
  2.15it/s]
Training Epoch: 0/3,
 step 29/249 completed (loss: 0.28569403290748596): : 465it [06:26,
  2.23it/s]
Training Epoch: 0/3,
 step 30/249 completed (loss: 0.3686480224132538): : 465it [06:26,
  2.23it/s] 
Training Epoch: 0/3,
 step 30/249 completed (loss: 0.3686480224132538): : 496it [06:38,
  2.31it/s]
Training Epoch: 0/3,
 step 31/249 completed (loss: 0.2956194579601288): : 496it [06:39,
  2.31it/s]
Training Epoch: 0/3,
 step 31/249 completed (loss: 0.2956194579601288): : 528it [06:51,
  2.39it/s]
Training Epoch: 0/3,
 step 32/249 completed (loss: 0.1707688868045807): : 528it [06:51,
  2.39it/s]
Training Epoch: 0/3,
 step 32/249 completed (loss: 0.1707688868045807): : 561it [07:03,
  2.47it/s]
Training Epoch: 0/3,
 step 33/249 completed (loss: 0.5126186013221741): : 561it [07:03,
  2.47it/s]
Training Epoch: 0/3,
 step 33/249 completed (loss: 0.5126186013221741): : 595it [07:16,
  2.55it/s]
Training Epoch: 0/3,
 step 34/249 completed (loss: 0.3168738782405853): : 595it [07:16,
  2.55it/s]
Training Epoch: 0/3,
 step 34/249 completed (loss: 0.3168738782405853): : 630it [07:28,
  2.63it/s]
Training Epoch: 0/3,
 step 35/249 completed (loss: 0.43196234107017517): : 630it [07:28,
  2.63it/s]
Training Epoch: 0/3,
 step 35/249 completed (loss: 0.43196234107017517): : 666it [07:41,
  2.71it/s]
Training Epoch: 0/3,
 step 36/249 completed (loss: 0.4325784742832184): : 666it [07:41,
  2.71it/s] 
Training Epoch: 0/3,
 step 36/249 completed (loss: 0.4325784742832184): : 703it [07:53,
  2.79it/s]
Training Epoch: 0/3,
 step 37/249 completed (loss: 0.28625401854515076): : 703it [07:53,
  2.79it/s]
Training Epoch: 0/3,
 step 37/249 completed (loss: 0.28625401854515076): : 741it [08:05,
  2.87it/s]
Training Epoch: 0/3,
 step 38/249 completed (loss: 0.3569069802761078): : 741it [08:06,
  2.87it/s] 
Training Epoch: 0/3,
 step 38/249 completed (loss: 0.3569069802761078): : 780it [08:18,
  2.95it/s]
Training Epoch: 0/3,
 step 39/249 completed (loss: 0.22486995160579681): : 780it [08:18,
  2.95it/s]
Training Epoch: 0/3,
 step 39/249 completed (loss: 0.22486995160579681): : 820it [08:30,
  3.03it/s]
Training Epoch: 0/3,
 step 40/249 completed (loss: 0.2336236536502838): : 820it [08:30,
  3.03it/s] 
Training Epoch: 0/3,
 step 40/249 completed (loss: 0.2336236536502838): : 861it [08:43,
  3.11it/s]
Training Epoch: 0/3,
 step 41/249 completed (loss: 0.18951089680194855): : 861it [08:43,
  3.11it/s]
Training Epoch: 0/3,
 step 41/249 completed (loss: 0.18951089680194855): : 903it [08:55,
  3.19it/s]
Training Epoch: 0/3,
 step 42/249 completed (loss: 0.1374029517173767): : 903it [08:55,
  3.19it/s] 
Training Epoch: 0/3,
 step 42/249 completed (loss: 0.1374029517173767): : 946it [09:08,
  3.27it/s]
Training Epoch: 0/3,
 step 43/249 completed (loss: 0.1986415684223175): : 946it [09:08,
  3.27it/s]
Training Epoch: 0/3,
 step 43/249 completed (loss: 0.1986415684223175): : 990it [09:20,
  3.35it/s]
Training Epoch: 0/3,
 step 44/249 completed (loss: 0.2947719097137451): : 990it [09:20,
  3.35it/s]
Training Epoch: 0/3,
 step 44/249 completed (loss: 0.2947719097137451): : 1035it [09:32,
  3.44it/s]
Training Epoch: 0/3,
 step 45/249 completed (loss: 0.134923055768013): : 1035it [09:33,
  3.44it/s] 
Training Epoch: 0/3,
 step 45/249 completed (loss: 0.134923055768013): : 1081it [09:45,
  3.52it/s]
Training Epoch: 0/3,
 step 46/249 completed (loss: 0.2545899450778961): : 1081it [09:45,
  3.52it/s]
Training Epoch: 0/3,
 step 46/249 completed (loss: 0.2545899450778961): : 1128it [09:57,
  3.59it/s]
Training Epoch: 0/3,
 step 47/249 completed (loss: 0.26104798913002014): : 1128it [09:57,
  3.59it/s]
Training Epoch: 0/3,
 step 47/249 completed (loss: 0.26104798913002014): : 1176it [10:10,
  3.68it/s]
Training Epoch: 0/3,
 step 48/249 completed (loss: 0.24439795315265656): : 1176it [10:10,
  3.68it/s]
Training Epoch: 0/3,
 step 48/249 completed (loss: 0.24439795315265656): : 1225it [10:22,
  3.76it/s]
Training Epoch: 0/3,
 step 49/249 completed (loss: 0.22636368870735168): : 1225it [10:22,
  3.76it/s]
Training Epoch: 0/3,
 step 49/249 completed (loss: 0.22636368870735168): : 1275it [10:34,
  3.84it/s]
Training Epoch: 0/3,
 step 50/249 completed (loss: 0.15773899853229523): : 1275it [10:35,
  3.84it/s]
Training Epoch: 0/3,
 step 50/249 completed (loss: 0.15773899853229523): : 1326it [10:47,
  3.92it/s]
Training Epoch: 0/3,
 step 51/249 completed (loss: 0.15044298768043518): : 1326it [10:47,
  3.92it/s]
Training Epoch: 0/3,
 step 51/249 completed (loss: 0.15044298768043518): : 1378it [10:59,
  4.00it/s]
Training Epoch: 0/3,
 step 52/249 completed (loss: 0.2328989952802658): : 1378it [10:59,
  4.00it/s] 
Training Epoch: 0/3,
 step 52/249 completed (loss: 0.2328989952802658): : 1431it [11:12,
  4.08it/s]
Training Epoch: 0/3,
 step 53/249 completed (loss: 0.25072285532951355): : 1431it [11:12,
  4.08it/s]
Training Epoch: 0/3,
 step 53/249 completed (loss: 0.25072285532951355): : 1485it [11:24,
  4.16it/s]
Training Epoch: 0/3,
 step 54/249 completed (loss: 0.335595041513443): : 1485it [11:24,
  4.16it/s]  
Training Epoch: 0/3,
 step 54/249 completed (loss: 0.335595041513443): : 1540it [11:37,
  4.24it/s]
Training Epoch: 0/3,
 step 55/249 completed (loss: 0.3742099702358246): : 1540it [11:37,
  4.24it/s]
Training Epoch: 0/3,
 step 55/249 completed (loss: 0.3742099702358246): : 1596it [11:49,
  4.32it/s]
Training Epoch: 0/3,
 step 56/249 completed (loss: 0.08182288706302643): : 1596it [11:49,
  4.32it/s]
Training Epoch: 0/3,
 step 56/249 completed (loss: 0.08182288706302643): : 1653it [12:01,
  4.40it/s]
Training Epoch: 0/3,
 step 57/249 completed (loss: 0.19403402507305145): : 1653it [12:02,
  4.40it/s]
Training Epoch: 0/3,
 step 57/249 completed (loss: 0.19403402507305145): : 1711it [12:14,
  4.48it/s]
Training Epoch: 0/3,
 step 58/249 completed (loss: 0.17416489124298096): : 1711it [12:14,
  4.48it/s]
Training Epoch: 0/3,
 step 58/249 completed (loss: 0.17416489124298096): : 1770it [12:26,
  4.56it/s]
Training Epoch: 0/3,
 step 59/249 completed (loss: 0.21221311390399933): : 1770it [12:26,
  4.56it/s]
Training Epoch: 0/3,
 step 59/249 completed (loss: 0.21221311390399933): : 1830it [12:39,
  4.65it/s]
Training Epoch: 0/3,
 step 60/249 completed (loss: 0.07237042486667633): : 1830it [12:39,
  4.65it/s]
Training Epoch: 0/3,
 step 60/249 completed (loss: 0.07237042486667633): : 1891it [12:51,
  4.72it/s]
Training Epoch: 0/3,
 step 61/249 completed (loss: 0.19095568358898163): : 1891it [12:51,
  4.72it/s]
Training Epoch: 0/3,
 step 61/249 completed (loss: 0.19095568358898163): : 1953it [13:03,
  4.80it/s]
Training Epoch: 0/3,
 step 62/249 completed (loss: 0.137324258685112): : 1953it [13:04,
  4.80it/s]  
Training Epoch: 0/3,
 step 62/249 completed (loss: 0.137324258685112): : 2016it [13:16,
  4.88it/s]
Training Epoch: 0/3,
 step 63/249 completed (loss: 0.16945599019527435): : 2016it [13:16,
  4.88it/s]
Training Epoch: 0/3,
 step 63/249 completed (loss: 0.16945599019527435): : 2080it [13:28,
  4.97it/s]
Training Epoch: 0/3,
 step 64/249 completed (loss: 0.08631503582000732): : 2080it [13:28,
  4.97it/s]
Training Epoch: 0/3,
 step 64/249 completed (loss: 0.08631503582000732): : 2145it [13:41,
  5.05it/s]
Training Epoch: 0/3,
 step 65/249 completed (loss: 0.1836252212524414): : 2145it [13:41,
  5.05it/s] 
Training Epoch: 0/3,
 step 65/249 completed (loss: 0.1836252212524414): : 2211it [13:53,
  5.13it/s]
Training Epoch: 0/3,
 step 66/249 completed (loss: 0.16588525474071503): : 2211it [13:53,
  5.13it/s]
Training Epoch: 0/3,
 step 66/249 completed (loss: 0.16588525474071503): : 2278it [14:06,
  5.21it/s]
Training Epoch: 0/3,
 step 67/249 completed (loss: 0.050743453204631805): : 2278it [14:06,
  5.21it/s]
Training Epoch: 0/3,
 step 67/249 completed (loss: 0.050743453204631805): : 2346it [14:18,
  5.29it/s]
Training Epoch: 0/3,
 step 68/249 completed (loss: 0.17745022475719452): : 2346it [14:18,
  5.29it/s] 
Training Epoch: 0/3,
 step 68/249 completed (loss: 0.17745022475719452): : 2415it [14:30,
  5.36it/s]
Training Epoch: 0/3,
 step 69/249 completed (loss: 0.16767194867134094): : 2415it [14:31,
  5.36it/s]
Training Epoch: 0/3,
 step 69/249 completed (loss: 0.16767194867134094): : 2485it [14:43,
  5.45it/s]
Training Epoch: 0/3,
 step 70/249 completed (loss: 0.1437397301197052): : 2485it [14:43,
  5.45it/s] 
Training Epoch: 0/3,
 step 70/249 completed (loss: 0.1437397301197052): : 2556it [14:55,
  5.53it/s]
Training Epoch: 0/3,
 step 71/249 completed (loss: 0.17440366744995117): : 2556it [14:55,
  5.53it/s]
Training Epoch: 0/3,
 step 71/249 completed (loss: 0.17440366744995117): : 2628it [15:08,
  5.61it/s]
Training Epoch: 0/3,
 step 72/249 completed (loss: 0.20031611621379852): : 2628it [15:08,
  5.61it/s]
Training Epoch: 0/3,
 step 72/249 completed (loss: 0.20031611621379852): : 2701it [15:20,
  5.69it/s]
Training Epoch: 0/3,
 step 73/249 completed (loss: 0.11284036189317703): : 2701it [15:20,
  5.69it/s]
Training Epoch: 0/3,
 step 73/249 completed (loss: 0.11284036189317703): : 2775it [15:33,
  5.77it/s]
Training Epoch: 0/3,
 step 74/249 completed (loss: 0.07691819220781326): : 2775it [15:33,
  5.77it/s]
Training Epoch: 0/3,
 step 74/249 completed (loss: 0.07691819220781326): : 2850it [15:45,
  5.85it/s]
Training Epoch: 0/3,
 step 75/249 completed (loss: 0.15185411274433136): : 2850it [15:45,
  5.85it/s]
Training Epoch: 0/3,
 step 75/249 completed (loss: 0.15185411274433136): : 2926it [15:57,
  5.93it/s]
Training Epoch: 0/3,
 step 76/249 completed (loss: 0.17135660350322723): : 2926it [15:58,
  5.93it/s]
Training Epoch: 0/3,
 step 76/249 completed (loss: 0.17135660350322723): : 3003it [16:10,
  6.02it/s]
Training Epoch: 0/3,
 step 77/249 completed (loss: 0.20289088785648346): : 3003it [16:10,
  6.02it/s]
Training Epoch: 0/3,
 step 77/249 completed (loss: 0.20289088785648346): : 3081it [16:22,
  6.10it/s]
Training Epoch: 0/3,
 step 78/249 completed (loss: 0.1140783280134201): : 3081it [16:22,
  6.10it/s] 
Training Epoch: 0/3,
 step 78/249 completed (loss: 0.1140783280134201): : 3160it [16:35,
  6.18it/s]
Training Epoch: 0/3,
 step 79/249 completed (loss: 0.1673893928527832): : 3160it [16:35,
  6.18it/s]
Training Epoch: 0/3,
 step 79/249 completed (loss: 0.1673893928527832): : 3240it [16:47,
  6.25it/s]
Training Epoch: 0/3,
 step 80/249 completed (loss: 0.135664165019989): : 3240it [16:47,
  6.25it/s] 
Training Epoch: 0/3,
 step 80/249 completed (loss: 0.135664165019989): : 3321it [16:59,
  6.33it/s]
Training Epoch: 0/3,
 step 81/249 completed (loss: 0.17091703414916992): : 3321it [17:00,
  6.33it/s]
Training Epoch: 0/3,
 step 81/249 completed (loss: 0.17091703414916992): : 3403it [17:12,
  6.41it/s]
Training Epoch: 0/3,
 step 82/249 completed (loss: 0.1384325623512268): : 3403it [17:12,
  6.41it/s] 
Training Epoch: 0/3,
 step 82/249 completed (loss: 0.1384325623512268): : 3486it [17:24,
  6.49it/s]
Training Epoch: 0/3,
 step 83/249 completed (loss: 0.11259009689092636): : 3486it [17:25,
  6.49it/s]
Training Epoch: 0/3,
 step 83/249 completed (loss: 0.11259009689092636): : 3570it [17:37,
  6.57it/s]
Training Epoch: 0/3,
 step 84/249 completed (loss: 0.17577703297138214): : 3570it [17:37,
  6.57it/s]
Training Epoch: 0/3,
 step 84/249 completed (loss: 0.17577703297138214): : 3655it [17:49,
  6.65it/s]
Training Epoch: 0/3,
 step 85/249 completed (loss: 0.13477212190628052): : 3655it [17:49,
  6.65it/s]
Training Epoch: 0/3,
 step 85/249 completed (loss: 0.13477212190628052): : 3741it [18:02,
  6.73it/s]
Training Epoch: 0/3,
 step 86/249 completed (loss: 0.09248295426368713): : 3741it [18:02,
  6.73it/s]
Training Epoch: 0/3,
 step 86/249 completed (loss: 0.09248295426368713): : 3828it [18:14,
  6.81it/s]
Training Epoch: 0/3,
 step 87/249 completed (loss: 0.12596248090267181): : 3828it [18:14,
  6.81it/s]
Training Epoch: 0/3,
 step 87/249 completed (loss: 0.12596248090267181): : 3916it [18:26,
  6.89it/s]
Training Epoch: 0/3,
 step 88/249 completed (loss: 0.08400600403547287): : 3916it [18:27,
  6.89it/s]
Training Epoch: 0/3,
 step 88/249 completed (loss: 0.08400600403547287): : 4005it [18:39,
  6.97it/s]
Training Epoch: 0/3,
 step 89/249 completed (loss: 0.09158090502023697): : 4005it [18:39,
  6.97it/s]
Training Epoch: 0/3,
 step 89/249 completed (loss: 0.09158090502023697): : 4095it [18:51,
  7.05it/s]
Training Epoch: 0/3,
 step 90/249 completed (loss: 0.07459362596273422): : 4095it [18:52,
  7.05it/s]
Training Epoch: 0/3,
 step 90/249 completed (loss: 0.07459362596273422): : 4186it [19:04,
  7.13it/s]
Training Epoch: 0/3,
 step 91/249 completed (loss: 0.23731303215026855): : 4186it [19:04,
  7.13it/s]
Training Epoch: 0/3,
 step 91/249 completed (loss: 0.23731303215026855): : 4278it [19:16,
  7.21it/s]
Training Epoch: 0/3,
 step 92/249 completed (loss: 0.13635414838790894): : 4278it [19:16,
  7.21it/s]
Training Epoch: 0/3,
 step 92/249 completed (loss: 0.13635414838790894): : 4371it [19:29,
  7.29it/s]
Training Epoch: 0/3,
 step 93/249 completed (loss: 0.23610667884349823): : 4371it [19:29,
  7.29it/s]
Training Epoch: 0/3,
 step 93/249 completed (loss: 0.23610667884349823): : 4465it [19:41,
  7.38it/s]
Training Epoch: 0/3,
 step 94/249 completed (loss: 0.16360773146152496): : 4465it [19:41,
  7.38it/s]
Training Epoch: 0/3,
 step 94/249 completed (loss: 0.16360773146152496): : 4560it [19:54,
  7.46it/s]
Training Epoch: 0/3,
 step 95/249 completed (loss: 0.13909131288528442): : 4560it [19:54,
  7.46it/s]
Training Epoch: 0/3,
 step 95/249 completed (loss: 0.13909131288528442): : 4656it [20:06,
  7.54it/s]
Training Epoch: 0/3,
 step 96/249 completed (loss: 0.19901123642921448): : 4656it [20:06,
  7.54it/s]
Training Epoch: 0/3,
 step 96/249 completed (loss: 0.19901123642921448): : 4753it [20:18,
  7.61it/s]
Training Epoch: 0/3,
 step 97/249 completed (loss: 0.16197706758975983): : 4753it [20:19,
  7.61it/s]
Training Epoch: 0/3,
 step 97/249 completed (loss: 0.16197706758975983): : 4851it [20:31,
  7.70it/s]
Training Epoch: 0/3,
 step 98/249 completed (loss: 0.1901259422302246): : 4851it [20:31,
  7.70it/s] 
Training Epoch: 0/3,
 step 98/249 completed (loss: 0.1901259422302246): : 4950it [20:43,
  7.78it/s]
Training Epoch: 0/3,
 step 99/249 completed (loss: 0.15232713520526886): : 4950it [20:43,
  7.78it/s]
Training Epoch: 0/3,
 step 99/249 completed (loss: 0.15232713520526886): : 5050it [20:56,
  7.86it/s]
Training Epoch: 0/3,
 step 100/249 completed (loss: 0.1549428254365921): : 5050it [20:56,
  7.86it/s]
Training Epoch: 0/3,
 step 100/249 completed (loss: 0.1549428254365921): : 5151it [21:08,
  7.94it/s]
Training Epoch: 0/3,
 step 101/249 completed (loss: 0.09909038245677948): : 5151it [21:08,
  7.94it/s]
Training Epoch: 0/3,
 step 101/249 completed (loss: 0.09909038245677948): : 5253it [21:21,
  8.02it/s]
Training Epoch: 0/3,
 step 102/249 completed (loss: 0.1623285561800003): : 5253it [21:21,
  8.02it/s] 
Training Epoch: 0/3,
 step 102/249 completed (loss: 0.1623285561800003): : 5356it [21:33,
  8.10it/s]
Training Epoch: 0/3,
 step 103/249 completed (loss: 0.134830042719841): : 5356it [21:33,
  8.10it/s] 
Training Epoch: 0/3,
 step 103/249 completed (loss: 0.134830042719841): : 5460it [21:45,
  8.19it/s]
Training Epoch: 0/3,
 step 104/249 completed (loss: 0.08979932963848114): : 5460it [21:46,
  8.19it/s]
Training Epoch: 0/3,
 step 104/249 completed (loss: 0.08979932963848114): : 5565it [21:58,
  8.27it/s]
Training Epoch: 0/3,
 step 105/249 completed (loss: 0.11947955936193466): : 5565it [21:58,
  8.27it/s]
Training Epoch: 0/3,
 step 105/249 completed (loss: 0.11947955936193466): : 5671it [22:10,
  8.35it/s]
Training Epoch: 0/3,
 step 106/249 completed (loss: 0.037673935294151306): : 5671it [22:10,
  8.35it/s]
Training Epoch: 0/3,
 step 106/249 completed (loss: 0.037673935294151306): : 5778it [22:23,
  8.42it/s]
Training Epoch: 0/3,
 step 107/249 completed (loss: 0.06963565945625305): : 5778it [22:23,
  8.42it/s] 
Training Epoch: 0/3,
 step 107/249 completed (loss: 0.06963565945625305): : 5886it [22:35,
  8.50it/s]
Training Epoch: 0/3,
 step 108/249 completed (loss: 0.1116911843419075): : 5886it [22:35,
  8.50it/s] 
Training Epoch: 0/3,
 step 108/249 completed (loss: 0.1116911843419075): : 5995it [22:47,
  8.59it/s]
Training Epoch: 0/3,
 step 109/249 completed (loss: 0.11460695415735245): : 5995it [22:48,
  8.59it/s]
Training Epoch: 0/3,
 step 109/249 completed (loss: 0.11460695415735245): : 6105it [23:00,
  8.66it/s]
Training Epoch: 0/3,
 step 110/249 completed (loss: 0.06477335095405579): : 6105it [23:00,
  8.66it/s]
Training Epoch: 0/3,
 step 110/249 completed (loss: 0.06477335095405579): : 6216it [23:12,
  8.74it/s]
Training Epoch: 0/3,
 step 111/249 completed (loss: 0.18070505559444427): : 6216it [23:13,
  8.74it/s]
Training Epoch: 0/3,
 step 111/249 completed (loss: 0.18070505559444427): : 6328it [23:25,
  8.82it/s]
Training Epoch: 0/3,
 step 112/249 completed (loss: 0.22890403866767883): : 6328it [23:25,
  8.82it/s]
Training Epoch: 0/3,
 step 112/249 completed (loss: 0.22890403866767883): : 6441it [23:37,
  8.90it/s]
Training Epoch: 0/3,
 step 113/249 completed (loss: 0.16392900049686432): : 6441it [23:37,
  8.90it/s]
Training Epoch: 0/3,
 step 113/249 completed (loss: 0.16392900049686432): : 6555it [23:50,
  8.98it/s]
Training Epoch: 0/3,
 step 114/249 completed (loss: 0.313656210899353): : 6555it [23:50,
  8.98it/s]  
Training Epoch: 0/3,
 step 114/249 completed (loss: 0.313656210899353): : 6670it [24:02,
  9.06it/s]
Training Epoch: 0/3,
 step 115/249 completed (loss: 0.12220147997140884): : 6670it [24:02,
  9.06it/s]
Training Epoch: 0/3,
 step 115/249 completed (loss: 0.12220147997140884): : 6786it [24:14,
  9.14it/s]
Training Epoch: 0/3,
 step 116/249 completed (loss: 0.15225215256214142): : 6786it [24:15,
  9.14it/s]
Training Epoch: 0/3,
 step 116/249 completed (loss: 0.15225215256214142): : 6903it [24:27,
  9.22it/s]
Training Epoch: 0/3,
 step 117/249 completed (loss: 0.11909294128417969): : 6903it [24:27,
  9.22it/s]
Training Epoch: 0/3,
 step 117/249 completed (loss: 0.11909294128417969): : 7021it [24:39,
  9.31it/s]
Training Epoch: 0/3,
 step 118/249 completed (loss: 0.06542760878801346): : 7021it [24:40,
  9.31it/s]
Training Epoch: 0/3,
 step 118/249 completed (loss: 0.06542760878801346): : 7140it [24:52,
  9.39it/s]
Training Epoch: 0/3,
 step 119/249 completed (loss: 0.19324471056461334): : 7140it [24:52,
  9.39it/s]
Training Epoch: 0/3,
 step 119/249 completed (loss: 0.19324471056461334): : 7260it [25:04,
  9.47it/s]
Training Epoch: 0/3,
 step 120/249 completed (loss: 0.1657474935054779): : 7260it [25:04,
  9.47it/s] 
Training Epoch: 0/3,
 step 120/249 completed (loss: 0.1657474935054779): : 7381it [25:17,
  9.55it/s]
Training Epoch: 0/3,
 step 121/249 completed (loss: 0.10812544077634811): : 7381it [25:17,
  9.55it/s]
Training Epoch: 0/3,
 step 121/249 completed (loss: 0.10812544077634811): : 7503it [25:29,
  9.63it/s]
Training Epoch: 0/3,
 step 122/249 completed (loss: 0.09933706372976303): : 7503it [25:29,
  9.63it/s]
Training Epoch: 0/3,
 step 122/249 completed (loss: 0.09933706372976303): : 7626it [25:41,
  9.71it/s]
Training Epoch: 0/3,
 step 123/249 completed (loss: 0.07622239738702774): : 7626it [25:42,
  9.71it/s]
Training Epoch: 0/3,
 step 123/249 completed (loss: 0.07622239738702774): : 7750it [25:54,
  9.79it/s]
Training Epoch: 0/3,
 step 124/249 completed (loss: 0.12086983025074005): : 7750it [25:54,
  9.79it/s]
Training Epoch: 0/3,
 step 124/249 completed (loss: 0.12086983025074005): : 7875it [26:06,
  9.87it/s]
Training Epoch: 0/3,
 step 125/249 completed (loss: 0.0626116544008255): : 7875it [26:06,
  9.87it/s] 
Training Epoch: 0/3,
 step 125/249 completed (loss: 0.0626116544008255): : 7875it [26:18,
  9.87it/s]
Training Epoch: 0/3,
 step 125/249 completed (loss: 0.0626116544008255): : 8001it [26:19,
  9.95it/s]
Training Epoch: 0/3,
 step 126/249 completed (loss: 0.07412618398666382): : 8001it [26:19,
  9.95it/s]
Training Epoch: 0/3,
 step 126/249 completed (loss: 0.07412618398666382): : 8001it [26:30,
  9.95it/s]
Training Epoch: 0/3,
 step 126/249 completed (loss: 0.07412618398666382): : 8128it [26:31,
 10.03it/s]
Training Epoch: 0/3,
 step 127/249 completed (loss: 0.04317738488316536): : 8128it [26:31,
 10.03it/s]
Training Epoch: 0/3,
 step 127/249 completed (loss: 0.04317738488316536): : 8256it [26:44,
 10.12it/s]
Training Epoch: 0/3,
 step 128/249 completed (loss: 0.04573668912053108): : 8256it [26:44,
 10.12it/s]
Training Epoch: 0/3,
 step 128/249 completed (loss: 0.04573668912053108): : 8385it [26:56,
 10.20it/s]
Training Epoch: 0/3,
 step 129/249 completed (loss: 0.10468948632478714): : 8385it [26:56,
 10.20it/s]
Training Epoch: 0/3,
 step 129/249 completed (loss: 0.10468948632478714): : 8385it [27:08,
 10.20it/s]
Training Epoch: 0/3,
 step 129/249 completed (loss: 0.10468948632478714): : 8515it [27:08,
 10.28it/s]
Training Epoch: 0/3,
 step 130/249 completed (loss: 0.017520125955343246): : 8515it [27:09,
 10.28it/s]
Training Epoch: 0/3,
 step 130/249 completed (loss: 0.017520125955343246): : 8515it [27:20,
 10.28it/s]
Training Epoch: 0/3,
 step 130/249 completed (loss: 0.017520125955343246): : 8646it [27:21,
 10.36it/s]
Training Epoch: 0/3,
 step 131/249 completed (loss: 0.1282440572977066): : 8646it [27:21,
 10.36it/s]  
Training Epoch: 0/3,
 step 131/249 completed (loss: 0.1282440572977066): : 8778it [27:33,
 10.44it/s]
Training Epoch: 0/3,
 step 132/249 completed (loss: 0.14272183179855347): : 8778it [27:33,
 10.44it/s]
Training Epoch: 0/3,
 step 132/249 completed (loss: 0.14272183179855347): : 8911it [27:46,
 10.51it/s]
Training Epoch: 0/3,
 step 133/249 completed (loss: 0.11112898588180542): : 8911it [27:46,
 10.51it/s]
Training Epoch: 0/3,
 step 133/249 completed (loss: 0.11112898588180542): : 8911it [27:58,
 10.51it/s]
Training Epoch: 0/3,
 step 133/249 completed (loss: 0.11112898588180542): : 9045it [27:58,
 10.59it/s]
Training Epoch: 0/3,
 step 134/249 completed (loss: 0.09141512960195541): : 9045it [27:58,
 10.59it/s]
Training Epoch: 0/3,
 step 134/249 completed (loss: 0.09141512960195541): : 9045it [28:10,
 10.59it/s]
Training Epoch: 0/3,
 step 134/249 completed (loss: 0.09141512960195541): : 9180it [28:11,
 10.68it/s]
Training Epoch: 0/3,
 step 135/249 completed (loss: 0.1317656934261322): : 9180it [28:11,
 10.68it/s] 
Training Epoch: 0/3,
 step 135/249 completed (loss: 0.1317656934261322): : 9316it [28:23,
 10.76it/s]
Training Epoch: 0/3,
 step 136/249 completed (loss: 0.11473194509744644): : 9316it [28:23,
 10.76it/s]
Training Epoch: 0/3,
 step 136/249 completed (loss: 0.11473194509744644): : 9453it [28:35,
 10.84it/s]
Training Epoch: 0/3,
 step 137/249 completed (loss: 0.15364162623882294): : 9453it [28:36,
 10.84it/s]
Training Epoch: 0/3,
 step 137/249 completed (loss: 0.15364162623882294): : 9453it [28:48,
 10.84it/s]
Training Epoch: 0/3,
 step 137/249 completed (loss: 0.15364162623882294): : 9591it [28:48,
 10.92it/s]
Training Epoch: 0/3,
 step 138/249 completed (loss: 0.10574652999639511): : 9591it [28:48,
 10.92it/s]
Training Epoch: 0/3,
 step 138/249 completed (loss: 0.10574652999639511): : 9591it [29:00,
 10.92it/s]
Training Epoch: 0/3,
 step 138/249 completed (loss: 0.10574652999639511): : 9730it [29:00,
 11.00it/s]
Training Epoch: 0/3,
 step 139/249 completed (loss: 0.04936787486076355): : 9730it [29:00,
 11.00it/s]
Training Epoch: 0/3,
 step 139/249 completed (loss: 0.04936787486076355): : 9870it [29:13,
 11.08it/s]
Training Epoch: 0/3,
 step 140/249 completed (loss: 0.03407103940844536): : 9870it [29:13,
 11.08it/s]
Training Epoch: 0/3,
 step 140/249 completed (loss: 0.03407103940844536): : 10011it [29:25,
 11.17it/s]
Training Epoch: 0/3,
 step 141/249 completed (loss: 0.10206428915262222): : 10011it [29:25,
 11.17it/s]
Training Epoch: 0/3,
 step 141/249 completed (loss: 0.10206428915262222): : 10153it [29:37,
 11.24it/s]
Training Epoch: 0/3,
 step 142/249 completed (loss: 0.10938520729541779): : 10153it [29:38,
 11.24it/s]
Training Epoch: 0/3,
 step 142/249 completed (loss: 0.10938520729541779): : 10153it [29:48,
 11.24it/s]
Training Epoch: 0/3,
 step 142/249 completed (loss: 0.10938520729541779): : 10296it [29:50,
 11.32it/s]
Training Epoch: 0/3,
 step 143/249 completed (loss: 0.050721775740385056): : 10296it [29:50,
 11.32it/s]
Training Epoch: 0/3,
 step 143/249 completed (loss: 0.050721775740385056): : 10440it [30:02,
 11.40it/s]
Training Epoch: 0/3,
 step 144/249 completed (loss: 0.11361044645309448): : 10440it [30:03,
 11.40it/s] 
Training Epoch: 0/3,
 step 144/249 completed (loss: 0.11361044645309448): : 10585it [30:15,
 11.50it/s]
Training Epoch: 0/3,
 step 145/249 completed (loss: 0.10665157437324524): : 10585it [30:15,
 11.50it/s]
Training Epoch: 0/3,
 step 145/249 completed (loss: 0.10665157437324524): : 10731it [30:27,
 11.57it/s]
Training Epoch: 0/3,
 step 146/249 completed (loss: 0.14311206340789795): : 10731it [30:27,
 11.57it/s]
Training Epoch: 0/3,
 step 146/249 completed (loss: 0.14311206340789795): : 10731it [30:38,
 11.57it/s]
Training Epoch: 0/3,
 step 146/249 completed (loss: 0.14311206340789795): : 10878it [30:40,
 11.65it/s]
Training Epoch: 0/3,
 step 147/249 completed (loss: 0.19857069849967957): : 10878it [30:40,
 11.65it/s]
Training Epoch: 0/3,
 step 147/249 completed (loss: 0.19857069849967957): : 11026it [30:52,
 11.74it/s]
Training Epoch: 0/3,
 step 148/249 completed (loss: 0.14702312648296356): : 11026it [30:52,
 11.74it/s]
Training Epoch: 0/3,
 step 148/249 completed (loss: 0.14702312648296356): : 11175it [31:04,
 11.81it/s]
Training Epoch: 0/3,
 step 149/249 completed (loss: 0.3602505922317505): : 11175it [31:05,
 11.81it/s] 
Training Epoch: 0/3,
 step 149/249 completed (loss: 0.3602505922317505): : 11325it [31:17,
 11.88it/s]
Training Epoch: 0/3,
 step 150/249 completed (loss: 0.1972191035747528): : 11325it [31:17,
 11.88it/s]
Training Epoch: 0/3,
 step 150/249 completed (loss: 0.1972191035747528): : 11325it [31:28,
 11.88it/s]
Training Epoch: 0/3,
 step 150/249 completed (loss: 0.1972191035747528): : 11476it [31:29,
 11.97it/s]
Training Epoch: 0/3,
 step 151/249 completed (loss: 0.07813943922519684): : 11476it [31:29,
 11.97it/s]
Training Epoch: 0/3,
 step 151/249 completed (loss: 0.07813943922519684): : 11476it [31:40,
 11.97it/s]
Training Epoch: 0/3,
 step 151/249 completed (loss: 0.07813943922519684): : 11628it [31:42,
 12.05it/s]
Training Epoch: 0/3,
 step 152/249 completed (loss: 0.1730862557888031): : 11628it [31:42,
 12.05it/s] 
Training Epoch: 0/3,
 step 152/249 completed (loss: 0.1730862557888031): : 11781it [31:54,
 12.12it/s]
Training Epoch: 0/3,
 step 153/249 completed (loss: 0.12212742865085602): : 11781it [31:54,
 12.12it/s]
Training Epoch: 0/3,
 step 153/249 completed (loss: 0.12212742865085602): : 11935it [32:07,
 12.20it/s]
Training Epoch: 0/3,
 step 154/249 completed (loss: 0.02895304188132286): : 11935it [32:07,
 12.20it/s]
Training Epoch: 0/3,
 step 154/249 completed (loss: 0.02895304188132286): : 11935it [32:18,
 12.20it/s]
Training Epoch: 0/3,
 step 154/249 completed (loss: 0.02895304188132286): : 12090it [32:19,
 12.29it/s]
Training Epoch: 0/3,
 step 155/249 completed (loss: 0.18783915042877197): : 12090it [32:19,
 12.29it/s]
Training Epoch: 0/3,
 step 155/249 completed (loss: 0.18783915042877197): : 12090it [32:30,
 12.29it/s]
Training Epoch: 0/3,
 step 155/249 completed (loss: 0.18783915042877197): : 12246it [32:31,
 12.37it/s]
Training Epoch: 0/3,
 step 156/249 completed (loss: 0.05669334530830383): : 12246it [32:32,
 12.37it/s]
Training Epoch: 0/3,
 step 156/249 completed (loss: 0.05669334530830383): : 12403it [32:44,
 12.46it/s]
Training Epoch: 0/3,
 step 157/249 completed (loss: 0.18351180851459503): : 12403it [32:44,
 12.46it/s]
Training Epoch: 0/3,
 step 157/249 completed (loss: 0.18351180851459503): : 12561it [32:56,
 12.54it/s]
Training Epoch: 0/3,
 step 158/249 completed (loss: 0.11614631116390228): : 12561it [32:56,
 12.54it/s]
Training Epoch: 0/3,
 step 158/249 completed (loss: 0.11614631116390228): : 12561it [33:08,
 12.54it/s]
Training Epoch: 0/3,
 step 158/249 completed (loss: 0.11614631116390228): : 12720it [33:09,
 12.62it/s]
Training Epoch: 0/3,
 step 159/249 completed (loss: 0.03063807263970375): : 12720it [33:09,
 12.62it/s]
Training Epoch: 0/3,
 step 159/249 completed (loss: 0.03063807263970375): : 12720it [33:20,
 12.62it/s]
Training Epoch: 0/3,
 step 159/249 completed (loss: 0.03063807263970375): : 12880it [33:21,
 12.69it/s]
Training Epoch: 0/3,
 step 160/249 completed (loss: 0.1076163500547409): : 12880it [33:21,
 12.69it/s] 
Training Epoch: 0/3,
 step 160/249 completed (loss: 0.1076163500547409): : 13041it [33:33,
 12.78it/s]
Training Epoch: 0/3,
 step 161/249 completed (loss: 0.11879182606935501): : 13041it [33:34,
 12.78it/s]
Training Epoch: 0/3,
 step 161/249 completed (loss: 0.11879182606935501): : 13203it [33:46,
 12.85it/s]
Training Epoch: 0/3,
 step 162/249 completed (loss: 0.06532404571771622): : 13203it [33:46,
 12.85it/s]
Training Epoch: 0/3,
 step 162/249 completed (loss: 0.06532404571771622): : 13203it [33:58,
 12.85it/s]
Training Epoch: 0/3,
 step 162/249 completed (loss: 0.06532404571771622): : 13366it [33:58,
 12.94it/s]
Training Epoch: 0/3,
 step 163/249 completed (loss: 0.16696485877037048): : 13366it [33:58,
 12.94it/s]
Training Epoch: 0/3,
 step 163/249 completed (loss: 0.16696485877037048): : 13366it [34:10,
 12.94it/s]
Training Epoch: 0/3,
 step 163/249 completed (loss: 0.16696485877037048): : 13530it [34:11,
 13.02it/s]
Training Epoch: 0/3,
 step 164/249 completed (loss: 0.0924934670329094): : 13530it [34:11,
 13.02it/s] 
Training Epoch: 0/3,
 step 164/249 completed (loss: 0.0924934670329094): : 13695it [34:23,
 13.11it/s]
Training Epoch: 0/3,
 step 165/249 completed (loss: 0.07423283904790878): : 13695it [34:23,
 13.11it/s]
Training Epoch: 0/3,
 step 165/249 completed (loss: 0.07423283904790878): : 13861it [34:36,
 13.19it/s]
Training Epoch: 0/3,
 step 166/249 completed (loss: 0.09975437819957733): : 13861it [34:36,
 13.19it/s]
Training Epoch: 0/3,
 step 166/249 completed (loss: 0.09975437819957733): : 13861it [34:48,
 13.19it/s]
Training Epoch: 0/3,
 step 166/249 completed (loss: 0.09975437819957733): : 14028it [34:48,
 13.27it/s]
Training Epoch: 0/3,
 step 167/249 completed (loss: 0.08201664686203003): : 14028it [34:48,
 13.27it/s]
Training Epoch: 0/3,
 step 167/249 completed (loss: 0.08201664686203003): : 14028it [35:00,
 13.27it/s]
Training Epoch: 0/3,
 step 167/249 completed (loss: 0.08201664686203003): : 14196it [35:00,
 13.34it/s]
Training Epoch: 0/3,
 step 168/249 completed (loss: 0.08609598875045776): : 14196it [35:01,
 13.34it/s]
Training Epoch: 0/3,
 step 168/249 completed (loss: 0.08609598875045776): : 14365it [35:13,
 13.43it/s]
Training Epoch: 0/3,
 step 169/249 completed (loss: 0.06750018894672394): : 14365it [35:13,
 13.43it/s]
Training Epoch: 0/3,
 step 169/249 completed (loss: 0.06750018894672394): : 14535it [35:25,
 13.50it/s]
Training Epoch: 0/3,
 step 170/249 completed (loss: 0.08720456808805466): : 14535it [35:25,
 13.50it/s]
Training Epoch: 0/3,
 step 170/249 completed (loss: 0.08720456808805466): : 14535it [35:38,
 13.50it/s]
Training Epoch: 0/3,
 step 170/249 completed (loss: 0.08720456808805466): : 14706it [35:38,
 13.58it/s]
Training Epoch: 0/3,
 step 171/249 completed (loss: 0.1629631370306015): : 14706it [35:38,
 13.58it/s] 
Training Epoch: 0/3,
 step 171/249 completed (loss: 0.1629631370306015): : 14706it [35:50,
 13.58it/s]
Training Epoch: 0/3,
 step 171/249 completed (loss: 0.1629631370306015): : 14878it [35:50,
 13.66it/s]
Training Epoch: 0/3,
 step 172/249 completed (loss: 0.08055311441421509): : 14878it [35:50,
 13.66it/s]
Training Epoch: 0/3,
 step 172/249 completed (loss: 0.08055311441421509): : 15051it [36:02,
 13.73it/s]
Training Epoch: 0/3,
 step 173/249 completed (loss: 0.06039528176188469): : 15051it [36:03,
 13.73it/s]
Training Epoch: 0/3,
 step 173/249 completed (loss: 0.06039528176188469): : 15225it [36:15,
 13.82it/s]
Training Epoch: 0/3,
 step 174/249 completed (loss: 0.15222184360027313): : 15225it [36:15,
 13.82it/s]
Training Epoch: 0/3,
 step 174/249 completed (loss: 0.15222184360027313): : 15400it [36:27,
 13.90it/s]
Training Epoch: 0/3,
 step 175/249 completed (loss: 0.16558139026165009): : 15400it [36:28,
 13.90it/s]
Training Epoch: 0/3,
 step 175/249 completed (loss: 0.16558139026165009): : 15400it [36:38,
 13.90it/s]
Training Epoch: 0/3,
 step 175/249 completed (loss: 0.16558139026165009): : 15576it [36:40,
 13.95it/s]
Training Epoch: 0/3,
 step 176/249 completed (loss: 0.18185484409332275): : 15576it [36:40,
 13.95it/s]
Training Epoch: 0/3,
 step 176/249 completed (loss: 0.18185484409332275): : 15753it [36:52,
 14.04it/s]
Training Epoch: 0/3,
 step 177/249 completed (loss: 0.07648938149213791): : 15753it [36:52,
 14.04it/s]
Training Epoch: 0/3,
 step 177/249 completed (loss: 0.07648938149213791): : 15931it [37:05,
 14.13it/s]
Training Epoch: 0/3,
 step 178/249 completed (loss: 0.15212789177894592): : 15931it [37:05,
 14.13it/s]
Training Epoch: 0/3,
 step 178/249 completed (loss: 0.15212789177894592): : 16110it [37:17,
 14.22it/s]
Training Epoch: 0/3,
 step 179/249 completed (loss: 0.05155893787741661): : 16110it [37:17,
 14.22it/s]
Training Epoch: 0/3,
 step 179/249 completed (loss: 0.05155893787741661): : 16110it [37:28,
 14.22it/s]
Training Epoch: 0/3,
 step 179/249 completed (loss: 0.05155893787741661): : 16290it [37:30,
 14.30it/s]
Training Epoch: 0/3,
 step 180/249 completed (loss: 0.1187143474817276): : 16290it [37:30,
 14.30it/s] 
Training Epoch: 0/3,
 step 180/249 completed (loss: 0.1187143474817276): : 16290it [37:40,
 14.30it/s]
Training Epoch: 0/3,
 step 180/249 completed (loss: 0.1187143474817276): : 16471it [37:42,
 14.38it/s]
Training Epoch: 0/3,
 step 181/249 completed (loss: 0.16381146013736725): : 16471it [37:42,
 14.38it/s]
Training Epoch: 0/3,
 step 181/249 completed (loss: 0.16381146013736725): : 16653it [37:54,
 14.47it/s]
Training Epoch: 0/3,
 step 182/249 completed (loss: 0.09552925825119019): : 16653it [37:54,
 14.47it/s]
Training Epoch: 0/3,
 step 182/249 completed (loss: 0.09552925825119019): : 16836it [38:07,
 14.55it/s]
Training Epoch: 0/3,
 step 183/249 completed (loss: 0.1638360619544983): : 16836it [38:07,
 14.55it/s] 
Training Epoch: 0/3,
 step 183/249 completed (loss: 0.1638360619544983): : 16836it [38:18,
 14.55it/s]
Training Epoch: 0/3,
 step 183/249 completed (loss: 0.1638360619544983): : 17020it [38:19,
 14.63it/s]
Training Epoch: 0/3,
 step 184/249 completed (loss: 0.10085363686084747): : 17020it [38:19,
 14.63it/s]
Training Epoch: 0/3,
 step 184/249 completed (loss: 0.10085363686084747): : 17020it [38:30,
 14.63it/s]
Training Epoch: 0/3,
 step 184/249 completed (loss: 0.10085363686084747): : 17205it [38:32,
 14.70it/s]
Training Epoch: 0/3,
 step 185/249 completed (loss: 0.16598355770111084): : 17205it [38:32,
 14.70it/s]
Training Epoch: 0/3,
 step 185/249 completed (loss: 0.16598355770111084): : 17391it [38:44,
 14.79it/s]
Training Epoch: 0/3,
 step 186/249 completed (loss: 0.14385122060775757): : 17391it [38:44,
 14.79it/s]
Training Epoch: 0/3,
 step 186/249 completed (loss: 0.14385122060775757): : 17578it [38:56,
 14.87it/s]
Training Epoch: 0/3,
 step 187/249 completed (loss: 0.08160606771707535): : 17578it [38:57,
 14.87it/s]
Training Epoch: 0/3,
 step 187/249 completed (loss: 0.08160606771707535): : 17578it [39:08,
 14.87it/s]
Training Epoch: 0/3,
 step 187/249 completed (loss: 0.08160606771707535): : 17766it [39:09,
 14.95it/s]
Training Epoch: 0/3,
 step 188/249 completed (loss: 0.09475404769182205): : 17766it [39:09,
 14.95it/s]
Training Epoch: 0/3,
 step 188/249 completed (loss: 0.09475404769182205): : 17766it [39:20,
 14.95it/s]
Training Epoch: 0/3,
 step 188/249 completed (loss: 0.09475404769182205): : 17955it [39:21,
 15.03it/s]
Training Epoch: 0/3,
 step 189/249 completed (loss: 0.11578863859176636): : 17955it [39:21,
 15.03it/s]
Training Epoch: 0/3,
 step 189/249 completed (loss: 0.11578863859176636): : 18145it [39:34,
 15.12it/s]
Training Epoch: 0/3,
 step 190/249 completed (loss: 0.06694785505533218): : 18145it [39:34,
 15.12it/s]
Training Epoch: 0/3,
 step 190/249 completed (loss: 0.06694785505533218): : 18336it [39:46,
 15.20it/s]
Training Epoch: 0/3,
 step 191/249 completed (loss: 0.12815627455711365): : 18336it [39:46,
 15.20it/s]
Training Epoch: 0/3,
 step 191/249 completed (loss: 0.12815627455711365): : 18336it [39:58,
 15.20it/s]
Training Epoch: 0/3,
 step 191/249 completed (loss: 0.12815627455711365): : 18528it [39:58,
 15.28it/s]
Training Epoch: 0/3,
 step 192/249 completed (loss: 0.2496267408132553): : 18528it [39:59,
 15.28it/s] 
Training Epoch: 0/3,
 step 192/249 completed (loss: 0.2496267408132553): : 18528it [40:10,
 15.28it/s]
Training Epoch: 0/3,
 step 192/249 completed (loss: 0.2496267408132553): : 18721it [40:11,
 15.36it/s]
Training Epoch: 0/3,
 step 193/249 completed (loss: 0.12504038214683533): : 18721it [40:11,
 15.36it/s]
Training Epoch: 0/3,
 step 193/249 completed (loss: 0.12504038214683533): : 18915it [40:23,
 15.40it/s]
Training Epoch: 0/3,
 step 194/249 completed (loss: 0.10914535075426102): : 18915it [40:24,
 15.40it/s]
Training Epoch: 0/3,
 step 194/249 completed (loss: 0.10914535075426102): : 19110it [40:36,
 15.49it/s]
Training Epoch: 0/3,
 step 195/249 completed (loss: 0.28568193316459656): : 19110it [40:36,
 15.49it/s]
Training Epoch: 0/3,
 step 195/249 completed (loss: 0.28568193316459656): : 19110it [40:48,
 15.49it/s]
Training Epoch: 0/3,
 step 195/249 completed (loss: 0.28568193316459656): : 19306it [40:48,
 15.58it/s]
Training Epoch: 0/3,
 step 196/249 completed (loss: 0.04407523199915886): : 19306it [40:48,
 15.58it/s]
Training Epoch: 0/3,
 step 196/249 completed (loss: 0.04407523199915886): : 19306it [41:00,
 15.58it/s]
Training Epoch: 0/3,
 step 196/249 completed (loss: 0.04407523199915886): : 19503it [41:01,
 15.66it/s]
Training Epoch: 0/3,
 step 197/249 completed (loss: 0.07508552819490433): : 19503it [41:01,
 15.66it/s]
Training Epoch: 0/3,
 step 197/249 completed (loss: 0.07508552819490433): : 19701it [41:13,
 15.75it/s]
Training Epoch: 0/3,
 step 198/249 completed (loss: 0.10374661535024643): : 19701it [41:13,
 15.75it/s]
Training Epoch: 0/3,
 step 198/249 completed (loss: 0.10374661535024643): : 19900it [41:25,
 15.83it/s]
Training Epoch: 0/3,
 step 199/249 completed (loss: 0.18676279485225677): : 19900it [41:26,
 15.83it/s]
Training Epoch: 0/3,
 step 199/249 completed (loss: 0.18676279485225677): : 19900it [41:38,
 15.83it/s]
Training Epoch: 0/3,
 step 199/249 completed (loss: 0.18676279485225677): : 20100it [41:38,
 15.91it/s]
Training Epoch: 0/3,
 step 200/249 completed (loss: 0.03702868893742561): : 20100it [41:38,
 15.91it/s]
Training Epoch: 0/3,
 step 200/249 completed (loss: 0.03702868893742561): : 20100it [41:50,
 15.91it/s]
Training Epoch: 0/3,
 step 200/249 completed (loss: 0.03702868893742561): : 20301it [41:50,
 15.99it/s]
Training Epoch: 0/3,
 step 201/249 completed (loss: 0.10313976556062698): : 20301it [41:51,
 15.99it/s]
Training Epoch: 0/3,
 step 201/249 completed (loss: 0.10313976556062698): : 20503it [42:03,
 16.07it/s]
Training Epoch: 0/3,
 step 202/249 completed (loss: 0.05974183976650238): : 20503it [42:03,
 16.07it/s]
Training Epoch: 0/3,
 step 202/249 completed (loss: 0.05974183976650238): : 20706it [42:15,
 16.15it/s]
Training Epoch: 0/3,
 step 203/249 completed (loss: 0.03233979269862175): : 20706it [42:15,
 16.15it/s]
Training Epoch: 0/3,
 step 203/249 completed (loss: 0.03233979269862175): : 20706it [42:28,
 16.15it/s]
Training Epoch: 0/3,
 step 203/249 completed (loss: 0.03233979269862175): : 20910it [42:28,
 16.24it/s]
Training Epoch: 0/3,
 step 204/249 completed (loss: 0.029227910563349724): : 20910it [42:28,
 16.24it/s]
Training Epoch: 0/3,
 step 204/249 completed (loss: 0.029227910563349724): : 20910it [42:40,
 16.24it/s]
Training Epoch: 0/3,
 step 204/249 completed (loss: 0.029227910563349724): : 21115it [42:40,
 16.33it/s]
Training Epoch: 0/3,
 step 205/249 completed (loss: 0.0688604936003685): : 21115it [42:40,
 16.33it/s]  
Training Epoch: 0/3,
 step 205/249 completed (loss: 0.0688604936003685): : 21321it [42:52,
 16.41it/s]
Training Epoch: 0/3,
 step 206/249 completed (loss: 0.016773121431469917): : 21321it [42:53,
 16.41it/s]
Training Epoch: 0/3,
 step 206/249 completed (loss: 0.016773121431469917): : 21528it [43:05,
 16.48it/s]
Training Epoch: 0/3,
 step 207/249 completed (loss: 0.053338054567575455): : 21528it [43:05,
 16.48it/s]
Training Epoch: 0/3,
 step 207/249 completed (loss: 0.053338054567575455): : 21736it [43:17,
 16.56it/s]
Training Epoch: 0/3,
 step 208/249 completed (loss: 0.10922181606292725): : 21736it [43:17,
 16.56it/s] 
Training Epoch: 0/3,
 step 208/249 completed (loss: 0.10922181606292725): : 21736it [43:28,
 16.56it/s]
Training Epoch: 0/3,
 step 208/249 completed (loss: 0.10922181606292725): : 21945it [43:30,
 16.64it/s]
Training Epoch: 0/3,
 step 209/249 completed (loss: 0.20651808381080627): : 21945it [43:30,
 16.64it/s]
Training Epoch: 0/3,
 step 209/249 completed (loss: 0.20651808381080627): : 22155it [43:42,
 16.73it/s]
Training Epoch: 0/3,
 step 210/249 completed (loss: 0.13798147439956665): : 22155it [43:42,
 16.73it/s]
Training Epoch: 0/3,
 step 210/249 completed (loss: 0.13798147439956665): : 22366it [43:55,
 16.81it/s]
Training Epoch: 0/3,
 step 211/249 completed (loss: 0.08369633555412292): : 22366it [43:55,
 16.81it/s]
Training Epoch: 0/3,
 step 211/249 completed (loss: 0.08369633555412292): : 22578it [44:07,
 16.88it/s]
Training Epoch: 0/3,
 step 212/249 completed (loss: 0.10865089297294617): : 22578it [44:07,
 16.88it/s]
Training Epoch: 0/3,
 step 212/249 completed (loss: 0.10865089297294617): : 22578it [44:18,
 16.88it/s]
Training Epoch: 0/3,
 step 212/249 completed (loss: 0.10865089297294617): : 22791it [44:19,
 16.96it/s]
Training Epoch: 0/3,
 step 213/249 completed (loss: 0.1071707010269165): : 22791it [44:20,
 16.96it/s] 
Training Epoch: 0/3,
 step 213/249 completed (loss: 0.1071707010269165): : 22791it [44:30,
 16.96it/s]
Training Epoch: 0/3,
 step 213/249 completed (loss: 0.1071707010269165): : 23005it [44:32,
 17.03it/s]
Training Epoch: 0/3,
 step 214/249 completed (loss: 0.037117306143045425): : 23005it [44:32,
 17.03it/s]
Training Epoch: 0/3,
 step 214/249 completed (loss: 0.037117306143045425): : 23220it [44:44,
 17.07it/s]
Training Epoch: 0/3,
 step 215/249 completed (loss: 0.0990334153175354): : 23220it [44:45,
 17.07it/s]  
Training Epoch: 0/3,
 step 215/249 completed (loss: 0.0990334153175354): : 23436it [44:57,
 17.07it/s]
Training Epoch: 0/3,
 step 216/249 completed (loss: 0.11622027307748795): : 23436it [44:57,
 17.07it/s]
Training Epoch: 0/3,
 step 216/249 completed (loss: 0.11622027307748795): : 23436it [45:08,
 17.07it/s]
Training Epoch: 0/3,
 step 216/249 completed (loss: 0.11622027307748795): : 23653it [45:10,
 17.10it/s]
Training Epoch: 0/3,
 step 217/249 completed (loss: 0.08171793073415756): : 23653it [45:10,
 17.10it/s]
Training Epoch: 0/3,
 step 217/249 completed (loss: 0.08171793073415756): : 23871it [45:22,
 17.15it/s]
Training Epoch: 0/3,
 step 218/249 completed (loss: 0.13853850960731506): : 23871it [45:22,
 17.15it/s]
Training Epoch: 0/3,
 step 218/249 completed (loss: 0.13853850960731506): : 24090it [45:35,
 17.19it/s]
Training Epoch: 0/3,
 step 219/249 completed (loss: 0.09200301766395569): : 24090it [45:35,
 17.19it/s]
Training Epoch: 0/3,
 step 219/249 completed (loss: 0.09200301766395569): : 24310it [45:48,
 17.27it/s]
Training Epoch: 0/3,
 step 220/249 completed (loss: 0.16343191266059875): : 24310it [45:48,
 17.27it/s]
Training Epoch: 0/3,
 step 220/249 completed (loss: 0.16343191266059875): : 24310it [45:58,
 17.27it/s]
Training Epoch: 0/3,
 step 220/249 completed (loss: 0.16343191266059875): : 24531it [46:00,
 17.36it/s]
Training Epoch: 0/3,
 step 221/249 completed (loss: 0.04676763340830803): : 24531it [46:00,
 17.36it/s]
Training Epoch: 0/3,
 step 221/249 completed (loss: 0.04676763340830803): : 24753it [46:13,
 17.42it/s]
Training Epoch: 0/3,
 step 222/249 completed (loss: 0.06149844825267792): : 24753it [46:13,
 17.42it/s]
Training Epoch: 0/3,
 step 222/249 completed (loss: 0.06149844825267792): : 24976it [46:25,
 17.48it/s]
Training Epoch: 0/3,
 step 223/249 completed (loss: 0.10476559400558472): : 24976it [46:26,
 17.48it/s]
Training Epoch: 0/3,
 step 223/249 completed (loss: 0.10476559400558472): : 24976it [46:38,
 17.48it/s]
Training Epoch: 0/3,
 step 223/249 completed (loss: 0.10476559400558472): : 25200it [46:38,
 17.55it/s]
Training Epoch: 0/3,
 step 224/249 completed (loss: 0.09394311904907227): : 25200it [46:38,
 17.55it/s]
Training Epoch: 0/3,
 step 224/249 completed (loss: 0.09394311904907227): : 25200it [46:50,
 17.55it/s]
Training Epoch: 0/3,
 step 224/249 completed (loss: 0.09394311904907227): : 25425it [46:51,
 17.65it/s]
Training Epoch: 0/3,
 step 225/249 completed (loss: 0.08185150474309921): : 25425it [46:51,
 17.65it/s]
Training Epoch: 0/3,
 step 225/249 completed (loss: 0.08185150474309921): : 25651it [47:03,
 17.74it/s]
Training Epoch: 0/3,
 step 226/249 completed (loss: 0.12290354073047638): : 25651it [47:03,
 17.74it/s]
Training Epoch: 0/3,
 step 226/249 completed (loss: 0.12290354073047638): : 25878it [47:16,
 17.85it/s]
Training Epoch: 0/3,
 step 227/249 completed (loss: 0.14589940011501312): : 25878it [47:16,
 17.85it/s]
Training Epoch: 0/3,
 step 227/249 completed (loss: 0.14589940011501312): : 25878it [47:28,
 17.85it/s]
Training Epoch: 0/3,
 step 227/249 completed (loss: 0.14589940011501312): : 26106it [47:28,
 17.92it/s]
Training Epoch: 0/3,
 step 228/249 completed (loss: 0.08260603249073029): : 26106it [47:29,
 17.92it/s]
Training Epoch: 0/3,
 step 228/249 completed (loss: 0.08260603249073029): : 26106it [47:40,
 17.92it/s]
Training Epoch: 0/3,
 step 228/249 completed (loss: 0.08260603249073029): : 26335it [47:41,
 17.97it/s]
Training Epoch: 0/3,
 step 229/249 completed (loss: 0.08627097308635712): : 26335it [47:41,
 17.97it/s]
Training Epoch: 0/3,
 step 229/249 completed (loss: 0.08627097308635712): : 26565it [47:54,
 18.00it/s]
Training Epoch: 0/3,
 step 230/249 completed (loss: 0.06685362756252289): : 26565it [47:54,
 18.00it/s]
Training Epoch: 0/3,
 step 230/249 completed (loss: 0.06685362756252289): : 26796it [48:06,
 18.06it/s]
Training Epoch: 0/3,
 step 231/249 completed (loss: 0.023895304650068283): : 26796it [48:07,
 18.06it/s]
Training Epoch: 0/3,
 step 231/249 completed (loss: 0.023895304650068283): : 26796it [48:18,
 18.06it/s]
Training Epoch: 0/3,
 step 231/249 completed (loss: 0.023895304650068283): : 27028it [48:19,
 18.19it/s]
Training Epoch: 0/3,
 step 232/249 completed (loss: 0.1096002459526062): : 27028it [48:19,
 18.19it/s]  
Training Epoch: 0/3,
 step 232/249 completed (loss: 0.1096002459526062): : 27028it [48:30,
 18.19it/s]
Training Epoch: 0/3,
 step 232/249 completed (loss: 0.1096002459526062): : 27261it [48:32,
 18.30it/s]
Training Epoch: 0/3,
 step 233/249 completed (loss: 0.04756787419319153): : 27261it [48:32,
 18.30it/s]
Training Epoch: 0/3,
 step 233/249 completed (loss: 0.04756787419319153): : 27495it [48:44,
 18.39it/s]
Training Epoch: 0/3,
 step 234/249 completed (loss: 0.06762165576219559): : 27495it [48:44,
 18.39it/s]
Training Epoch: 0/3,
 step 234/249 completed (loss: 0.06762165576219559): : 27730it [48:57,
 18.50it/s]
Training Epoch: 0/3,
 step 235/249 completed (loss: 0.13577665388584137): : 27730it [48:57,
 18.50it/s]
Training Epoch: 0/3,
 step 235/249 completed (loss: 0.13577665388584137): : 27730it [49:08,
 18.50it/s]
Training Epoch: 0/3,
 step 235/249 completed (loss: 0.13577665388584137): : 27966it [49:09,
 18.56it/s]
Training Epoch: 0/3,
 step 236/249 completed (loss: 0.21618010103702545): : 27966it [49:09,
 18.56it/s]
Training Epoch: 0/3,
 step 236/249 completed (loss: 0.21618010103702545): : 27966it [49:20,
 18.56it/s]
Training Epoch: 0/3,
 step 236/249 completed (loss: 0.21618010103702545): : 28203it [49:22,
 18.61it/s]
Training Epoch: 0/3,
 step 237/249 completed (loss: 0.13371382653713226): : 28203it [49:22,
 18.61it/s]
Training Epoch: 0/3,
 step 237/249 completed (loss: 0.13371382653713226): : 28441it [49:34,
 18.72it/s]
Training Epoch: 0/3,
 step 238/249 completed (loss: 0.10637539625167847): : 28441it [49:35,
 18.72it/s]
Training Epoch: 0/3,
 step 238/249 completed (loss: 0.10637539625167847): : 28680it [49:47,
 18.87it/s]
Training Epoch: 0/3,
 step 239/249 completed (loss: 0.05020677670836449): : 28680it [49:47,
 18.87it/s]
Training Epoch: 0/3,
 step 239/249 completed (loss: 0.05020677670836449): : 28680it [49:58,
 18.87it/s]
Training Epoch: 0/3,
 step 239/249 completed (loss: 0.05020677670836449): : 28920it [49:59,
 19.00it/s]
Training Epoch: 0/3,
 step 240/249 completed (loss: 0.10517366975545883): : 28920it [50:00,
 19.00it/s]
Training Epoch: 0/3,
 step 240/249 completed (loss: 0.10517366975545883): : 28920it [50:10,
 19.00it/s]
Training Epoch: 0/3,
 step 240/249 completed (loss: 0.10517366975545883): : 29161it [50:12,
 19.12it/s]
Training Epoch: 0/3,
 step 241/249 completed (loss: 0.284273236989975): : 29161it [50:12,
 19.12it/s]  
Training Epoch: 0/3,
 step 241/249 completed (loss: 0.284273236989975): : 29403it [50:24,
 19.24it/s]
Training Epoch: 0/3,
 step 242/249 completed (loss: 0.04313541576266289): : 29403it [50:24,
 19.24it/s]
Training Epoch: 0/3,
 step 242/249 completed (loss: 0.04313541576266289): : 29646it [50:37,
 19.35it/s]
Training Epoch: 0/3,
 step 243/249 completed (loss: 0.04680834338068962): : 29646it [50:37,
 19.35it/s]
Training Epoch: 0/3,
 step 243/249 completed (loss: 0.04680834338068962): : 29646it [50:48,
 19.35it/s]
Training Epoch: 0/3,
 step 243/249 completed (loss: 0.04680834338068962): : 29890it [50:49,
 19.44it/s]
Training Epoch: 0/3,
 step 244/249 completed (loss: 0.11367843300104141): : 29890it [50:49,
 19.44it/s]
Training Epoch: 0/3,
 step 244/249 completed (loss: 0.11367843300104141): : 29890it [51:00,
 19.44it/s]
Training Epoch: 0/3,
 step 244/249 completed (loss: 0.11367843300104141): : 30135it [51:01,
 19.54it/s]
Training Epoch: 0/3,
 step 245/249 completed (loss: 0.17806924879550934): : 30135it [51:02,
 19.54it/s]
Training Epoch: 0/3,
 step 245/249 completed (loss: 0.17806924879550934): : 30381it [51:14,
 19.63it/s]
Training Epoch: 0/3,
 step 246/249 completed (loss: 0.10195297002792358): : 30381it [51:14,
 19.63it/s]
Training Epoch: 0/3,
 step 246/249 completed (loss: 0.10195297002792358): : 30628it [51:26,
 19.71it/s]
Training Epoch: 0/3,
 step 247/249 completed (loss: 0.11111065745353699): : 30628it [51:26,
 19.71it/s]
Training Epoch: 0/3,
 step 247/249 completed (loss: 0.11111065745353699): : 30628it [51:38,
 19.71it/s]
Training Epoch: 0/3,
 step 247/249 completed (loss: 0.11111065745353699): : 30876it [51:39,
 19.80it/s]
Training Epoch: 0/3,
 step 248/249 completed (loss: 0.024863319471478462): : 30876it [51:39,
 19.80it/s]Max CUDA memory allocated was 9 GB
Max CUDA memory reserved was 12 GB
Peak active CUDA memory was 9 GB
Cuda Malloc retires : 0
CPU Total Peak Memory consumed during the train (max): 8 GB


evaluating Epoch:   0%|[32m          [0m| 0/200 [00:00<?,
 ?it/s][A

evaluating Epoch:   0%|[32m          [0m| 1/200 [00:00<02:11,
  1.51it/s][A

evaluating Epoch:   1%|[32m          [0m| 2/200 [00:01<02:00,
  1.64it/s][A

evaluating Epoch:   2%|[32m▏         [0m| 3/200 [00:01<01:56,
  1.69it/s][A

evaluating Epoch:   2%|[32m▏         [0m| 4/200 [00:02<01:54,
  1.71it/s][A

evaluating Epoch:   2%|[32m▎         [0m| 5/200 [00:02<01:52,
  1.73it/s][A

evaluating Epoch:   3%|[32m▎         [0m| 6/200 [00:03<01:51,
  1.74it/s][A

evaluating Epoch:   4%|[32m▎         [0m| 7/200 [00:04<01:51,
  1.74it/s][A

evaluating Epoch:   4%|[32m▍         [0m| 8/200 [00:04<01:49,
  1.76it/s][A

evaluating Epoch:   4%|[32m▍         [0m| 9/200 [00:05<01:48,
  1.77it/s][A

evaluating Epoch:   5%|[32m▌         [0m| 10/200 [00:05<01:46,
  1.78it/s][A

evaluating Epoch:   6%|[32m▌         [0m| 11/200 [00:06<01:46,
  1.77it/s][A

evaluating Epoch:   6%|[32m▌         [0m| 12/200 [00:06<01:44,
  1.79it/s][A

evaluating Epoch:   6%|[32m▋         [0m| 13/200 [00:07<01:44,
  1.79it/s][A

evaluating Epoch:   7%|[32m▋         [0m| 14/200 [00:08<01:44,
  1.77it/s][A

evaluating Epoch:   8%|[32m▊         [0m| 15/200 [00:08<01:45,
  1.76it/s][A

evaluating Epoch:   8%|[32m▊         [0m| 16/200 [00:09<01:44,
  1.77it/s][A

evaluating Epoch:   8%|[32m▊         [0m| 17/200 [00:09<01:42,
  1.78it/s][A

evaluating Epoch:   9%|[32m▉         [0m| 18/200 [00:10<01:41,
  1.79it/s][A
Training Epoch: 0/3,
 step 248/249 completed (loss: 0.024863319471478462): : 30876it [51:50,
 19.80it/s]

evaluating Epoch:  10%|[32m▉         [0m| 19/200 [00:10<01:42,
  1.76it/s][A

evaluating Epoch:  10%|[32m█         [0m| 20/200 [00:11<01:42,
  1.75it/s][A

evaluating Epoch:  10%|[32m█         [0m| 21/200 [00:11<01:42,
  1.75it/s][A

evaluating Epoch:  11%|[32m█         [0m| 22/200 [00:12<01:42,
  1.73it/s][A

evaluating Epoch:  12%|[32m█▏        [0m| 23/200 [00:13<01:42,
  1.73it/s][A

evaluating Epoch:  12%|[32m█▏        [0m| 24/200 [00:13<01:41,
  1.73it/s][A

evaluating Epoch:  12%|[32m█▎        [0m| 25/200 [00:14<01:40,
  1.74it/s][A

evaluating Epoch:  13%|[32m█▎        [0m| 26/200 [00:14<01:40,
  1.74it/s][A

evaluating Epoch:  14%|[32m█▎        [0m| 27/200 [00:15<01:38,
  1.76it/s][A

evaluating Epoch:  14%|[32m█▍        [0m| 28/200 [00:16<01:37,
  1.76it/s][A

evaluating Epoch:  14%|[32m█▍        [0m| 29/200 [00:16<01:37,
  1.75it/s][A

evaluating Epoch:  15%|[32m█▌        [0m| 30/200 [00:17<01:36,
  1.76it/s][A

evaluating Epoch:  16%|[32m█▌        [0m| 31/200 [00:17<01:37,
  1.74it/s][A

evaluating Epoch:  16%|[32m█▌        [0m| 32/200 [00:18<01:36,
  1.74it/s][A

evaluating Epoch:  16%|[32m█▋        [0m| 33/200 [00:18<01:35,
  1.75it/s][A

evaluating Epoch:  17%|[32m█▋        [0m| 34/200 [00:19<01:33,
  1.78it/s][A

evaluating Epoch:  18%|[32m█▊        [0m| 35/200 [00:20<01:33,
  1.76it/s][A

evaluating Epoch:  18%|[32m█▊        [0m| 36/200 [00:20<01:32,
  1.77it/s][A

evaluating Epoch:  18%|[32m█▊        [0m| 37/200 [00:21<01:32,
  1.77it/s][A

evaluating Epoch:  19%|[32m█▉        [0m| 38/200 [00:21<01:31,
  1.76it/s][A

evaluating Epoch:  20%|[32m█▉        [0m| 39/200 [00:22<01:31,
  1.76it/s][A

evaluating Epoch:  20%|[32m██        [0m| 40/200 [00:22<01:31,
  1.75it/s][A

evaluating Epoch:  20%|[32m██        [0m| 41/200 [00:23<01:30,
  1.75it/s][A

evaluating Epoch:  21%|[32m██        [0m| 42/200 [00:23<01:28,
  1.78it/s][A

evaluating Epoch:  22%|[32m██▏       [0m| 43/200 [00:24<01:28,
  1.77it/s][A

evaluating Epoch:  22%|[32m██▏       [0m| 44/200 [00:25<01:29,
  1.75it/s][A

evaluating Epoch:  22%|[32m██▎       [0m| 45/200 [00:25<01:28,
  1.76it/s][A

evaluating Epoch:  23%|[32m██▎       [0m| 46/200 [00:26<01:28,
  1.74it/s][A

evaluating Epoch:  24%|[32m██▎       [0m| 47/200 [00:26<01:26,
  1.76it/s][A

evaluating Epoch:  24%|[32m██▍       [0m| 48/200 [00:27<01:25,
  1.77it/s][A

evaluating Epoch:  24%|[32m██▍       [0m| 49/200 [00:27<01:25,
  1.76it/s][A

evaluating Epoch:  25%|[32m██▌       [0m| 50/200 [00:28<01:25,
  1.75it/s][A

evaluating Epoch:  26%|[32m██▌       [0m| 51/200 [00:29<01:24,
  1.77it/s][A

evaluating Epoch:  26%|[32m██▌       [0m| 52/200 [00:29<01:23,
  1.77it/s][A

evaluating Epoch:  26%|[32m██▋       [0m| 53/200 [00:30<01:21,
  1.80it/s][A

evaluating Epoch:  27%|[32m██▋       [0m| 54/200 [00:30<01:20,
  1.82it/s][A

evaluating Epoch:  28%|[32m██▊       [0m| 55/200 [00:31<01:20,
  1.81it/s][A

evaluating Epoch:  28%|[32m██▊       [0m| 56/200 [00:31<01:19,
  1.81it/s][A

evaluating Epoch:  28%|[32m██▊       [0m| 57/200 [00:32<01:18,
  1.82it/s][A

evaluating Epoch:  29%|[32m██▉       [0m| 58/200 [00:32<01:17,
  1.83it/s][A

evaluating Epoch:  30%|[32m██▉       [0m| 59/200 [00:33<01:17,
  1.81it/s][A

evaluating Epoch:  30%|[32m███       [0m| 60/200 [00:34<01:17,
  1.80it/s][A

evaluating Epoch:  30%|[32m███       [0m| 61/200 [00:34<01:17,
  1.79it/s][A

evaluating Epoch:  31%|[32m███       [0m| 62/200 [00:35<01:16,
  1.80it/s][A

evaluating Epoch:  32%|[32m███▏      [0m| 63/200 [00:35<01:16,
  1.80it/s][A

evaluating Epoch:  32%|[32m███▏      [0m| 64/200 [00:36<01:15,
  1.79it/s][A

evaluating Epoch:  32%|[32m███▎      [0m| 65/200 [00:36<01:15,
  1.78it/s][A

evaluating Epoch:  33%|[32m███▎      [0m| 66/200 [00:37<01:15,
  1.77it/s][A

evaluating Epoch:  34%|[32m███▎      [0m| 67/200 [00:37<01:15,
  1.77it/s][A

evaluating Epoch:  34%|[32m███▍      [0m| 68/200 [00:38<01:14,
  1.77it/s][A

evaluating Epoch:  34%|[32m███▍      [0m| 69/200 [00:39<01:13,
  1.79it/s][A

evaluating Epoch:  35%|[32m███▌      [0m| 70/200 [00:39<01:11,
  1.81it/s][A

evaluating Epoch:  36%|[32m███▌      [0m| 71/200 [00:40<01:11,
  1.80it/s][A

evaluating Epoch:  36%|[32m███▌      [0m| 72/200 [00:40<01:11,
  1.79it/s][A

evaluating Epoch:  36%|[32m███▋      [0m| 73/200 [00:41<01:10,
  1.79it/s][A

evaluating Epoch:  37%|[32m███▋      [0m| 74/200 [00:41<01:11,
  1.77it/s][A

evaluating Epoch:  38%|[32m███▊      [0m| 75/200 [00:42<01:11,
  1.75it/s][A

evaluating Epoch:  38%|[32m███▊      [0m| 76/200 [00:43<01:10,
  1.76it/s][A

evaluating Epoch:  38%|[32m███▊      [0m| 77/200 [00:43<01:08,
  1.79it/s][A

evaluating Epoch:  39%|[32m███▉      [0m| 78/200 [00:44<01:08,
  1.79it/s][A

evaluating Epoch:  40%|[32m███▉      [0m| 79/200 [00:44<01:07,
  1.79it/s][A

evaluating Epoch:  40%|[32m████      [0m| 80/200 [00:45<01:06,
  1.80it/s][A

evaluating Epoch:  40%|[32m████      [0m| 81/200 [00:45<01:06,
  1.79it/s][A

evaluating Epoch:  41%|[32m████      [0m| 82/200 [00:46<01:05,
  1.79it/s][A

evaluating Epoch:  42%|[32m████▏     [0m| 83/200 [00:46<01:05,
  1.78it/s][A

evaluating Epoch:  42%|[32m████▏     [0m| 84/200 [00:47<01:05,
  1.77it/s][A

evaluating Epoch:  42%|[32m████▎     [0m| 85/200 [00:48<01:05,
  1.76it/s][A

evaluating Epoch:  43%|[32m████▎     [0m| 86/200 [00:48<01:03,
  1.78it/s][A

evaluating Epoch:  44%|[32m████▎     [0m| 87/200 [00:49<01:03,
  1.79it/s][A

evaluating Epoch:  44%|[32m████▍     [0m| 88/200 [00:49<01:03,
  1.78it/s][A

evaluating Epoch:  44%|[32m████▍     [0m| 89/200 [00:50<01:01,
  1.79it/s][A

evaluating Epoch:  45%|[32m████▌     [0m| 90/200 [00:50<01:01,
  1.79it/s][A

evaluating Epoch:  46%|[32m████▌     [0m| 91/200 [00:51<01:00,
  1.79it/s][A

evaluating Epoch:  46%|[32m████▌     [0m| 92/200 [00:51<01:00,
  1.78it/s][A

evaluating Epoch:  46%|[32m████▋     [0m| 93/200 [00:52<01:00,
  1.78it/s][A

evaluating Epoch:  47%|[32m████▋     [0m| 94/200 [00:53<00:59,
  1.78it/s][A

evaluating Epoch:  48%|[32m████▊     [0m| 95/200 [00:53<00:58,
  1.81it/s][A

evaluating Epoch:  48%|[32m████▊     [0m| 96/200 [00:54<00:57,
  1.82it/s][A

evaluating Epoch:  48%|[32m████▊     [0m| 97/200 [00:54<00:56,
  1.82it/s][A

evaluating Epoch:  49%|[32m████▉     [0m| 98/200 [00:55<00:56,
  1.81it/s][A

evaluating Epoch:  50%|[32m████▉     [0m| 99/200 [00:55<00:55,
  1.80it/s][A

evaluating Epoch:  50%|[32m█████     [0m| 100/200 [00:56<00:55,
  1.81it/s][A

evaluating Epoch:  50%|[32m█████     [0m| 101/200 [00:56<00:54,
  1.82it/s][A

evaluating Epoch:  51%|[32m█████     [0m| 102/200 [00:57<00:54,
  1.81it/s][A

evaluating Epoch:  52%|[32m█████▏    [0m| 103/200 [00:58<00:53,
  1.80it/s][A

evaluating Epoch:  52%|[32m█████▏    [0m| 104/200 [00:58<00:53,
  1.79it/s][A

evaluating Epoch:  52%|[32m█████▎    [0m| 105/200 [00:59<00:53,
  1.79it/s][A

evaluating Epoch:  53%|[32m█████▎    [0m| 106/200 [00:59<00:53,
  1.77it/s][A

evaluating Epoch:  54%|[32m█████▎    [0m| 107/200 [01:00<00:51,
  1.79it/s][A

evaluating Epoch:  54%|[32m█████▍    [0m| 108/200 [01:00<00:51,
  1.79it/s][A

evaluating Epoch:  55%|[32m█████▍    [0m| 109/200 [01:01<00:51,
  1.78it/s][A

evaluating Epoch:  55%|[32m█████▌    [0m| 110/200 [01:02<00:50,
  1.79it/s][A

evaluating Epoch:  56%|[32m█████▌    [0m| 111/200 [01:02<00:50,
  1.77it/s][A

evaluating Epoch:  56%|[32m█████▌    [0m| 112/200 [01:03<00:49,
  1.77it/s][A

evaluating Epoch:  56%|[32m█████▋    [0m| 113/200 [01:03<00:48,
  1.79it/s][A

evaluating Epoch:  57%|[32m█████▋    [0m| 114/200 [01:04<00:47,
  1.81it/s][A

evaluating Epoch:  57%|[32m█████▊    [0m| 115/200 [01:04<00:47,
  1.80it/s][A

evaluating Epoch:  58%|[32m█████▊    [0m| 116/200 [01:05<00:46,
  1.79it/s][A

evaluating Epoch:  58%|[32m█████▊    [0m| 117/200 [01:05<00:46,
  1.78it/s][A

evaluating Epoch:  59%|[32m█████▉    [0m| 118/200 [01:06<00:45,
  1.79it/s][A

evaluating Epoch:  60%|[32m█████▉    [0m| 119/200 [01:07<00:44,
  1.81it/s][A

evaluating Epoch:  60%|[32m██████    [0m| 120/200 [01:07<00:44,
  1.80it/s][A

evaluating Epoch:  60%|[32m██████    [0m| 121/200 [01:08<00:44,
  1.78it/s][A

evaluating Epoch:  61%|[32m██████    [0m| 122/200 [01:08<00:43,
  1.78it/s][A

evaluating Epoch:  62%|[32m██████▏   [0m| 123/200 [01:09<00:42,
  1.80it/s][A

evaluating Epoch:  62%|[32m██████▏   [0m| 124/200 [01:09<00:42,
  1.79it/s][A

evaluating Epoch:  62%|[32m██████▎   [0m| 125/200 [01:10<00:41,
  1.79it/s][A

evaluating Epoch:  63%|[32m██████▎   [0m| 126/200 [01:10<00:41,
  1.76it/s][A

evaluating Epoch:  64%|[32m██████▎   [0m| 127/200 [01:11<00:41,
  1.74it/s][A

evaluating Epoch:  64%|[32m██████▍   [0m| 128/200 [01:12<00:41,
  1.75it/s][A

evaluating Epoch:  64%|[32m██████▍   [0m| 129/200 [01:12<00:39,
  1.78it/s][A

evaluating Epoch:  65%|[32m██████▌   [0m| 130/200 [01:13<00:39,
  1.76it/s][A

evaluating Epoch:  66%|[32m██████▌   [0m| 131/200 [01:13<00:38,
  1.78it/s][A

evaluating Epoch:  66%|[32m██████▌   [0m| 132/200 [01:14<00:38,
  1.77it/s][A

evaluating Epoch:  66%|[32m██████▋   [0m| 133/200 [01:14<00:37,
  1.78it/s][A

evaluating Epoch:  67%|[32m██████▋   [0m| 134/200 [01:15<00:37,
  1.78it/s][A

evaluating Epoch:  68%|[32m██████▊   [0m| 135/200 [01:16<00:36,
  1.80it/s][A

evaluating Epoch:  68%|[32m██████▊   [0m| 136/200 [01:16<00:35,
  1.80it/s][A

evaluating Epoch:  68%|[32m██████▊   [0m| 137/200 [01:17<00:35,
  1.79it/s][A

evaluating Epoch:  69%|[32m██████▉   [0m| 138/200 [01:17<00:34,
  1.77it/s][A

evaluating Epoch:  70%|[32m██████▉   [0m| 139/200 [01:18<00:34,
  1.76it/s][A

evaluating Epoch:  70%|[32m███████   [0m| 140/200 [01:18<00:33,
  1.79it/s][A

evaluating Epoch:  70%|[32m███████   [0m| 141/200 [01:19<00:33,
  1.79it/s][A

evaluating Epoch:  71%|[32m███████   [0m| 142/200 [01:19<00:32,
  1.79it/s][A

evaluating Epoch:  72%|[32m███████▏  [0m| 143/200 [01:20<00:31,
  1.79it/s][A

evaluating Epoch:  72%|[32m███████▏  [0m| 144/200 [01:21<00:31,
  1.79it/s][A

evaluating Epoch:  72%|[32m███████▎  [0m| 145/200 [01:21<00:30,
  1.80it/s][A

evaluating Epoch:  73%|[32m███████▎  [0m| 146/200 [01:22<00:30,
  1.79it/s][A

evaluating Epoch:  74%|[32m███████▎  [0m| 147/200 [01:22<00:29,
  1.80it/s][A

evaluating Epoch:  74%|[32m███████▍  [0m| 148/200 [01:23<00:28,
  1.80it/s][A

evaluating Epoch:  74%|[32m███████▍  [0m| 149/200 [01:23<00:28,
  1.81it/s][A

evaluating Epoch:  75%|[32m███████▌  [0m| 150/200 [01:24<00:27,
  1.79it/s][A

evaluating Epoch:  76%|[32m███████▌  [0m| 151/200 [01:24<00:27,
  1.78it/s][A

evaluating Epoch:  76%|[32m███████▌  [0m| 152/200 [01:25<00:26,
  1.81it/s][A

evaluating Epoch:  76%|[32m███████▋  [0m| 153/200 [01:26<00:26,
  1.79it/s][A

evaluating Epoch:  77%|[32m███████▋  [0m| 154/200 [01:26<00:25,
  1.79it/s][A

evaluating Epoch:  78%|[32m███████▊  [0m| 155/200 [01:27<00:25,
  1.79it/s][A

evaluating Epoch:  78%|[32m███████▊  [0m| 156/200 [01:27<00:24,
  1.78it/s][A

evaluating Epoch:  78%|[32m███████▊  [0m| 157/200 [01:28<00:24,
  1.77it/s][A

evaluating Epoch:  79%|[32m███████▉  [0m| 158/200 [01:28<00:24,
  1.75it/s][A

evaluating Epoch:  80%|[32m███████▉  [0m| 159/200 [01:29<00:23,
  1.76it/s][A

evaluating Epoch:  80%|[32m████████  [0m| 160/200 [01:30<00:22,
  1.75it/s][A

evaluating Epoch:  80%|[32m████████  [0m| 161/200 [01:30<00:22,
  1.74it/s][A

evaluating Epoch:  81%|[32m████████  [0m| 162/200 [01:31<00:21,
  1.76it/s][A

evaluating Epoch:  82%|[32m████████▏ [0m| 163/200 [01:31<00:20,
  1.78it/s][A

evaluating Epoch:  82%|[32m████████▏ [0m| 164/200 [01:32<00:20,
  1.77it/s][A

evaluating Epoch:  82%|[32m████████▎ [0m| 165/200 [01:32<00:19,
  1.78it/s][A

evaluating Epoch:  83%|[32m████████▎ [0m| 166/200 [01:33<00:19,
  1.78it/s][A

evaluating Epoch:  84%|[32m████████▎ [0m| 167/200 [01:34<00:18,
  1.77it/s][A

evaluating Epoch:  84%|[32m████████▍ [0m| 168/200 [01:34<00:18,
  1.77it/s][A

evaluating Epoch:  84%|[32m████████▍ [0m| 169/200 [01:35<00:17,
  1.76it/s][A

evaluating Epoch:  85%|[32m████████▌ [0m| 170/200 [01:35<00:16,
  1.78it/s][A

evaluating Epoch:  86%|[32m████████▌ [0m| 171/200 [01:36<00:16,
  1.78it/s][A

evaluating Epoch:  86%|[32m████████▌ [0m| 172/200 [01:36<00:15,
  1.78it/s][A

evaluating Epoch:  86%|[32m████████▋ [0m| 173/200 [01:37<00:15,
  1.79it/s][A

evaluating Epoch:  87%|[32m████████▋ [0m| 174/200 [01:37<00:14,
  1.77it/s][A

evaluating Epoch:  88%|[32m████████▊ [0m| 175/200 [01:38<00:14,
  1.78it/s][A

evaluating Epoch:  88%|[32m████████▊ [0m| 176/200 [01:39<00:13,
  1.78it/s][A

evaluating Epoch:  88%|[32m████████▊ [0m| 177/200 [01:39<00:12,
  1.77it/s][A

evaluating Epoch:  89%|[32m████████▉ [0m| 178/200 [01:40<00:12,
  1.78it/s][A

evaluating Epoch:  90%|[32m████████▉ [0m| 179/200 [01:40<00:11,
  1.77it/s][A

evaluating Epoch:  90%|[32m█████████ [0m| 180/200 [01:41<00:11,
  1.78it/s][A

evaluating Epoch:  90%|[32m█████████ [0m| 181/200 [01:41<00:10,
  1.78it/s][A

evaluating Epoch:  91%|[32m█████████ [0m| 182/200 [01:42<00:10,
  1.77it/s][A

evaluating Epoch:  92%|[32m█████████▏[0m| 183/200 [01:43<00:09,
  1.78it/s][A

evaluating Epoch:  92%|[32m█████████▏[0m| 184/200 [01:43<00:09,
  1.77it/s][A

evaluating Epoch:  92%|[32m█████████▎[0m| 185/200 [01:44<00:08,
  1.78it/s][A

evaluating Epoch:  93%|[32m█████████▎[0m| 186/200 [01:44<00:07,
  1.76it/s][A

evaluating Epoch:  94%|[32m█████████▎[0m| 187/200 [01:45<00:07,
  1.76it/s][A

evaluating Epoch:  94%|[32m█████████▍[0m| 188/200 [01:45<00:06,
  1.78it/s][A

evaluating Epoch:  94%|[32m█████████▍[0m| 189/200 [01:46<00:06,
  1.80it/s][A

evaluating Epoch:  95%|[32m█████████▌[0m| 190/200 [01:46<00:05,
  1.80it/s][A

evaluating Epoch:  96%|[32m█████████▌[0m| 191/200 [01:47<00:04,
  1.80it/s][A

evaluating Epoch:  96%|[32m█████████▌[0m| 192/200 [01:48<00:04,
  1.77it/s][A

evaluating Epoch:  96%|[32m█████████▋[0m| 193/200 [01:48<00:03,
  1.76it/s][A

evaluating Epoch:  97%|[32m█████████▋[0m| 194/200 [01:49<00:03,
  1.76it/s][A

evaluating Epoch:  98%|[32m█████████▊[0m| 195/200 [01:49<00:02,
  1.78it/s][A

evaluating Epoch:  98%|[32m█████████▊[0m| 196/200 [01:50<00:02,
  1.79it/s][A

evaluating Epoch:  98%|[32m█████████▊[0m| 197/200 [01:50<00:01,
  1.77it/s][A

evaluating Epoch:  99%|[32m█████████▉[0m| 198/200 [01:51<00:01,
  1.75it/s][A

evaluating Epoch: 100%|[32m█████████▉[0m| 199/200 [01:52<00:00,
  1.74it/s][A

evaluating Epoch: 100%|[32m██████████[0m| 200/200 [01:52<00:00,
  1.74it/s][A
evaluating Epoch: 100%|[32m██████████[0m| 200/200 [01:52<00:00,
  1.77it/s]
 eval_ppl=tensor(1.1605,
 device='cuda:0') eval_epoch_loss=tensor(0.1489,
 device='cuda:0')
we are about to save the PEFT modules
PEFT modules are saved in ./output_dir_whole_dataset directory
best eval loss on epoch 0 is 0.14886374771595
Epoch 1: train_perplexity=1.2701,
 train_epoch_loss=0.2391,
 epcoh time 3099.772042863071s


Training Epoch: 1:   0%|[34m          [0m| 0/249 [00:00<?,
 ?it/s][A
Training Epoch: 0/3,
 step 248/249 completed (loss: 0.024863319471478462): : 30876it [53:32,
  9.61it/s]


Training Epoch: 1:   0%|[34m          [0m| 0/249 [00:12<?,
 ?it/s][A

Training Epoch: 1/3,
 step 0/249 completed (loss: 0.11722183227539062):   0%|[34m          [0m| 0/249 [00:12<?,
 ?it/s][A

Training Epoch: 1/3,
 step 0/249 completed (loss: 0.11722183227539062):   0%|[34m          [0m| 1/249 [00:24<51:16,
 12.40s/it][A

Training Epoch: 1/3,
 step 1/249 completed (loss: 0.11164526641368866):   0%|[34m          [0m| 1/249 [00:24<51:16,
 12.40s/it][A

Training Epoch: 1/3,
 step 1/249 completed (loss: 0.11164526641368866):   1%|[34m          [0m| 3/249 [00:37<32:03,
  7.82s/it][A

Training Epoch: 1/3,
 step 2/249 completed (loss: 0.0703200250864029):   1%|[34m          [0m| 3/249 [00:37<32:03,
  7.82s/it] [A

Training Epoch: 1/3,
 step 2/249 completed (loss: 0.0703200250864029):   2%|[34m▏         [0m| 6/249 [00:49<22:31,
  5.56s/it][A

Training Epoch: 1/3,
 step 3/249 completed (loss: 0.13514451682567596):   2%|[34m▏         [0m| 6/249 [00:49<22:31,
  5.56s/it][A

Training Epoch: 1/3,
 step 3/249 completed (loss: 0.13514451682567596):   4%|[34m▍         [0m| 10/249 [01:02<16:52,
  4.24s/it][A

Training Epoch: 1/3,
 step 4/249 completed (loss: 0.15145471692085266):   4%|[34m▍         [0m| 10/249 [01:02<16:52,
  4.24s/it][A

Training Epoch: 1/3,
 step 4/249 completed (loss: 0.15145471692085266):   6%|[34m▌         [0m| 15/249 [01:14<13:10,
  3.38s/it][A

Training Epoch: 1/3,
 step 5/249 completed (loss: 0.13923433423042297):   6%|[34m▌         [0m| 15/249 [01:14<13:10,
  3.38s/it][A

Training Epoch: 1/3,
 step 5/249 completed (loss: 0.13923433423042297):   8%|[34m▊         [0m| 21/249 [01:26<10:33,
  2.78s/it][A

Training Epoch: 1/3,
 step 6/249 completed (loss: 0.2199632227420807):   8%|[34m▊         [0m| 21/249 [01:27<10:33,
  2.78s/it] [A

Training Epoch: 1/3,
 step 6/249 completed (loss: 0.2199632227420807):  11%|[34m█         [0m| 28/249 [01:39<08:38,
  2.34s/it][A

Training Epoch: 1/3,
 step 7/249 completed (loss: 0.03254566341638565):  11%|[34m█         [0m| 28/249 [01:39<08:38,
  2.34s/it][A

Training Epoch: 1/3,
 step 7/249 completed (loss: 0.03254566341638565):  14%|[34m█▍        [0m| 36/249 [01:51<07:09,
  2.02s/it][A

Training Epoch: 1/3,
 step 8/249 completed (loss: 0.1016196608543396):  14%|[34m█▍        [0m| 36/249 [01:51<07:09,
  2.02s/it] [A

Training Epoch: 1/3,
 step 8/249 completed (loss: 0.1016196608543396):  18%|[34m█▊        [0m| 45/249 [02:04<05:59,
  1.76s/it][A

Training Epoch: 1/3,
 step 9/249 completed (loss: 0.1418910175561905):  18%|[34m█▊        [0m| 45/249 [02:04<05:59,
  1.76s/it][A

Training Epoch: 1/3,
 step 9/249 completed (loss: 0.1418910175561905):  22%|[34m██▏       [0m| 55/249 [02:16<05:02,
  1.56s/it][A

Training Epoch: 1/3,
 step 10/249 completed (loss: 0.07646206766366959):  22%|[34m██▏       [0m| 55/249 [02:16<05:02,
  1.56s/it][A

Training Epoch: 1/3,
 step 10/249 completed (loss: 0.07646206766366959):  27%|[34m██▋       [0m| 66/249 [02:28<04:15,
  1.40s/it][A

Training Epoch: 1/3,
 step 11/249 completed (loss: 0.08000905066728592):  27%|[34m██▋       [0m| 66/249 [02:29<04:15,
  1.40s/it][A

Training Epoch: 1/3,
 step 11/249 completed (loss: 0.08000905066728592):  31%|[34m███▏      [0m| 78/249 [02:41<03:35,
  1.26s/it][A

Training Epoch: 1/3,
 step 12/249 completed (loss: 0.13927972316741943):  31%|[34m███▏      [0m| 78/249 [02:41<03:35,
  1.26s/it][A

Training Epoch: 1/3,
 step 12/249 completed (loss: 0.13927972316741943):  37%|[34m███▋      [0m| 91/249 [02:53<03:01,
  1.15s/it][A

Training Epoch: 1/3,
 step 13/249 completed (loss: 0.0954667255282402):  37%|[34m███▋      [0m| 91/249 [02:54<03:01,
  1.15s/it] [A

Training Epoch: 1/3,
 step 13/249 completed (loss: 0.0954667255282402):  42%|[34m████▏     [0m| 105/249 [03:06<02:32,
  1.06s/it][A

Training Epoch: 1/3,
 step 14/249 completed (loss: 0.1147107258439064):  42%|[34m████▏     [0m| 105/249 [03:06<02:32,
  1.06s/it][A

Training Epoch: 1/3,
 step 14/249 completed (loss: 0.1147107258439064):  48%|[34m████▊     [0m| 120/249 [03:18<02:05,
  1.02it/s][A

Training Epoch: 1/3,
 step 15/249 completed (loss: 0.08418373018503189):  48%|[34m████▊     [0m| 120/249 [03:18<02:05,
  1.02it/s][A

Training Epoch: 1/3,
 step 15/249 completed (loss: 0.08418373018503189):  55%|[34m█████▍    [0m| 136/249 [03:31<01:42,
  1.10it/s][A

Training Epoch: 1/3,
 step 16/249 completed (loss: 0.11411431431770325):  55%|[34m█████▍    [0m| 136/249 [03:31<01:42,
  1.10it/s][A

Training Epoch: 1/3,
 step 16/249 completed (loss: 0.11411431431770325):  61%|[34m██████▏   [0m| 153/249 [03:43<01:21,
  1.18it/s][A

Training Epoch: 1/3,
 step 17/249 completed (loss: 0.1041579321026802):  61%|[34m██████▏   [0m| 153/249 [03:43<01:21,
  1.18it/s] [A

Training Epoch: 1/3,
 step 17/249 completed (loss: 0.1041579321026802):  69%|[34m██████▊   [0m| 171/249 [03:55<01:01,
  1.26it/s][A

Training Epoch: 1/3,
 step 18/249 completed (loss: 0.045857515186071396):  69%|[34m██████▊   [0m| 171/249 [03:56<01:01,
  1.26it/s][A

Training Epoch: 1/3,
 step 18/249 completed (loss: 0.045857515186071396):  76%|[34m███████▋  [0m| 190/249 [04:08<00:43,
  1.34it/s][A

Training Epoch: 1/3,
 step 19/249 completed (loss: 0.07406510412693024):  76%|[34m███████▋  [0m| 190/249 [04:08<00:43,
  1.34it/s] [A

Training Epoch: 1/3,
 step 19/249 completed (loss: 0.07406510412693024):  84%|[34m████████▍ [0m| 210/249 [04:20<00:27,
  1.42it/s][A

Training Epoch: 1/3,
 step 20/249 completed (loss: 0.08690892159938812):  84%|[34m████████▍ [0m| 210/249 [04:20<00:27,
  1.42it/s][A

Training Epoch: 1/3,
 step 20/249 completed (loss: 0.08690892159938812):  93%|[34m█████████▎[0m| 231/249 [04:33<00:11,
  1.50it/s][A

Training Epoch: 1/3,
 step 21/249 completed (loss: 0.003739528590813279):  93%|[34m█████████▎[0m| 231/249 [04:33<00:11,
  1.50it/s][A

Training Epoch: 1/3,
 step 21/249 completed (loss: 0.003739528590813279): : 253it [04:45,
  1.58it/s]                       [A

Training Epoch: 1/3,
 step 22/249 completed (loss: 0.07141102850437164): : 253it [04:45,
  1.58it/s] [A

Training Epoch: 1/3,
 step 22/249 completed (loss: 0.07141102850437164): : 276it [04:58,
  1.67it/s][A

Training Epoch: 1/3,
 step 23/249 completed (loss: 0.06102024391293526): : 276it [04:58,
  1.67it/s][A

Training Epoch: 1/3,
 step 23/249 completed (loss: 0.06102024391293526): : 300it [05:10,
  1.74it/s][A

Training Epoch: 1/3,
 step 24/249 completed (loss: 0.08587609231472015): : 300it [05:10,
  1.74it/s][A

Training Epoch: 1/3,
 step 24/249 completed (loss: 0.08587609231472015): : 325it [05:22,
  1.83it/s][A

Training Epoch: 1/3,
 step 25/249 completed (loss: 0.04369720071554184): : 325it [05:23,
  1.83it/s][A

Training Epoch: 1/3,
 step 25/249 completed (loss: 0.04369720071554184): : 351it [05:35,
  1.91it/s][A

Training Epoch: 1/3,
 step 26/249 completed (loss: 0.03976128250360489): : 351it [05:35,
  1.91it/s][A

Training Epoch: 1/3,
 step 26/249 completed (loss: 0.03976128250360489): : 378it [05:47,
  1.99it/s][A

Training Epoch: 1/3,
 step 27/249 completed (loss: 0.08820948749780655): : 378it [05:47,
  1.99it/s][A

Training Epoch: 1/3,
 step 27/249 completed (loss: 0.08820948749780655): : 406it [06:00,
  2.07it/s][A

Training Epoch: 1/3,
 step 28/249 completed (loss: 0.05729304254055023): : 406it [06:00,
  2.07it/s][A

Training Epoch: 1/3,
 step 28/249 completed (loss: 0.05729304254055023): : 435it [06:12,
  2.15it/s][A

Training Epoch: 1/3,
 step 29/249 completed (loss: 0.10658400505781174): : 435it [06:12,
  2.15it/s][A

Training Epoch: 1/3,
 step 29/249 completed (loss: 0.10658400505781174): : 465it [06:24,
  2.23it/s][A

Training Epoch: 1/3,
 step 30/249 completed (loss: 0.0969386100769043): : 465it [06:25,
  2.23it/s] [A

Training Epoch: 1/3,
 step 30/249 completed (loss: 0.0969386100769043): : 496it [06:37,
  2.31it/s][A

Training Epoch: 1/3,
 step 31/249 completed (loss: 0.10687676072120667): : 496it [06:37,
  2.31it/s][A

Training Epoch: 1/3,
 step 31/249 completed (loss: 0.10687676072120667): : 528it [06:49,
  2.39it/s][A

Training Epoch: 1/3,
 step 32/249 completed (loss: 0.021663764491677284): : 528it [06:50,
  2.39it/s][A

Training Epoch: 1/3,
 step 32/249 completed (loss: 0.021663764491677284): : 561it [07:02,
  2.47it/s][A

Training Epoch: 1/3,
 step 33/249 completed (loss: 0.14244794845581055): : 561it [07:02,
  2.47it/s] [A

Training Epoch: 1/3,
 step 33/249 completed (loss: 0.14244794845581055): : 595it [07:14,
  2.55it/s][A

Training Epoch: 1/3,
 step 34/249 completed (loss: 0.03806120157241821): : 595it [07:14,
  2.55it/s][A

Training Epoch: 1/3,
 step 34/249 completed (loss: 0.03806120157241821): : 630it [07:27,
  2.63it/s][A

Training Epoch: 1/3,
 step 35/249 completed (loss: 0.06007655709981918): : 630it [07:27,
  2.63it/s][A

Training Epoch: 1/3,
 step 35/249 completed (loss: 0.06007655709981918): : 666it [07:39,
  2.71it/s][A

Training Epoch: 1/3,
 step 36/249 completed (loss: 0.04919823259115219): : 666it [07:39,
  2.71it/s][A

Training Epoch: 1/3,
 step 36/249 completed (loss: 0.04919823259115219): : 703it [07:51,
  2.79it/s][A

Training Epoch: 1/3,
 step 37/249 completed (loss: 0.027918675914406776): : 703it [07:52,
  2.79it/s][A

Training Epoch: 1/3,
 step 37/249 completed (loss: 0.027918675914406776): : 741it [08:04,
  2.87it/s][A

Training Epoch: 1/3,
 step 38/249 completed (loss: 0.07446073740720749): : 741it [08:04,
  2.87it/s] [A

Training Epoch: 1/3,
 step 38/249 completed (loss: 0.07446073740720749): : 780it [08:16,
  2.95it/s][A

Training Epoch: 1/3,
 step 39/249 completed (loss: 0.06368657201528549): : 780it [08:16,
  2.95it/s][A

Training Epoch: 1/3,
 step 39/249 completed (loss: 0.06368657201528549): : 820it [08:29,
  3.04it/s][A

Training Epoch: 1/3,
 step 40/249 completed (loss: 0.03195079416036606): : 820it [08:29,
  3.04it/s][A

Training Epoch: 1/3,
 step 40/249 completed (loss: 0.03195079416036606): : 861it [08:41,
  3.11it/s][A

Training Epoch: 1/3,
 step 41/249 completed (loss: 0.14104413986206055): : 861it [08:41,
  3.11it/s][A

Training Epoch: 1/3,
 step 41/249 completed (loss: 0.14104413986206055): : 903it [08:54,
  3.19it/s][A

Training Epoch: 1/3,
 step 42/249 completed (loss: 0.03368869051337242): : 903it [08:54,
  3.19it/s][A

Training Epoch: 1/3,
 step 42/249 completed (loss: 0.03368869051337242): : 946it [09:06,
  3.27it/s][A

Training Epoch: 1/3,
 step 43/249 completed (loss: 0.14587226510047913): : 946it [09:06,
  3.27it/s][A

Training Epoch: 1/3,
 step 43/249 completed (loss: 0.14587226510047913): : 990it [09:18,
  3.35it/s][A

Training Epoch: 1/3,
 step 44/249 completed (loss: 0.03978903964161873): : 990it [09:19,
  3.35it/s][A

Training Epoch: 1/3,
 step 44/249 completed (loss: 0.03978903964161873): : 1035it [09:31,
  3.44it/s][A

Training Epoch: 1/3,
 step 45/249 completed (loss: 0.026382185518741608): : 1035it [09:31,
  3.44it/s][A

Training Epoch: 1/3,
 step 45/249 completed (loss: 0.026382185518741608): : 1081it [09:43,
  3.52it/s][A

Training Epoch: 1/3,
 step 46/249 completed (loss: 0.055021174252033234): : 1081it [09:43,
  3.52it/s][A

Training Epoch: 1/3,
 step 46/249 completed (loss: 0.055021174252033234): : 1128it [09:56,
  3.60it/s][A

Training Epoch: 1/3,
 step 47/249 completed (loss: 0.11893310397863388): : 1128it [09:56,
  3.60it/s] [A

Training Epoch: 1/3,
 step 47/249 completed (loss: 0.11893310397863388): : 1176it [10:08,
  3.68it/s][A

Training Epoch: 1/3,
 step 48/249 completed (loss: 0.22573316097259521): : 1176it [10:08,
  3.68it/s][A

Training Epoch: 1/3,
 step 48/249 completed (loss: 0.22573316097259521): : 1225it [10:20,
  3.76it/s][A

Training Epoch: 1/3,
 step 49/249 completed (loss: 0.27071860432624817): : 1225it [10:21,
  3.76it/s][A

Training Epoch: 1/3,
 step 49/249 completed (loss: 0.27071860432624817): : 1275it [10:33,
  3.84it/s][A

Training Epoch: 1/3,
 step 50/249 completed (loss: 0.028558315709233284): : 1275it [10:33,
  3.84it/s][A

Training Epoch: 1/3,
 step 50/249 completed (loss: 0.028558315709233284): : 1326it [10:45,
  3.92it/s][A

Training Epoch: 1/3,
 step 51/249 completed (loss: 0.07710679620504379): : 1326it [10:45,
  3.92it/s] [A

Training Epoch: 1/3,
 step 51/249 completed (loss: 0.07710679620504379): : 1378it [10:58,
  4.00it/s][A

Training Epoch: 1/3,
 step 52/249 completed (loss: 0.09019651263952255): : 1378it [10:58,
  4.00it/s][A

Training Epoch: 1/3,
 step 52/249 completed (loss: 0.09019651263952255): : 1431it [11:10,
  4.08it/s][A

Training Epoch: 1/3,
 step 53/249 completed (loss: 0.049174606800079346): : 1431it [11:10,
  4.08it/s][A

Training Epoch: 1/3,
 step 53/249 completed (loss: 0.049174606800079346): : 1485it [11:23,
  4.16it/s][A

Training Epoch: 1/3,
 step 54/249 completed (loss: 0.1266806572675705): : 1485it [11:23,
  4.16it/s]  [A

Training Epoch: 1/3,
 step 54/249 completed (loss: 0.1266806572675705): : 1540it [11:35,
  4.24it/s][A

Training Epoch: 1/3,
 step 55/249 completed (loss: 0.16204717755317688): : 1540it [11:35,
  4.24it/s][A

Training Epoch: 1/3,
 step 55/249 completed (loss: 0.16204717755317688): : 1596it [11:47,
  4.32it/s][A

Training Epoch: 1/3,
 step 56/249 completed (loss: 0.03779930993914604): : 1596it [11:48,
  4.32it/s][A

Training Epoch: 1/3,
 step 56/249 completed (loss: 0.03779930993914604): : 1653it [12:00,
  4.40it/s][A

Training Epoch: 1/3,
 step 57/249 completed (loss: 0.040838565677404404): : 1653it [12:00,
  4.40it/s][A

Training Epoch: 1/3,
 step 57/249 completed (loss: 0.040838565677404404): : 1711it [12:12,
  4.48it/s][A

Training Epoch: 1/3,
 step 58/249 completed (loss: 0.04727852717041969): : 1711it [12:12,
  4.48it/s] [A

Training Epoch: 1/3,
 step 58/249 completed (loss: 0.04727852717041969): : 1770it [12:25,
  4.56it/s][A

Training Epoch: 1/3,
 step 59/249 completed (loss: 0.05157580226659775): : 1770it [12:25,
  4.56it/s][A

Training Epoch: 1/3,
 step 59/249 completed (loss: 0.05157580226659775): : 1830it [12:37,
  4.65it/s][A

Training Epoch: 1/3,
 step 60/249 completed (loss: 0.017880095168948174): : 1830it [12:37,
  4.65it/s][A

Training Epoch: 1/3,
 step 60/249 completed (loss: 0.017880095168948174): : 1891it [12:49,
  4.72it/s][A

Training Epoch: 1/3,
 step 61/249 completed (loss: 0.15569233894348145): : 1891it [12:50,
  4.72it/s] [A

Training Epoch: 1/3,
 step 61/249 completed (loss: 0.15569233894348145): : 1953it [13:02,
  4.80it/s][A

Training Epoch: 1/3,
 step 62/249 completed (loss: 0.07919947803020477): : 1953it [13:02,
  4.80it/s][A

Training Epoch: 1/3,
 step 62/249 completed (loss: 0.07919947803020477): : 2016it [13:14,
  4.88it/s][A

Training Epoch: 1/3,
 step 63/249 completed (loss: 0.15272924304008484): : 2016it [13:15,
  4.88it/s][A

Training Epoch: 1/3,
 step 63/249 completed (loss: 0.15272924304008484): : 2080it [13:27,
  4.96it/s][A

Training Epoch: 1/3,
 step 64/249 completed (loss: 0.045681118965148926): : 2080it [13:27,
  4.96it/s][A

Training Epoch: 1/3,
 step 64/249 completed (loss: 0.045681118965148926): : 2145it [13:39,
  5.05it/s][A

Training Epoch: 1/3,
 step 65/249 completed (loss: 0.06644025444984436): : 2145it [13:39,
  5.05it/s] [A

Training Epoch: 1/3,
 step 65/249 completed (loss: 0.06644025444984436): : 2211it [13:52,
  5.13it/s][A

Training Epoch: 1/3,
 step 66/249 completed (loss: 0.056533634662628174): : 2211it [13:52,
  5.13it/s][A

Training Epoch: 1/3,
 step 66/249 completed (loss: 0.056533634662628174): : 2278it [14:04,
  5.21it/s][A

Training Epoch: 1/3,
 step 67/249 completed (loss: 0.03329352289438248): : 2278it [14:04,
  5.21it/s] [A

Training Epoch: 1/3,
 step 67/249 completed (loss: 0.03329352289438248): : 2346it [14:16,
  5.29it/s][A

Training Epoch: 1/3,
 step 68/249 completed (loss: 0.08932485431432724): : 2346it [14:17,
  5.29it/s][A

Training Epoch: 1/3,
 step 68/249 completed (loss: 0.08932485431432724): : 2415it [14:29,
  5.37it/s][A

Training Epoch: 1/3,
 step 69/249 completed (loss: 0.10851456224918365): : 2415it [14:29,
  5.37it/s][A

Training Epoch: 1/3,
 step 69/249 completed (loss: 0.10851456224918365): : 2485it [14:41,
  5.45it/s][A

Training Epoch: 1/3,
 step 70/249 completed (loss: 0.11952622979879379): : 2485it [14:41,
  5.45it/s][A

Training Epoch: 1/3,
 step 70/249 completed (loss: 0.11952622979879379): : 2556it [14:54,
  5.53it/s][A

Training Epoch: 1/3,
 step 71/249 completed (loss: 0.050823092460632324): : 2556it [14:54,
  5.53it/s][A

Training Epoch: 1/3,
 step 71/249 completed (loss: 0.050823092460632324): : 2628it [15:06,
  5.61it/s][A

Training Epoch: 1/3,
 step 72/249 completed (loss: 0.10597632825374603): : 2628it [15:06,
  5.61it/s] [A

Training Epoch: 1/3,
 step 72/249 completed (loss: 0.10597632825374603): : 2701it [15:19,
  5.69it/s][A

Training Epoch: 1/3,
 step 73/249 completed (loss: 0.06227642670273781): : 2701it [15:19,
  5.69it/s][A

Training Epoch: 1/3,
 step 73/249 completed (loss: 0.06227642670273781): : 2775it [15:31,
  5.77it/s][A

Training Epoch: 1/3,
 step 74/249 completed (loss: 0.028498496860265732): : 2775it [15:31,
  5.77it/s][A

Training Epoch: 1/3,
 step 74/249 completed (loss: 0.028498496860265732): : 2850it [15:43,
  5.85it/s][A

Training Epoch: 1/3,
 step 75/249 completed (loss: 0.018999865278601646): : 2850it [15:44,
  5.85it/s][A

Training Epoch: 1/3,
 step 75/249 completed (loss: 0.018999865278601646): : 2926it [15:56,
  5.93it/s][A

Training Epoch: 1/3,
 step 76/249 completed (loss: 0.05267995595932007): : 2926it [15:56,
  5.93it/s] [A

Training Epoch: 1/3,
 step 76/249 completed (loss: 0.05267995595932007): : 3003it [16:08,
  6.01it/s][A

Training Epoch: 1/3,
 step 77/249 completed (loss: 0.0652293711900711): : 3003it [16:08,
  6.01it/s] [A

Training Epoch: 1/3,
 step 77/249 completed (loss: 0.0652293711900711): : 3081it [16:21,
  6.09it/s][A

Training Epoch: 1/3,
 step 78/249 completed (loss: 0.0833704024553299): : 3081it [16:21,
  6.09it/s][A

Training Epoch: 1/3,
 step 78/249 completed (loss: 0.0833704024553299): : 3160it [16:33,
  6.17it/s][A

Training Epoch: 1/3,
 step 79/249 completed (loss: 0.1790439784526825): : 3160it [16:33,
  6.17it/s][A

Training Epoch: 1/3,
 step 79/249 completed (loss: 0.1790439784526825): : 3240it [16:46,
  6.25it/s][A

Training Epoch: 1/3,
 step 80/249 completed (loss: 0.09828756749629974): : 3240it [16:46,
  6.25it/s][A

Training Epoch: 1/3,
 step 80/249 completed (loss: 0.09828756749629974): : 3321it [16:58,
  6.33it/s][A

Training Epoch: 1/3,
 step 81/249 completed (loss: 0.06473453342914581): : 3321it [16:58,
  6.33it/s][A

Training Epoch: 1/3,
 step 81/249 completed (loss: 0.06473453342914581): : 3403it [17:10,
  6.41it/s][A

Training Epoch: 1/3,
 step 82/249 completed (loss: 0.058096230030059814): : 3403it [17:11,
  6.41it/s][A

Training Epoch: 1/3,
 step 82/249 completed (loss: 0.058096230030059814): : 3486it [17:23,
  6.49it/s][A

Training Epoch: 1/3,
 step 83/249 completed (loss: 0.04352068528532982): : 3486it [17:23,
  6.49it/s] [A

Training Epoch: 1/3,
 step 83/249 completed (loss: 0.04352068528532982): : 3570it [17:35,
  6.57it/s][A

Training Epoch: 1/3,
 step 84/249 completed (loss: 0.06435416638851166): : 3570it [17:35,
  6.57it/s][A

Training Epoch: 1/3,
 step 84/249 completed (loss: 0.06435416638851166): : 3655it [17:48,
  6.65it/s][A

Training Epoch: 1/3,
 step 85/249 completed (loss: 0.102060467004776): : 3655it [17:48,
  6.65it/s]  [A

Training Epoch: 1/3,
 step 85/249 completed (loss: 0.102060467004776): : 3741it [18:00,
  6.73it/s][A

Training Epoch: 1/3,
 step 86/249 completed (loss: 0.04019976034760475): : 3741it [18:00,
  6.73it/s][A

Training Epoch: 1/3,
 step 86/249 completed (loss: 0.04019976034760475): : 3828it [18:13,
  6.81it/s][A

Training Epoch: 1/3,
 step 87/249 completed (loss: 0.09445157647132874): : 3828it [18:13,
  6.81it/s][A

Training Epoch: 1/3,
 step 87/249 completed (loss: 0.09445157647132874): : 3916it [18:25,
  6.89it/s][A

Training Epoch: 1/3,
 step 88/249 completed (loss: 0.028109824284911156): : 3916it [18:25,
  6.89it/s][A

Training Epoch: 1/3,
 step 88/249 completed (loss: 0.028109824284911156): : 4005it [18:37,
  6.97it/s][A

Training Epoch: 1/3,
 step 89/249 completed (loss: 0.062209438532590866): : 4005it [18:38,
  6.97it/s][A

Training Epoch: 1/3,
 step 89/249 completed (loss: 0.062209438532590866): : 4095it [18:50,
  7.05it/s][A

Training Epoch: 1/3,
 step 90/249 completed (loss: 0.03266126662492752): : 4095it [18:50,
  7.05it/s] [A

Training Epoch: 1/3,
 step 90/249 completed (loss: 0.03266126662492752): : 4186it [19:02,
  7.13it/s][A

Training Epoch: 1/3,
 step 91/249 completed (loss: 0.05749933421611786): : 4186it [19:02,
  7.13it/s][A

Training Epoch: 1/3,
 step 91/249 completed (loss: 0.05749933421611786): : 4278it [19:15,
  7.21it/s][A

Training Epoch: 1/3,
 step 92/249 completed (loss: 0.043320488184690475): : 4278it [19:15,
  7.21it/s][A

Training Epoch: 1/3,
 step 92/249 completed (loss: 0.043320488184690475): : 4371it [19:27,
  7.29it/s][A

Training Epoch: 1/3,
 step 93/249 completed (loss: 0.0645560696721077): : 4371it [19:27,
  7.29it/s]  [A

Training Epoch: 1/3,
 step 93/249 completed (loss: 0.0645560696721077): : 4465it [19:40,
  7.38it/s][A

Training Epoch: 1/3,
 step 94/249 completed (loss: 0.04421592131257057): : 4465it [19:40,
  7.38it/s][A

Training Epoch: 1/3,
 step 94/249 completed (loss: 0.04421592131257057): : 4560it [19:52,
  7.46it/s][A

Training Epoch: 1/3,
 step 95/249 completed (loss: 0.09086355566978455): : 4560it [19:52,
  7.46it/s][A

Training Epoch: 1/3,
 step 95/249 completed (loss: 0.09086355566978455): : 4656it [20:04,
  7.54it/s][A

Training Epoch: 1/3,
 step 96/249 completed (loss: 0.1444503366947174): : 4656it [20:05,
  7.54it/s] [A

Training Epoch: 1/3,
 step 96/249 completed (loss: 0.1444503366947174): : 4753it [20:17,
  7.62it/s][A

Training Epoch: 1/3,
 step 97/249 completed (loss: 0.04548763856291771): : 4753it [20:17,
  7.62it/s][A

Training Epoch: 1/3,
 step 97/249 completed (loss: 0.04548763856291771): : 4851it [20:29,
  7.70it/s][A

Training Epoch: 1/3,
 step 98/249 completed (loss: 0.0855422168970108): : 4851it [20:29,
  7.70it/s] [A

Training Epoch: 1/3,
 step 98/249 completed (loss: 0.0855422168970108): : 4950it [20:42,
  7.78it/s][A

Training Epoch: 1/3,
 step 99/249 completed (loss: 0.07959310710430145): : 4950it [20:42,
  7.78it/s][A

Training Epoch: 1/3,
 step 99/249 completed (loss: 0.07959310710430145): : 5050it [20:54,
  7.86it/s][A

Training Epoch: 1/3,
 step 100/249 completed (loss: 0.0915798619389534): : 5050it [20:54,
  7.86it/s][A

Training Epoch: 1/3,
 step 100/249 completed (loss: 0.0915798619389534): : 5151it [21:07,
  7.94it/s][A

Training Epoch: 1/3,
 step 101/249 completed (loss: 0.014309454709291458): : 5151it [21:07,
  7.94it/s][A

Training Epoch: 1/3,
 step 101/249 completed (loss: 0.014309454709291458): : 5253it [21:19,
  8.02it/s][A

Training Epoch: 1/3,
 step 102/249 completed (loss: 0.061000678688287735): : 5253it [21:19,
  8.02it/s][A

Training Epoch: 1/3,
 step 102/249 completed (loss: 0.061000678688287735): : 5356it [21:31,
  8.10it/s][A

Training Epoch: 1/3,
 step 103/249 completed (loss: 0.11303357779979706): : 5356it [21:32,
  8.10it/s] [A

Training Epoch: 1/3,
 step 103/249 completed (loss: 0.11303357779979706): : 5460it [21:44,
  8.18it/s][A

Training Epoch: 1/3,
 step 104/249 completed (loss: 0.09884218126535416): : 5460it [21:44,
  8.18it/s][A

Training Epoch: 1/3,
 step 104/249 completed (loss: 0.09884218126535416): : 5565it [21:56,
  8.27it/s][A

Training Epoch: 1/3,
 step 105/249 completed (loss: 0.087089404463768): : 5565it [21:56,
  8.27it/s]  [A

Training Epoch: 1/3,
 step 105/249 completed (loss: 0.087089404463768): : 5671it [22:09,
  8.35it/s][A

Training Epoch: 1/3,
 step 106/249 completed (loss: 0.014341610483825207): : 5671it [22:09,
  8.35it/s][A

Training Epoch: 1/3,
 step 106/249 completed (loss: 0.014341610483825207): : 5778it [22:21,
  8.42it/s][A

Training Epoch: 1/3,
 step 107/249 completed (loss: 0.03750757873058319): : 5778it [22:21,
  8.42it/s] [A

Training Epoch: 1/3,
 step 107/249 completed (loss: 0.03750757873058319): : 5886it [22:33,
  8.50it/s][A

Training Epoch: 1/3,
 step 108/249 completed (loss: 0.07518097013235092): : 5886it [22:34,
  8.50it/s][A

Training Epoch: 1/3,
 step 108/249 completed (loss: 0.07518097013235092): : 5995it [22:46,
  8.58it/s][A

Training Epoch: 1/3,
 step 109/249 completed (loss: 0.09110190719366074): : 5995it [22:46,
  8.58it/s][A

Training Epoch: 1/3,
 step 109/249 completed (loss: 0.09110190719366074): : 6105it [22:58,
  8.67it/s][A

Training Epoch: 1/3,
 step 110/249 completed (loss: 0.024791184812784195): : 6105it [22:59,
  8.67it/s][A

Training Epoch: 1/3,
 step 110/249 completed (loss: 0.024791184812784195): : 6216it [23:11,
  8.74it/s][A

Training Epoch: 1/3,
 step 111/249 completed (loss: 0.1627577841281891): : 6216it [23:11,
  8.74it/s]  [A

Training Epoch: 1/3,
 step 111/249 completed (loss: 0.1627577841281891): : 6328it [23:23,
  8.82it/s][A

Training Epoch: 1/3,
 step 112/249 completed (loss: 0.11996173858642578): : 6328it [23:23,
  8.82it/s][A

Training Epoch: 1/3,
 step 112/249 completed (loss: 0.11996173858642578): : 6441it [23:36,
  8.90it/s][A

Training Epoch: 1/3,
 step 113/249 completed (loss: 0.12406563013792038): : 6441it [23:36,
  8.90it/s][A

Training Epoch: 1/3,
 step 113/249 completed (loss: 0.12406563013792038): : 6555it [23:48,
  8.98it/s][A

Training Epoch: 1/3,
 step 114/249 completed (loss: 0.14510728418827057): : 6555it [23:48,
  8.98it/s][A

Training Epoch: 1/3,
 step 114/249 completed (loss: 0.14510728418827057): : 6670it [24:01,
  9.06it/s][A

Training Epoch: 1/3,
 step 115/249 completed (loss: 0.08494210243225098): : 6670it [24:01,
  9.06it/s][A

Training Epoch: 1/3,
 step 115/249 completed (loss: 0.08494210243225098): : 6786it [24:13,
  9.14it/s][A

Training Epoch: 1/3,
 step 116/249 completed (loss: 0.07773357629776001): : 6786it [24:13,
  9.14it/s][A

Training Epoch: 1/3,
 step 116/249 completed (loss: 0.07773357629776001): : 6903it [24:25,
  9.22it/s][A

Training Epoch: 1/3,
 step 117/249 completed (loss: 0.06774834543466568): : 6903it [24:26,
  9.22it/s][A

Training Epoch: 1/3,
 step 117/249 completed (loss: 0.06774834543466568): : 7021it [24:38,
  9.31it/s][A

Training Epoch: 1/3,
 step 118/249 completed (loss: 0.039528556168079376): : 7021it [24:38,
  9.31it/s][A

Training Epoch: 1/3,
 step 118/249 completed (loss: 0.039528556168079376): : 7140it [24:50,
  9.39it/s][A

Training Epoch: 1/3,
 step 119/249 completed (loss: 0.15649332106113434): : 7140it [24:50,
  9.39it/s] [A

Training Epoch: 1/3,
 step 119/249 completed (loss: 0.15649332106113434): : 7260it [25:03,
  9.47it/s][A

Training Epoch: 1/3,
 step 120/249 completed (loss: 0.019895825535058975): : 7260it [25:03,
  9.47it/s][A

Training Epoch: 1/3,
 step 120/249 completed (loss: 0.019895825535058975): : 7381it [25:15,
  9.55it/s][A

Training Epoch: 1/3,
 step 121/249 completed (loss: 0.009381363168358803): : 7381it [25:15,
  9.55it/s][A

Training Epoch: 1/3,
 step 121/249 completed (loss: 0.009381363168358803): : 7503it [25:27,
  9.63it/s][A

Training Epoch: 1/3,
 step 122/249 completed (loss: 0.05172012373805046): : 7503it [25:28,
  9.63it/s] [A

Training Epoch: 1/3,
 step 122/249 completed (loss: 0.05172012373805046): : 7626it [25:40,
  9.71it/s][A

Training Epoch: 1/3,
 step 123/249 completed (loss: 0.06028098985552788): : 7626it [25:40,
  9.71it/s][A

Training Epoch: 1/3,
 step 123/249 completed (loss: 0.06028098985552788): : 7750it [25:52,
  9.79it/s][A

Training Epoch: 1/3,
 step 124/249 completed (loss: 0.07716725766658783): : 7750it [25:52,
  9.79it/s][A

Training Epoch: 1/3,
 step 124/249 completed (loss: 0.07716725766658783): : 7875it [26:05,
  9.87it/s][A

Training Epoch: 1/3,
 step 125/249 completed (loss: 0.03388487547636032): : 7875it [26:05,
  9.87it/s][A

Training Epoch: 1/3,
 step 125/249 completed (loss: 0.03388487547636032): : 7875it [26:17,
  9.87it/s][A

Training Epoch: 1/3,
 step 125/249 completed (loss: 0.03388487547636032): : 8001it [26:17,
  9.95it/s][A

Training Epoch: 1/3,
 step 126/249 completed (loss: 0.029929397627711296): : 8001it [26:17,
  9.95it/s][A

Training Epoch: 1/3,
 step 126/249 completed (loss: 0.029929397627711296): : 8128it [26:30,
 10.03it/s][A

Training Epoch: 1/3,
 step 127/249 completed (loss: 0.02069728821516037): : 8128it [26:30,
 10.03it/s] [A

Training Epoch: 1/3,
 step 127/249 completed (loss: 0.02069728821516037): : 8256it [26:42,
 10.12it/s][A

Training Epoch: 1/3,
 step 128/249 completed (loss: 0.01634117215871811): : 8256it [26:42,
 10.12it/s][A

Training Epoch: 1/3,
 step 128/249 completed (loss: 0.01634117215871811): : 8385it [26:54,
 10.20it/s][A

Training Epoch: 1/3,
 step 129/249 completed (loss: 0.09848763793706894): : 8385it [26:55,
 10.20it/s][A

Training Epoch: 1/3,
 step 129/249 completed (loss: 0.09848763793706894): : 8385it [27:05,
 10.20it/s][A

Training Epoch: 1/3,
 step 129/249 completed (loss: 0.09848763793706894): : 8515it [27:07,
 10.28it/s][A

Training Epoch: 1/3,
 step 130/249 completed (loss: 0.0033357257489115): : 8515it [27:07,
 10.28it/s] [A

Training Epoch: 1/3,
 step 130/249 completed (loss: 0.0033357257489115): : 8646it [27:19,
 10.35it/s][A

Training Epoch: 1/3,
 step 131/249 completed (loss: 0.10006719082593918): : 8646it [27:19,
 10.35it/s][A

Training Epoch: 1/3,
 step 131/249 completed (loss: 0.10006719082593918): : 8778it [27:32,
 10.44it/s][A

Training Epoch: 1/3,
 step 132/249 completed (loss: 0.06979762762784958): : 8778it [27:32,
 10.44it/s][A

Training Epoch: 1/3,
 step 132/249 completed (loss: 0.06979762762784958): : 8911it [27:44,
 10.51it/s][A

Training Epoch: 1/3,
 step 133/249 completed (loss: 0.08536625653505325): : 8911it [27:44,
 10.51it/s][A

Training Epoch: 1/3,
 step 133/249 completed (loss: 0.08536625653505325): : 8911it [27:55,
 10.51it/s][A

Training Epoch: 1/3,
 step 133/249 completed (loss: 0.08536625653505325): : 9045it [27:57,
 10.59it/s][A

Training Epoch: 1/3,
 step 134/249 completed (loss: 0.05221947655081749): : 9045it [27:57,
 10.59it/s][A

Training Epoch: 1/3,
 step 134/249 completed (loss: 0.05221947655081749): : 9045it [28:07,
 10.59it/s][A

Training Epoch: 1/3,
 step 134/249 completed (loss: 0.05221947655081749): : 9180it [28:09,
 10.67it/s][A

Training Epoch: 1/3,
 step 135/249 completed (loss: 0.029493087902665138): : 9180it [28:09,
 10.67it/s][A

Training Epoch: 1/3,
 step 135/249 completed (loss: 0.029493087902665138): : 9316it [28:21,
 10.76it/s][A

Training Epoch: 1/3,
 step 136/249 completed (loss: 0.019775263965129852): : 9316it [28:22,
 10.76it/s][A

Training Epoch: 1/3,
 step 136/249 completed (loss: 0.019775263965129852): : 9453it [28:34,
 10.83it/s][A

Training Epoch: 1/3,
 step 137/249 completed (loss: 0.11227039247751236): : 9453it [28:34,
 10.83it/s] [A

Training Epoch: 1/3,
 step 137/249 completed (loss: 0.11227039247751236): : 9453it [28:45,
 10.83it/s][A

Training Epoch: 1/3,
 step 137/249 completed (loss: 0.11227039247751236): : 9591it [28:46,
 10.91it/s][A

Training Epoch: 1/3,
 step 138/249 completed (loss: 0.07739299535751343): : 9591it [28:46,
 10.91it/s][A

Training Epoch: 1/3,
 step 138/249 completed (loss: 0.07739299535751343): : 9591it [28:57,
 10.91it/s][A

Training Epoch: 1/3,
 step 138/249 completed (loss: 0.07739299535751343): : 9730it [28:59,
 10.99it/s][A

Training Epoch: 1/3,
 step 139/249 completed (loss: 0.023174546658992767): : 9730it [28:59,
 10.99it/s][A

Training Epoch: 1/3,
 step 139/249 completed (loss: 0.023174546658992767): : 9870it [29:11,
 11.08it/s][A

Training Epoch: 1/3,
 step 140/249 completed (loss: 0.024148903787136078): : 9870it [29:11,
 11.08it/s][A

Training Epoch: 1/3,
 step 140/249 completed (loss: 0.024148903787136078): : 10011it [29:24,
 11.16it/s][A

Training Epoch: 1/3,
 step 141/249 completed (loss: 0.03063521720468998): : 10011it [29:24,
 11.16it/s] [A

Training Epoch: 1/3,
 step 141/249 completed (loss: 0.03063521720468998): : 10011it [29:35,
 11.16it/s][A

Training Epoch: 1/3,
 step 141/249 completed (loss: 0.03063521720468998): : 10153it [29:36,
 11.24it/s][A

Training Epoch: 1/3,
 step 142/249 completed (loss: 0.08032272011041641): : 10153it [29:36,
 11.24it/s][A

Training Epoch: 1/3,
 step 142/249 completed (loss: 0.08032272011041641): : 10153it [29:47,
 11.24it/s][A

Training Epoch: 1/3,
 step 142/249 completed (loss: 0.08032272011041641): : 10296it [29:48,
 11.31it/s][A

Training Epoch: 1/3,
 step 143/249 completed (loss: 0.021483350545167923): : 10296it [29:49,
 11.31it/s][A

Training Epoch: 1/3,
 step 143/249 completed (loss: 0.021483350545167923): : 10440it [30:01,
 11.39it/s][A

Training Epoch: 1/3,
 step 144/249 completed (loss: 0.0685502365231514): : 10440it [30:01,
 11.39it/s]  [A

Training Epoch: 1/3,
 step 144/249 completed (loss: 0.0685502365231514): : 10585it [30:13,
 11.48it/s][A

Training Epoch: 1/3,
 step 145/249 completed (loss: 0.099781334400177): : 10585it [30:13,
 11.48it/s] [A

Training Epoch: 1/3,
 step 145/249 completed (loss: 0.099781334400177): : 10585it [30:25,
 11.48it/s][A

Training Epoch: 1/3,
 step 145/249 completed (loss: 0.099781334400177): : 10731it [30:26,
 11.56it/s][A

Training Epoch: 1/3,
 step 146/249 completed (loss: 0.04770605266094208): : 10731it [30:26,
 11.56it/s][A

Training Epoch: 1/3,
 step 146/249 completed (loss: 0.04770605266094208): : 10731it [30:37,
 11.56it/s][A

Training Epoch: 1/3,
 step 146/249 completed (loss: 0.04770605266094208): : 10878it [30:38,
 11.64it/s][A

Training Epoch: 1/3,
 step 147/249 completed (loss: 0.09269426763057709): : 10878it [30:38,
 11.64it/s][A

Training Epoch: 1/3,
 step 147/249 completed (loss: 0.09269426763057709): : 11026it [30:51,
 11.72it/s][A

Training Epoch: 1/3,
 step 148/249 completed (loss: 0.11177888512611389): : 11026it [30:51,
 11.72it/s][A

Training Epoch: 1/3,
 step 148/249 completed (loss: 0.11177888512611389): : 11175it [31:03,
 11.79it/s][A

Training Epoch: 1/3,
 step 149/249 completed (loss: 0.11283817887306213): : 11175it [31:03,
 11.79it/s][A

Training Epoch: 1/3,
 step 149/249 completed (loss: 0.11283817887306213): : 11175it [31:15,
 11.79it/s][A

Training Epoch: 1/3,
 step 149/249 completed (loss: 0.11283817887306213): : 11325it [31:15,
 11.87it/s][A

Training Epoch: 1/3,
 step 150/249 completed (loss: 0.04211634397506714): : 11325it [31:16,
 11.87it/s][A

Training Epoch: 1/3,
 step 150/249 completed (loss: 0.04211634397506714): : 11325it [31:27,
 11.87it/s][A

Training Epoch: 1/3,
 step 150/249 completed (loss: 0.04211634397506714): : 11476it [31:28,
 11.96it/s][A

Training Epoch: 1/3,
 step 151/249 completed (loss: 0.09374812245368958): : 11476it [31:28,
 11.96it/s][A

Training Epoch: 1/3,
 step 151/249 completed (loss: 0.09374812245368958): : 11628it [31:40,
 12.04it/s][A

Training Epoch: 1/3,
 step 152/249 completed (loss: 0.07199230790138245): : 11628it [31:41,
 12.04it/s][A

Training Epoch: 1/3,
 step 152/249 completed (loss: 0.07199230790138245): : 11781it [31:53,
 12.11it/s][A

Training Epoch: 1/3,
 step 153/249 completed (loss: 0.10169156640768051): : 11781it [31:53,
 12.11it/s][A

Training Epoch: 1/3,
 step 153/249 completed (loss: 0.10169156640768051): : 11781it [32:05,
 12.11it/s][A

Training Epoch: 1/3,
 step 153/249 completed (loss: 0.10169156640768051): : 11935it [32:05,
 12.19it/s][A

Training Epoch: 1/3,
 step 154/249 completed (loss: 0.022458121180534363): : 11935it [32:05,
 12.19it/s][A

Training Epoch: 1/3,
 step 154/249 completed (loss: 0.022458121180534363): : 11935it [32:17,
 12.19it/s][A

Training Epoch: 1/3,
 step 154/249 completed (loss: 0.022458121180534363): : 12090it [32:18,
 12.28it/s][A

Training Epoch: 1/3,
 step 155/249 completed (loss: 0.1260617971420288): : 12090it [32:18,
 12.28it/s]  [A

Training Epoch: 1/3,
 step 155/249 completed (loss: 0.1260617971420288): : 12246it [32:30,
 12.36it/s][A

Training Epoch: 1/3,
 step 156/249 completed (loss: 0.044289927929639816): : 12246it [32:30,
 12.36it/s][A

Training Epoch: 1/3,
 step 156/249 completed (loss: 0.044289927929639816): : 12403it [32:43,
 12.44it/s][A

Training Epoch: 1/3,
 step 157/249 completed (loss: 0.08958294242620468): : 12403it [32:43,
 12.44it/s] [A

Training Epoch: 1/3,
 step 157/249 completed (loss: 0.08958294242620468): : 12403it [32:55,
 12.44it/s][A

Training Epoch: 1/3,
 step 157/249 completed (loss: 0.08958294242620468): : 12561it [32:55,
 12.52it/s][A

Training Epoch: 1/3,
 step 158/249 completed (loss: 0.07173832505941391): : 12561it [32:55,
 12.52it/s][A

Training Epoch: 1/3,
 step 158/249 completed (loss: 0.07173832505941391): : 12561it [33:07,
 12.52it/s][A

Training Epoch: 1/3,
 step 158/249 completed (loss: 0.07173832505941391): : 12720it [33:07,
 12.61it/s][A

Training Epoch: 1/3,
 step 159/249 completed (loss: 0.01197753008455038): : 12720it [33:08,
 12.61it/s][A

Training Epoch: 1/3,
 step 159/249 completed (loss: 0.01197753008455038): : 12880it [33:20,
 12.68it/s][A

Training Epoch: 1/3,
 step 160/249 completed (loss: 0.017958907410502434): : 12880it [33:20,
 12.68it/s][A

Training Epoch: 1/3,
 step 160/249 completed (loss: 0.017958907410502434): : 13041it [33:32,
 12.76it/s][A

Training Epoch: 1/3,
 step 161/249 completed (loss: 0.08032173663377762): : 13041it [33:32,
 12.76it/s] [A

Training Epoch: 1/3,
 step 161/249 completed (loss: 0.08032173663377762): : 13041it [33:45,
 12.76it/s][A

Training Epoch: 1/3,
 step 161/249 completed (loss: 0.08032173663377762): : 13203it [33:45,
 12.83it/s][A

Training Epoch: 1/3,
 step 162/249 completed (loss: 0.034010663628578186): : 13203it [33:45,
 12.83it/s][A

Training Epoch: 1/3,
 step 162/249 completed (loss: 0.034010663628578186): : 13203it [33:55,
 12.83it/s][A

Training Epoch: 1/3,
 step 162/249 completed (loss: 0.034010663628578186): : 13366it [33:57,
 12.92it/s][A

Training Epoch: 1/3,
 step 163/249 completed (loss: 0.053161557763814926): : 13366it [33:57,
 12.92it/s][A

Training Epoch: 1/3,
 step 163/249 completed (loss: 0.053161557763814926): : 13530it [34:10,
 13.00it/s][A

Training Epoch: 1/3,
 step 164/249 completed (loss: 0.03256266936659813): : 13530it [34:10,
 13.00it/s] [A

Training Epoch: 1/3,
 step 164/249 completed (loss: 0.03256266936659813): : 13695it [34:22,
 13.09it/s][A

Training Epoch: 1/3,
 step 165/249 completed (loss: 0.022244488820433617): : 13695it [34:22,
 13.09it/s][A

Training Epoch: 1/3,
 step 165/249 completed (loss: 0.022244488820433617): : 13861it [34:34,
 13.17it/s][A

Training Epoch: 1/3,
 step 166/249 completed (loss: 0.04396108537912369): : 13861it [34:35,
 13.17it/s] [A

Training Epoch: 1/3,
 step 166/249 completed (loss: 0.04396108537912369): : 13861it [34:45,
 13.17it/s][A

Training Epoch: 1/3,
 step 166/249 completed (loss: 0.04396108537912369): : 14028it [34:47,
 13.25it/s][A

Training Epoch: 1/3,
 step 167/249 completed (loss: 0.013418249785900116): : 14028it [34:47,
 13.25it/s][A

Training Epoch: 1/3,
 step 167/249 completed (loss: 0.013418249785900116): : 14196it [34:59,
 13.33it/s][A

Training Epoch: 1/3,
 step 168/249 completed (loss: 0.04253425821661949): : 14196it [34:59,
 13.33it/s] [A

Training Epoch: 1/3,
 step 168/249 completed (loss: 0.04253425821661949): : 14365it [35:12,
 13.41it/s][A

Training Epoch: 1/3,
 step 169/249 completed (loss: 0.007075178436934948): : 14365it [35:12,
 13.41it/s][A

Training Epoch: 1/3,
 step 169/249 completed (loss: 0.007075178436934948): : 14535it [35:24,
 13.49it/s][A

Training Epoch: 1/3,
 step 170/249 completed (loss: 0.023508865386247635): : 14535it [35:24,
 13.49it/s][A

Training Epoch: 1/3,
 step 170/249 completed (loss: 0.023508865386247635): : 14535it [35:35,
 13.49it/s][A

Training Epoch: 1/3,
 step 170/249 completed (loss: 0.023508865386247635): : 14706it [35:37,
 13.57it/s][A

Training Epoch: 1/3,
 step 171/249 completed (loss: 0.008713534101843834): : 14706it [35:37,
 13.57it/s][A

Training Epoch: 1/3,
 step 171/249 completed (loss: 0.008713534101843834): : 14706it [35:47,
 13.57it/s][A

Training Epoch: 1/3,
 step 171/249 completed (loss: 0.008713534101843834): : 14878it [35:49,
 13.65it/s][A

Training Epoch: 1/3,
 step 172/249 completed (loss: 0.026698438450694084): : 14878it [35:49,
 13.65it/s][A

Training Epoch: 1/3,
 step 172/249 completed (loss: 0.026698438450694084): : 15051it [36:01,
 13.72it/s][A

Training Epoch: 1/3,
 step 173/249 completed (loss: 0.024552440270781517): : 15051it [36:02,
 13.72it/s][A

Training Epoch: 1/3,
 step 173/249 completed (loss: 0.024552440270781517): : 15225it [36:14,
 13.80it/s][A

Training Epoch: 1/3,
 step 174/249 completed (loss: 0.033930420875549316): : 15225it [36:14,
 13.80it/s][A

Training Epoch: 1/3,
 step 174/249 completed (loss: 0.033930420875549316): : 15225it [36:25,
 13.80it/s][A

Training Epoch: 1/3,
 step 174/249 completed (loss: 0.033930420875549316): : 15400it [36:26,
 13.89it/s][A

Training Epoch: 1/3,
 step 175/249 completed (loss: 0.15367572009563446): : 15400it [36:26,
 13.89it/s] [A

Training Epoch: 1/3,
 step 175/249 completed (loss: 0.15367572009563446): : 15400it [36:37,
 13.89it/s][A

Training Epoch: 1/3,
 step 175/249 completed (loss: 0.15367572009563446): : 15576it [36:39,
 13.94it/s][A

Training Epoch: 1/3,
 step 176/249 completed (loss: 0.1404782384634018): : 15576it [36:39,
 13.94it/s] [A

Training Epoch: 1/3,
 step 176/249 completed (loss: 0.1404782384634018): : 15753it [36:51,
 14.03it/s][A

Training Epoch: 1/3,
 step 177/249 completed (loss: 0.02822151966392994): : 15753it [36:51,
 14.03it/s][A

Training Epoch: 1/3,
 step 177/249 completed (loss: 0.02822151966392994): : 15931it [37:04,
 14.12it/s][A

Training Epoch: 1/3,
 step 178/249 completed (loss: 0.03834116831421852): : 15931it [37:04,
 14.12it/s][A

Training Epoch: 1/3,
 step 178/249 completed (loss: 0.03834116831421852): : 15931it [37:15,
 14.12it/s][A

Training Epoch: 1/3,
 step 178/249 completed (loss: 0.03834116831421852): : 16110it [37:16,
 14.21it/s][A

Training Epoch: 1/3,
 step 179/249 completed (loss: 0.0028163702227175236): : 16110it [37:16,
 14.21it/s][A

Training Epoch: 1/3,
 step 179/249 completed (loss: 0.0028163702227175236): : 16110it [37:27,
 14.21it/s][A

Training Epoch: 1/3,
 step 179/249 completed (loss: 0.0028163702227175236): : 16290it [37:28,
 14.29it/s][A

Training Epoch: 1/3,
 step 180/249 completed (loss: 0.08899659663438797): : 16290it [37:29,
 14.29it/s]  [A

Training Epoch: 1/3,
 step 180/249 completed (loss: 0.08899659663438797): : 16471it [37:41,
 14.37it/s][A

Training Epoch: 1/3,
 step 181/249 completed (loss: 0.20562244951725006): : 16471it [37:41,
 14.37it/s][A

Training Epoch: 1/3,
 step 181/249 completed (loss: 0.20562244951725006): : 16653it [37:53,
 14.46it/s][A

Training Epoch: 1/3,
 step 182/249 completed (loss: 0.09415645152330399): : 16653it [37:54,
 14.46it/s][A

Training Epoch: 1/3,
 step 182/249 completed (loss: 0.09415645152330399): : 16653it [38:05,
 14.46it/s][A

Training Epoch: 1/3,
 step 182/249 completed (loss: 0.09415645152330399): : 16836it [38:06,
 14.53it/s][A

Training Epoch: 1/3,
 step 183/249 completed (loss: 0.1490137279033661): : 16836it [38:06,
 14.53it/s] [A

Training Epoch: 1/3,
 step 183/249 completed (loss: 0.1490137279033661): : 16836it [38:17,
 14.53it/s][A

Training Epoch: 1/3,
 step 183/249 completed (loss: 0.1490137279033661): : 17020it [38:18,
 14.62it/s][A

Training Epoch: 1/3,
 step 184/249 completed (loss: 0.04928087815642357): : 17020it [38:18,
 14.62it/s][A

Training Epoch: 1/3,
 step 184/249 completed (loss: 0.04928087815642357): : 17205it [38:31,
 14.70it/s][A

Training Epoch: 1/3,
 step 185/249 completed (loss: 0.06515785306692123): : 17205it [38:31,
 14.70it/s][A

Training Epoch: 1/3,
 step 185/249 completed (loss: 0.06515785306692123): : 17391it [38:43,
 14.78it/s][A

Training Epoch: 1/3,
 step 186/249 completed (loss: 0.11137767136096954): : 17391it [38:43,
 14.78it/s][A

Training Epoch: 1/3,
 step 186/249 completed (loss: 0.11137767136096954): : 17391it [38:55,
 14.78it/s][A

Training Epoch: 1/3,
 step 186/249 completed (loss: 0.11137767136096954): : 17578it [38:55,
 14.86it/s][A

Training Epoch: 1/3,
 step 187/249 completed (loss: 0.031912535429000854): : 17578it [38:56,
 14.86it/s][A

Training Epoch: 1/3,
 step 187/249 completed (loss: 0.031912535429000854): : 17578it [39:07,
 14.86it/s][A

Training Epoch: 1/3,
 step 187/249 completed (loss: 0.031912535429000854): : 17766it [39:08,
 14.94it/s][A

Training Epoch: 1/3,
 step 188/249 completed (loss: 0.044998399913311005): : 17766it [39:08,
 14.94it/s][A

Training Epoch: 1/3,
 step 188/249 completed (loss: 0.044998399913311005): : 17955it [39:20,
 15.02it/s][A

Training Epoch: 1/3,
 step 189/249 completed (loss: 0.0858033150434494): : 17955it [39:21,
 15.02it/s]  [A

Training Epoch: 1/3,
 step 189/249 completed (loss: 0.0858033150434494): : 18145it [39:33,
 15.11it/s][A

Training Epoch: 1/3,
 step 190/249 completed (loss: 0.08385535329580307): : 18145it [39:33,
 15.11it/s][A

Training Epoch: 1/3,
 step 190/249 completed (loss: 0.08385535329580307): : 18145it [39:45,
 15.11it/s][A

Training Epoch: 1/3,
 step 190/249 completed (loss: 0.08385535329580307): : 18336it [39:45,
 15.19it/s][A

Training Epoch: 1/3,
 step 191/249 completed (loss: 0.11117003858089447): : 18336it [39:45,
 15.19it/s][A

Training Epoch: 1/3,
 step 191/249 completed (loss: 0.11117003858089447): : 18336it [39:57,
 15.19it/s][A

Training Epoch: 1/3,
 step 191/249 completed (loss: 0.11117003858089447): : 18528it [39:58,
 15.26it/s][A

Training Epoch: 1/3,
 step 192/249 completed (loss: 0.21036654710769653): : 18528it [39:58,
 15.26it/s][A

Training Epoch: 1/3,
 step 192/249 completed (loss: 0.21036654710769653): : 18721it [40:10,
 15.35it/s][A

Training Epoch: 1/3,
 step 193/249 completed (loss: 0.1130034476518631): : 18721it [40:10,
 15.35it/s] [A

Training Epoch: 1/3,
 step 193/249 completed (loss: 0.1130034476518631): : 18915it [40:22,
 15.42it/s][A

Training Epoch: 1/3,
 step 194/249 completed (loss: 0.08256532996892929): : 18915it [40:23,
 15.42it/s][A

Training Epoch: 1/3,
 step 194/249 completed (loss: 0.08256532996892929): : 18915it [40:35,
 15.42it/s][A

Training Epoch: 1/3,
 step 194/249 completed (loss: 0.08256532996892929): : 19110it [40:35,
 15.51it/s][A

Training Epoch: 1/3,
 step 195/249 completed (loss: 0.0460723340511322): : 19110it [40:35,
 15.51it/s] [A

Training Epoch: 1/3,
 step 195/249 completed (loss: 0.0460723340511322): : 19110it [40:47,
 15.51it/s][A

Training Epoch: 1/3,
 step 195/249 completed (loss: 0.0460723340511322): : 19306it [40:47,
 15.58it/s][A

Training Epoch: 1/3,
 step 196/249 completed (loss: 0.037735916674137115): : 19306it [40:47,
 15.58it/s][A

Training Epoch: 1/3,
 step 196/249 completed (loss: 0.037735916674137115): : 19503it [41:00,
 15.66it/s][A

Training Epoch: 1/3,
 step 197/249 completed (loss: 0.1193118542432785): : 19503it [41:00,
 15.66it/s]  [A

Training Epoch: 1/3,
 step 197/249 completed (loss: 0.1193118542432785): : 19701it [41:12,
 15.75it/s][A

Training Epoch: 1/3,
 step 198/249 completed (loss: 0.1554635912179947): : 19701it [41:12,
 15.75it/s][A

Training Epoch: 1/3,
 step 198/249 completed (loss: 0.1554635912179947): : 19900it [41:25,
 15.83it/s][A

Training Epoch: 1/3,
 step 199/249 completed (loss: 0.18427664041519165): : 19900it [41:25,
 15.83it/s][A

Training Epoch: 1/3,
 step 199/249 completed (loss: 0.18427664041519165): : 19900it [41:35,
 15.83it/s][A

Training Epoch: 1/3,
 step 199/249 completed (loss: 0.18427664041519165): : 20100it [41:37,
 15.90it/s][A

Training Epoch: 1/3,
 step 200/249 completed (loss: 0.04961210861802101): : 20100it [41:37,
 15.90it/s][A

Training Epoch: 1/3,
 step 200/249 completed (loss: 0.04961210861802101): : 20301it [41:49,
 15.98it/s][A

Training Epoch: 1/3,
 step 201/249 completed (loss: 0.05031074583530426): : 20301it [41:50,
 15.98it/s][A

Training Epoch: 1/3,
 step 201/249 completed (loss: 0.05031074583530426): : 20503it [42:02,
 16.06it/s][A

Training Epoch: 1/3,
 step 202/249 completed (loss: 0.04973524808883667): : 20503it [42:02,
 16.06it/s][A

Training Epoch: 1/3,
 step 202/249 completed (loss: 0.04973524808883667): : 20706it [42:14,
 16.15it/s][A

Training Epoch: 1/3,
 step 203/249 completed (loss: 0.02065931260585785): : 20706it [42:14,
 16.15it/s][A

Training Epoch: 1/3,
 step 203/249 completed (loss: 0.02065931260585785): : 20706it [42:25,
 16.15it/s][A

Training Epoch: 1/3,
 step 203/249 completed (loss: 0.02065931260585785): : 20910it [42:27,
 16.23it/s][A

Training Epoch: 1/3,
 step 204/249 completed (loss: 0.010967924259603024): : 20910it [42:27,
 16.23it/s][A

Training Epoch: 1/3,
 step 204/249 completed (loss: 0.010967924259603024): : 20910it [42:37,
 16.23it/s][A

Training Epoch: 1/3,
 step 204/249 completed (loss: 0.010967924259603024): : 21115it [42:39,
 16.31it/s][A

Training Epoch: 1/3,
 step 205/249 completed (loss: 0.06142709031701088): : 21115it [42:39,
 16.31it/s] [A

Training Epoch: 1/3,
 step 205/249 completed (loss: 0.06142709031701088): : 21321it [42:52,
 16.40it/s][A

Training Epoch: 1/3,
 step 206/249 completed (loss: 0.014264618046581745): : 21321it [42:52,
 16.40it/s][A

Training Epoch: 1/3,
 step 206/249 completed (loss: 0.014264618046581745): : 21528it [43:04,
 16.47it/s][A

Training Epoch: 1/3,
 step 207/249 completed (loss: 0.03930244967341423): : 21528it [43:04,
 16.47it/s] [A

Training Epoch: 1/3,
 step 207/249 completed (loss: 0.03930244967341423): : 21528it [43:15,
 16.47it/s][A

Training Epoch: 1/3,
 step 207/249 completed (loss: 0.03930244967341423): : 21736it [43:16,
 16.55it/s][A

Training Epoch: 1/3,
 step 208/249 completed (loss: 0.0802297592163086): : 21736it [43:17,
 16.55it/s] [A

Training Epoch: 1/3,
 step 208/249 completed (loss: 0.0802297592163086): : 21736it [43:27,
 16.55it/s][A

Training Epoch: 1/3,
 step 208/249 completed (loss: 0.0802297592163086): : 21945it [43:29,
 16.63it/s][A

Training Epoch: 1/3,
 step 209/249 completed (loss: 0.0955299586057663): : 21945it [43:29,
 16.63it/s][A

Training Epoch: 1/3,
 step 209/249 completed (loss: 0.0955299586057663): : 22155it [43:41,
 16.71it/s][A

Training Epoch: 1/3,
 step 210/249 completed (loss: 0.04876650869846344): : 22155it [43:41,
 16.71it/s][A

Training Epoch: 1/3,
 step 210/249 completed (loss: 0.04876650869846344): : 22366it [43:54,
 16.80it/s][A

Training Epoch: 1/3,
 step 211/249 completed (loss: 0.035213496536016464): : 22366it [43:54,
 16.80it/s][A

Training Epoch: 1/3,
 step 211/249 completed (loss: 0.035213496536016464): : 22366it [44:05,
 16.80it/s][A

Training Epoch: 1/3,
 step 211/249 completed (loss: 0.035213496536016464): : 22578it [44:06,
 16.87it/s][A

Training Epoch: 1/3,
 step 212/249 completed (loss: 0.08847659081220627): : 22578it [44:06,
 16.87it/s] [A

Training Epoch: 1/3,
 step 212/249 completed (loss: 0.08847659081220627): : 22578it [44:17,
 16.87it/s][A

Training Epoch: 1/3,
 step 212/249 completed (loss: 0.08847659081220627): : 22791it [44:19,
 16.95it/s][A

Training Epoch: 1/3,
 step 213/249 completed (loss: 0.0820218175649643): : 22791it [44:19,
 16.95it/s] [A

Training Epoch: 1/3,
 step 213/249 completed (loss: 0.0820218175649643): : 23005it [44:31,
 17.03it/s][A

Training Epoch: 1/3,
 step 214/249 completed (loss: 0.04125489667057991): : 23005it [44:31,
 17.03it/s][A

Training Epoch: 1/3,
 step 214/249 completed (loss: 0.04125489667057991): : 23220it [44:43,
 17.10it/s][A

Training Epoch: 1/3,
 step 215/249 completed (loss: 0.06570667028427124): : 23220it [44:44,
 17.10it/s][A

Training Epoch: 1/3,
 step 215/249 completed (loss: 0.06570667028427124): : 23220it [44:55,
 17.10it/s][A

Training Epoch: 1/3,
 step 215/249 completed (loss: 0.06570667028427124): : 23436it [44:56,
 17.18it/s][A

Training Epoch: 1/3,
 step 216/249 completed (loss: 0.0707252100110054): : 23436it [44:56,
 17.18it/s] [A

Training Epoch: 1/3,
 step 216/249 completed (loss: 0.0707252100110054): : 23436it [45:07,
 17.18it/s][A

Training Epoch: 1/3,
 step 216/249 completed (loss: 0.0707252100110054): : 23653it [45:08,
 17.27it/s][A

Training Epoch: 1/3,
 step 217/249 completed (loss: 0.03680785000324249): : 23653it [45:08,
 17.27it/s][A

Training Epoch: 1/3,
 step 217/249 completed (loss: 0.03680785000324249): : 23871it [45:21,
 17.35it/s][A

Training Epoch: 1/3,
 step 218/249 completed (loss: 0.09709443151950836): : 23871it [45:21,
 17.35it/s][A

Training Epoch: 1/3,
 step 218/249 completed (loss: 0.09709443151950836): : 24090it [45:33,
 17.43it/s][A

Training Epoch: 1/3,
 step 219/249 completed (loss: 0.08987141400575638): : 24090it [45:33,
 17.43it/s][A

Training Epoch: 1/3,
 step 219/249 completed (loss: 0.08987141400575638): : 24090it [45:45,
 17.43it/s][A

Training Epoch: 1/3,
 step 219/249 completed (loss: 0.08987141400575638): : 24310it [45:46,
 17.50it/s][A

Training Epoch: 1/3,
 step 220/249 completed (loss: 0.10094738006591797): : 24310it [45:46,
 17.50it/s][A

Training Epoch: 1/3,
 step 220/249 completed (loss: 0.10094738006591797): : 24310it [45:57,
 17.50it/s][A

Training Epoch: 1/3,
 step 220/249 completed (loss: 0.10094738006591797): : 24531it [45:58,
 17.59it/s][A

Training Epoch: 1/3,
 step 221/249 completed (loss: 0.018199583515524864): : 24531it [45:58,
 17.59it/s][A

Training Epoch: 1/3,
 step 221/249 completed (loss: 0.018199583515524864): : 24753it [46:10,
 17.67it/s][A

Training Epoch: 1/3,
 step 222/249 completed (loss: 0.0626237541437149): : 24753it [46:11,
 17.67it/s]  [A

Training Epoch: 1/3,
 step 222/249 completed (loss: 0.0626237541437149): : 24976it [46:23,
 17.75it/s][A

Training Epoch: 1/3,
 step 223/249 completed (loss: 0.06496953964233398): : 24976it [46:23,
 17.75it/s][A

Training Epoch: 1/3,
 step 223/249 completed (loss: 0.06496953964233398): : 24976it [46:35,
 17.75it/s][A

Training Epoch: 1/3,
 step 223/249 completed (loss: 0.06496953964233398): : 25200it [46:35,
 17.83it/s][A

Training Epoch: 1/3,
 step 224/249 completed (loss: 0.10780246555805206): : 25200it [46:35,
 17.83it/s][A

Training Epoch: 1/3,
 step 224/249 completed (loss: 0.10780246555805206): : 25200it [46:47,
 17.83it/s][A

Training Epoch: 1/3,
 step 224/249 completed (loss: 0.10780246555805206): : 25425it [46:48,
 17.92it/s][A

Training Epoch: 1/3,
 step 225/249 completed (loss: 0.04154970124363899): : 25425it [46:48,
 17.92it/s][A

Training Epoch: 1/3,
 step 225/249 completed (loss: 0.04154970124363899): : 25651it [47:00,
 17.99it/s][A

Training Epoch: 1/3,
 step 226/249 completed (loss: 0.09854774922132492): : 25651it [47:00,
 17.99it/s][A

Training Epoch: 1/3,
 step 226/249 completed (loss: 0.09854774922132492): : 25878it [47:13,
 18.08it/s][A

Training Epoch: 1/3,
 step 227/249 completed (loss: 0.08087582141160965): : 25878it [47:13,
 18.08it/s][A

Training Epoch: 1/3,
 step 227/249 completed (loss: 0.08087582141160965): : 25878it [47:25,
 18.08it/s][A

Training Epoch: 1/3,
 step 227/249 completed (loss: 0.08087582141160965): : 26106it [47:25,
 18.16it/s][A

Training Epoch: 1/3,
 step 228/249 completed (loss: 0.0714595764875412): : 26106it [47:25,
 18.16it/s] [A

Training Epoch: 1/3,
 step 228/249 completed (loss: 0.0714595764875412): : 26106it [47:37,
 18.16it/s][A

Training Epoch: 1/3,
 step 228/249 completed (loss: 0.0714595764875412): : 26335it [47:37,
 18.24it/s][A

Training Epoch: 1/3,
 step 229/249 completed (loss: 0.06801801174879074): : 26335it [47:38,
 18.24it/s][A

Training Epoch: 1/3,
 step 229/249 completed (loss: 0.06801801174879074): : 26565it [47:50,
 18.32it/s][A

Training Epoch: 1/3,
 step 230/249 completed (loss: 0.055119629949331284): : 26565it [47:50,
 18.32it/s][A

Training Epoch: 1/3,
 step 230/249 completed (loss: 0.055119629949331284): : 26796it [48:02,
 18.40it/s][A

Training Epoch: 1/3,
 step 231/249 completed (loss: 0.022322561591863632): : 26796it [48:02,
 18.40it/s][A

Training Epoch: 1/3,
 step 231/249 completed (loss: 0.022322561591863632): : 27028it [48:15,
 18.48it/s][A

Training Epoch: 1/3,
 step 232/249 completed (loss: 0.08156869560480118): : 27028it [48:15,
 18.48it/s] [A

Training Epoch: 1/3,
 step 232/249 completed (loss: 0.08156869560480118): : 27028it [48:25,
 18.48it/s][A

Training Epoch: 1/3,
 step 232/249 completed (loss: 0.08156869560480118): : 27261it [48:27,
 18.56it/s][A

Training Epoch: 1/3,
 step 233/249 completed (loss: 0.026343809440732002): : 27261it [48:27,
 18.56it/s][A

Training Epoch: 1/3,
 step 233/249 completed (loss: 0.026343809440732002): : 27495it [48:40,
 18.65it/s][A

Training Epoch: 1/3,
 step 234/249 completed (loss: 0.03606647625565529): : 27495it [48:40,
 18.65it/s] [A

Training Epoch: 1/3,
 step 234/249 completed (loss: 0.03606647625565529): : 27730it [48:52,
 18.73it/s][A

Training Epoch: 1/3,
 step 235/249 completed (loss: 0.056709226220846176): : 27730it [48:52,
 18.73it/s][A

Training Epoch: 1/3,
 step 235/249 completed (loss: 0.056709226220846176): : 27966it [49:04,
 18.81it/s][A

Training Epoch: 1/3,
 step 236/249 completed (loss: 0.0648680329322815): : 27966it [49:05,
 18.81it/s]  [A

Training Epoch: 1/3,
 step 236/249 completed (loss: 0.0648680329322815): : 27966it [49:15,
 18.81it/s][A

Training Epoch: 1/3,
 step 236/249 completed (loss: 0.0648680329322815): : 28203it [49:17,
 18.89it/s][A

Training Epoch: 1/3,
 step 237/249 completed (loss: 0.09981387853622437): : 28203it [49:17,
 18.89it/s][A

Training Epoch: 1/3,
 step 237/249 completed (loss: 0.09981387853622437): : 28441it [49:29,
 18.98it/s][A

Training Epoch: 1/3,
 step 238/249 completed (loss: 0.06948988139629364): : 28441it [49:29,
 18.98it/s][A

Training Epoch: 1/3,
 step 238/249 completed (loss: 0.06948988139629364): : 28680it [49:42,
 19.03it/s][A

Training Epoch: 1/3,
 step 239/249 completed (loss: 0.034947268664836884): : 28680it [49:42,
 19.03it/s][A

Training Epoch: 1/3,
 step 239/249 completed (loss: 0.034947268664836884): : 28920it [49:54,
 19.10it/s][A

Training Epoch: 1/3,
 step 240/249 completed (loss: 0.07673721015453339): : 28920it [49:54,
 19.10it/s] [A

Training Epoch: 1/3,
 step 240/249 completed (loss: 0.07673721015453339): : 28920it [50:05,
 19.10it/s][A

Training Epoch: 1/3,
 step 240/249 completed (loss: 0.07673721015453339): : 29161it [50:07,
 19.19it/s][A

Training Epoch: 1/3,
 step 241/249 completed (loss: 0.14377357065677643): : 29161it [50:07,
 19.19it/s][A

Training Epoch: 1/3,
 step 241/249 completed (loss: 0.14377357065677643): : 29161it [50:17,
 19.19it/s][A

Training Epoch: 1/3,
 step 241/249 completed (loss: 0.14377357065677643): : 29403it [50:19,
 19.27it/s][A

Training Epoch: 1/3,
 step 242/249 completed (loss: 0.020971069112420082): : 29403it [50:19,
 19.27it/s][A

Training Epoch: 1/3,
 step 242/249 completed (loss: 0.020971069112420082): : 29646it [50:31,
 19.35it/s][A

Training Epoch: 1/3,
 step 243/249 completed (loss: 0.0037471072282642126): : 29646it [50:32,
 19.35it/s][A

Training Epoch: 1/3,
 step 243/249 completed (loss: 0.0037471072282642126): : 29890it [50:44,
 19.44it/s][A

Training Epoch: 1/3,
 step 244/249 completed (loss: 0.07537484169006348): : 29890it [50:44,
 19.44it/s]  [A

Training Epoch: 1/3,
 step 244/249 completed (loss: 0.07537484169006348): : 29890it [50:55,
 19.44it/s][A

Training Epoch: 1/3,
 step 244/249 completed (loss: 0.07537484169006348): : 30135it [50:56,
 19.53it/s][A

Training Epoch: 1/3,
 step 245/249 completed (loss: 0.08011703938245773): : 30135it [50:56,
 19.53it/s][A

Training Epoch: 1/3,
 step 245/249 completed (loss: 0.08011703938245773): : 30135it [51:07,
 19.53it/s][A

Training Epoch: 1/3,
 step 245/249 completed (loss: 0.08011703938245773): : 30381it [51:09,
 19.61it/s][A

Training Epoch: 1/3,
 step 246/249 completed (loss: 0.04648366570472717): : 30381it [51:09,
 19.61it/s][A

Training Epoch: 1/3,
 step 246/249 completed (loss: 0.04648366570472717): : 30628it [51:21,
 19.69it/s][A

Training Epoch: 1/3,
 step 247/249 completed (loss: 0.08548961579799652): : 30628it [51:21,
 19.69it/s][A

Training Epoch: 1/3,
 step 247/249 completed (loss: 0.08548961579799652): : 30876it [51:34,
 19.78it/s][A

Training Epoch: 1/3,
 step 248/249 completed (loss: 0.005063027609139681): : 30876it [51:34,
 19.78it/s][AMax CUDA memory allocated was 9 GB
Max CUDA memory reserved was 12 GB
Peak active CUDA memory was 9 GB
Cuda Malloc retires : 0
CPU Total Peak Memory consumed during the train (max): 10 GB

evaluating Epoch:   0%|[32m          [0m| 0/200 [00:00<?,
 ?it/s]
evaluating Epoch:   0%|[32m          [0m| 1/200 [00:00<02:18,
  1.44it/s]
evaluating Epoch:   1%|[32m          [0m| 2/200 [00:01<02:02,
  1.61it/s]
evaluating Epoch:   2%|[32m▏         [0m| 3/200 [00:01<01:57,
  1.67it/s]
evaluating Epoch:   2%|[32m▏         [0m| 4/200 [00:02<01:55,
  1.70it/s]
evaluating Epoch:   2%|[32m▎         [0m| 5/200 [00:02<01:54,
  1.71it/s]
evaluating Epoch:   3%|[32m▎         [0m| 6/200 [00:03<01:52,
  1.73it/s]
evaluating Epoch:   4%|[32m▎         [0m| 7/200 [00:04<01:51,
  1.72it/s]
evaluating Epoch:   4%|[32m▍         [0m| 8/200 [00:04<01:49,
  1.75it/s]
evaluating Epoch:   4%|[32m▍         [0m| 9/200 [00:05<01:48,
  1.75it/s]
evaluating Epoch:   5%|[32m▌         [0m| 10/200 [00:05<01:48,
  1.76it/s]
evaluating Epoch:   6%|[32m▌         [0m| 11/200 [00:06<01:47,
  1.75it/s]
evaluating Epoch:   6%|[32m▌         [0m| 12/200 [00:06<01:46,
  1.76it/s]
evaluating Epoch:   6%|[32m▋         [0m| 13/200 [00:07<01:45,
  1.77it/s]
evaluating Epoch:   7%|[32m▋         [0m| 14/200 [00:08<01:45,
  1.76it/s]
evaluating Epoch:   8%|[32m▊         [0m| 15/200 [00:08<01:46,
  1.74it/s]
evaluating Epoch:   8%|[32m▊         [0m| 16/200 [00:09<01:44,
  1.76it/s]
evaluating Epoch:   8%|[32m▊         [0m| 17/200 [00:09<01:43,
  1.77it/s]
evaluating Epoch:   9%|[32m▉         [0m| 18/200 [00:10<01:42,
  1.78it/s]

Training Epoch: 1/3,
 step 248/249 completed (loss: 0.005063027609139681): : 30876it [51:45,
 19.78it/s][A
evaluating Epoch:  10%|[32m▉         [0m| 19/200 [00:10<01:42,
  1.76it/s]
evaluating Epoch:  10%|[32m█         [0m| 20/200 [00:11<01:43,
  1.74it/s]
evaluating Epoch:  10%|[32m█         [0m| 21/200 [00:12<01:42,
  1.74it/s]
evaluating Epoch:  11%|[32m█         [0m| 22/200 [00:12<01:42,
  1.74it/s]
evaluating Epoch:  12%|[32m█▏        [0m| 23/200 [00:13<01:41,
  1.74it/s]
evaluating Epoch:  12%|[32m█▏        [0m| 24/200 [00:13<01:41,
  1.74it/s]
evaluating Epoch:  12%|[32m█▎        [0m| 25/200 [00:14<01:40,
  1.74it/s]
evaluating Epoch:  13%|[32m█▎        [0m| 26/200 [00:14<01:39,
  1.74it/s]
evaluating Epoch:  14%|[32m█▎        [0m| 27/200 [00:15<01:39,
  1.75it/s]
evaluating Epoch:  14%|[32m█▍        [0m| 28/200 [00:16<01:38,
  1.74it/s]
evaluating Epoch:  14%|[32m█▍        [0m| 29/200 [00:16<01:38,
  1.74it/s]
evaluating Epoch:  15%|[32m█▌        [0m| 30/200 [00:17<01:36,
  1.77it/s]
evaluating Epoch:  16%|[32m█▌        [0m| 31/200 [00:17<01:35,
  1.76it/s]
evaluating Epoch:  16%|[32m█▌        [0m| 32/200 [00:18<01:35,
  1.76it/s]
evaluating Epoch:  16%|[32m█▋        [0m| 33/200 [00:18<01:34,
  1.76it/s]
evaluating Epoch:  17%|[32m█▋        [0m| 34/200 [00:19<01:33,
  1.78it/s]
evaluating Epoch:  18%|[32m█▊        [0m| 35/200 [00:20<01:33,
  1.76it/s]
evaluating Epoch:  18%|[32m█▊        [0m| 36/200 [00:20<01:32,
  1.78it/s]
evaluating Epoch:  18%|[32m█▊        [0m| 37/200 [00:21<01:31,
  1.78it/s]
evaluating Epoch:  19%|[32m█▉        [0m| 38/200 [00:21<01:31,
  1.77it/s]
evaluating Epoch:  20%|[32m█▉        [0m| 39/200 [00:22<01:31,
  1.76it/s]
evaluating Epoch:  20%|[32m██        [0m| 40/200 [00:22<01:31,
  1.75it/s]
evaluating Epoch:  20%|[32m██        [0m| 41/200 [00:23<01:30,
  1.76it/s]
evaluating Epoch:  21%|[32m██        [0m| 42/200 [00:24<01:28,
  1.78it/s]
evaluating Epoch:  22%|[32m██▏       [0m| 43/200 [00:24<01:27,
  1.79it/s]
evaluating Epoch:  22%|[32m██▏       [0m| 44/200 [00:25<01:27,
  1.79it/s]
evaluating Epoch:  22%|[32m██▎       [0m| 45/200 [00:25<01:26,
  1.80it/s]
evaluating Epoch:  23%|[32m██▎       [0m| 46/200 [00:26<01:25,
  1.79it/s]
evaluating Epoch:  24%|[32m██▎       [0m| 47/200 [00:26<01:25,
  1.79it/s]
evaluating Epoch:  24%|[32m██▍       [0m| 48/200 [00:27<01:24,
  1.81it/s]
evaluating Epoch:  24%|[32m██▍       [0m| 49/200 [00:27<01:24,
  1.79it/s]
evaluating Epoch:  25%|[32m██▌       [0m| 50/200 [00:28<01:24,
  1.78it/s]
evaluating Epoch:  26%|[32m██▌       [0m| 51/200 [00:29<01:24,
  1.77it/s]
evaluating Epoch:  26%|[32m██▌       [0m| 52/200 [00:29<01:23,
  1.77it/s]
evaluating Epoch:  26%|[32m██▋       [0m| 53/200 [00:30<01:22,
  1.78it/s]
evaluating Epoch:  27%|[32m██▋       [0m| 54/200 [00:30<01:21,
  1.80it/s]
evaluating Epoch:  28%|[32m██▊       [0m| 55/200 [00:31<01:20,
  1.80it/s]
evaluating Epoch:  28%|[32m██▊       [0m| 56/200 [00:31<01:20,
  1.80it/s]
evaluating Epoch:  28%|[32m██▊       [0m| 57/200 [00:32<01:19,
  1.79it/s]
evaluating Epoch:  29%|[32m██▉       [0m| 58/200 [00:32<01:19,
  1.79it/s]
evaluating Epoch:  30%|[32m██▉       [0m| 59/200 [00:33<01:19,
  1.77it/s]
evaluating Epoch:  30%|[32m███       [0m| 60/200 [00:34<01:19,
  1.75it/s]
evaluating Epoch:  30%|[32m███       [0m| 61/200 [00:34<01:19,
  1.76it/s]
evaluating Epoch:  31%|[32m███       [0m| 62/200 [00:35<01:18,
  1.75it/s]
evaluating Epoch:  32%|[32m███▏      [0m| 63/200 [00:35<01:18,
  1.74it/s]
evaluating Epoch:  32%|[32m███▏      [0m| 64/200 [00:36<01:17,
  1.75it/s]
evaluating Epoch:  32%|[32m███▎      [0m| 65/200 [00:36<01:16,
  1.76it/s]
evaluating Epoch:  33%|[32m███▎      [0m| 66/200 [00:37<01:15,
  1.77it/s]
evaluating Epoch:  34%|[32m███▎      [0m| 67/200 [00:38<01:15,
  1.77it/s]
evaluating Epoch:  34%|[32m███▍      [0m| 68/200 [00:38<01:14,
  1.77it/s]
evaluating Epoch:  34%|[32m███▍      [0m| 69/200 [00:39<01:13,
  1.79it/s]
evaluating Epoch:  35%|[32m███▌      [0m| 70/200 [00:39<01:11,
  1.81it/s]
evaluating Epoch:  36%|[32m███▌      [0m| 71/200 [00:40<01:10,
  1.82it/s]
evaluating Epoch:  36%|[32m███▌      [0m| 72/200 [00:40<01:11,
  1.80it/s]
evaluating Epoch:  36%|[32m███▋      [0m| 73/200 [00:41<01:10,
  1.81it/s]
evaluating Epoch:  37%|[32m███▋      [0m| 74/200 [00:41<01:10,
  1.79it/s]
evaluating Epoch:  38%|[32m███▊      [0m| 75/200 [00:42<01:10,
  1.77it/s]
evaluating Epoch:  38%|[32m███▊      [0m| 76/200 [00:43<01:10,
  1.77it/s]
evaluating Epoch:  38%|[32m███▊      [0m| 77/200 [00:43<01:08,
  1.79it/s]
evaluating Epoch:  39%|[32m███▉      [0m| 78/200 [00:44<01:07,
  1.80it/s]
evaluating Epoch:  40%|[32m███▉      [0m| 79/200 [00:44<01:07,
  1.78it/s]
evaluating Epoch:  40%|[32m████      [0m| 80/200 [00:45<01:06,
  1.80it/s]
evaluating Epoch:  40%|[32m████      [0m| 81/200 [00:45<01:06,
  1.78it/s]
evaluating Epoch:  41%|[32m████      [0m| 82/200 [00:46<01:05,
  1.80it/s]
evaluating Epoch:  42%|[32m████▏     [0m| 83/200 [00:47<01:05,
  1.77it/s]
evaluating Epoch:  42%|[32m████▏     [0m| 84/200 [00:47<01:05,
  1.77it/s]
evaluating Epoch:  42%|[32m████▎     [0m| 85/200 [00:48<01:05,
  1.77it/s]
evaluating Epoch:  43%|[32m████▎     [0m| 86/200 [00:48<01:04,
  1.78it/s]
evaluating Epoch:  44%|[32m████▎     [0m| 87/200 [00:49<01:03,
  1.78it/s]
evaluating Epoch:  44%|[32m████▍     [0m| 88/200 [00:49<01:03,
  1.78it/s]
evaluating Epoch:  44%|[32m████▍     [0m| 89/200 [00:50<01:02,
  1.79it/s]
evaluating Epoch:  45%|[32m████▌     [0m| 90/200 [00:50<01:01,
  1.78it/s]
evaluating Epoch:  46%|[32m████▌     [0m| 91/200 [00:51<01:01,
  1.76it/s]
evaluating Epoch:  46%|[32m████▌     [0m| 92/200 [00:52<01:01,
  1.75it/s]
evaluating Epoch:  46%|[32m████▋     [0m| 93/200 [00:52<01:01,
  1.74it/s]
evaluating Epoch:  47%|[32m████▋     [0m| 94/200 [00:53<01:00,
  1.75it/s]
evaluating Epoch:  48%|[32m████▊     [0m| 95/200 [00:53<00:59,
  1.77it/s]
evaluating Epoch:  48%|[32m████▊     [0m| 96/200 [00:54<00:58,
  1.78it/s]
evaluating Epoch:  48%|[32m████▊     [0m| 97/200 [00:54<00:58,
  1.76it/s]
evaluating Epoch:  49%|[32m████▉     [0m| 98/200 [00:55<00:58,
  1.74it/s]
evaluating Epoch:  50%|[32m████▉     [0m| 99/200 [00:56<00:58,
  1.74it/s]
evaluating Epoch:  50%|[32m█████     [0m| 100/200 [00:56<00:56,
  1.76it/s]
evaluating Epoch:  50%|[32m█████     [0m| 101/200 [00:57<00:55,
  1.78it/s]
evaluating Epoch:  51%|[32m█████     [0m| 102/200 [00:57<00:55,
  1.77it/s]
evaluating Epoch:  52%|[32m█████▏    [0m| 103/200 [00:58<00:55,
  1.75it/s]
evaluating Epoch:  52%|[32m█████▏    [0m| 104/200 [00:58<00:54,
  1.75it/s]
evaluating Epoch:  52%|[32m█████▎    [0m| 105/200 [00:59<00:54,
  1.75it/s]
evaluating Epoch:  53%|[32m█████▎    [0m| 106/200 [01:00<00:54,
  1.74it/s]
evaluating Epoch:  54%|[32m█████▎    [0m| 107/200 [01:00<00:52,
  1.76it/s]
evaluating Epoch:  54%|[32m█████▍    [0m| 108/200 [01:01<00:52,
  1.75it/s]
evaluating Epoch:  55%|[32m█████▍    [0m| 109/200 [01:01<00:52,
  1.74it/s]
evaluating Epoch:  55%|[32m█████▌    [0m| 110/200 [01:02<00:50,
  1.77it/s]
evaluating Epoch:  56%|[32m█████▌    [0m| 111/200 [01:02<00:50,
  1.75it/s]
evaluating Epoch:  56%|[32m█████▌    [0m| 112/200 [01:03<00:50,
  1.75it/s]
evaluating Epoch:  56%|[32m█████▋    [0m| 113/200 [01:04<00:49,
  1.77it/s]
evaluating Epoch:  57%|[32m█████▋    [0m| 114/200 [01:04<00:48,
  1.77it/s]
evaluating Epoch:  57%|[32m█████▊    [0m| 115/200 [01:05<00:47,
  1.78it/s]
evaluating Epoch:  58%|[32m█████▊    [0m| 116/200 [01:05<00:47,
  1.77it/s]
evaluating Epoch:  58%|[32m█████▊    [0m| 117/200 [01:06<00:47,
  1.75it/s]
evaluating Epoch:  59%|[32m█████▉    [0m| 118/200 [01:06<00:46,
  1.77it/s]
evaluating Epoch:  60%|[32m█████▉    [0m| 119/200 [01:07<00:45,
  1.77it/s]
evaluating Epoch:  60%|[32m██████    [0m| 120/200 [01:08<00:45,
  1.75it/s]
evaluating Epoch:  60%|[32m██████    [0m| 121/200 [01:08<00:45,
  1.74it/s]
evaluating Epoch:  61%|[32m██████    [0m| 122/200 [01:09<00:45,
  1.73it/s]
evaluating Epoch:  62%|[32m██████▏   [0m| 123/200 [01:09<00:43,
  1.76it/s]
evaluating Epoch:  62%|[32m██████▏   [0m| 124/200 [01:10<00:43,
  1.76it/s]
evaluating Epoch:  62%|[32m██████▎   [0m| 125/200 [01:10<00:42,
  1.75it/s]
evaluating Epoch:  63%|[32m██████▎   [0m| 126/200 [01:11<00:42,
  1.73it/s]
evaluating Epoch:  64%|[32m██████▎   [0m| 127/200 [01:12<00:42,
  1.72it/s]
evaluating Epoch:  64%|[32m██████▍   [0m| 128/200 [01:12<00:41,
  1.72it/s]
evaluating Epoch:  64%|[32m██████▍   [0m| 129/200 [01:13<00:40,
  1.73it/s]
evaluating Epoch:  65%|[32m██████▌   [0m| 130/200 [01:13<00:40,
  1.74it/s]
evaluating Epoch:  66%|[32m██████▌   [0m| 131/200 [01:14<00:39,
  1.75it/s]
evaluating Epoch:  66%|[32m██████▌   [0m| 132/200 [01:14<00:38,
  1.74it/s]
evaluating Epoch:  66%|[32m██████▋   [0m| 133/200 [01:15<00:37,
  1.77it/s]
evaluating Epoch:  67%|[32m██████▋   [0m| 134/200 [01:16<00:37,
  1.75it/s]
evaluating Epoch:  68%|[32m██████▊   [0m| 135/200 [01:16<00:36,
  1.76it/s]
evaluating Epoch:  68%|[32m██████▊   [0m| 136/200 [01:17<00:36,
  1.76it/s]
evaluating Epoch:  68%|[32m██████▊   [0m| 137/200 [01:17<00:36,
  1.75it/s]
evaluating Epoch:  69%|[32m██████▉   [0m| 138/200 [01:18<00:35,
  1.75it/s]
evaluating Epoch:  70%|[32m██████▉   [0m| 139/200 [01:18<00:35,
  1.74it/s]
evaluating Epoch:  70%|[32m███████   [0m| 140/200 [01:19<00:34,
  1.76it/s]
evaluating Epoch:  70%|[32m███████   [0m| 141/200 [01:20<00:33,
  1.75it/s]
evaluating Epoch:  71%|[32m███████   [0m| 142/200 [01:20<00:33,
  1.74it/s]
evaluating Epoch:  72%|[32m███████▏  [0m| 143/200 [01:21<00:32,
  1.74it/s]
evaluating Epoch:  72%|[32m███████▏  [0m| 144/200 [01:21<00:32,
  1.74it/s]
evaluating Epoch:  72%|[32m███████▎  [0m| 145/200 [01:22<00:31,
  1.76it/s]
evaluating Epoch:  73%|[32m███████▎  [0m| 146/200 [01:22<00:30,
  1.75it/s]
evaluating Epoch:  74%|[32m███████▎  [0m| 147/200 [01:23<00:30,
  1.76it/s]
evaluating Epoch:  74%|[32m███████▍  [0m| 148/200 [01:24<00:29,
  1.76it/s]
evaluating Epoch:  74%|[32m███████▍  [0m| 149/200 [01:24<00:28,
  1.78it/s]
evaluating Epoch:  75%|[32m███████▌  [0m| 150/200 [01:25<00:28,
  1.78it/s]
evaluating Epoch:  76%|[32m███████▌  [0m| 151/200 [01:25<00:27,
  1.78it/s]
evaluating Epoch:  76%|[32m███████▌  [0m| 152/200 [01:26<00:26,
  1.80it/s]
evaluating Epoch:  76%|[32m███████▋  [0m| 153/200 [01:26<00:26,
  1.79it/s]
evaluating Epoch:  77%|[32m███████▋  [0m| 154/200 [01:27<00:25,
  1.77it/s]
evaluating Epoch:  78%|[32m███████▊  [0m| 155/200 [01:27<00:25,
  1.77it/s]
evaluating Epoch:  78%|[32m███████▊  [0m| 156/200 [01:28<00:24,
  1.76it/s]
evaluating Epoch:  78%|[32m███████▊  [0m| 157/200 [01:29<00:24,
  1.76it/s]
evaluating Epoch:  79%|[32m███████▉  [0m| 158/200 [01:29<00:23,
  1.77it/s]
evaluating Epoch:  80%|[32m███████▉  [0m| 159/200 [01:30<00:22,
  1.79it/s]
evaluating Epoch:  80%|[32m████████  [0m| 160/200 [01:30<00:22,
  1.78it/s]
evaluating Epoch:  80%|[32m████████  [0m| 161/200 [01:31<00:22,
  1.77it/s]
evaluating Epoch:  81%|[32m████████  [0m| 162/200 [01:31<00:21,
  1.79it/s]
evaluating Epoch:  82%|[32m████████▏ [0m| 163/200 [01:32<00:20,
  1.79it/s]
evaluating Epoch:  82%|[32m████████▏ [0m| 164/200 [01:33<00:20,
  1.79it/s]
evaluating Epoch:  82%|[32m████████▎ [0m| 165/200 [01:33<00:19,
  1.79it/s]
evaluating Epoch:  83%|[32m████████▎ [0m| 166/200 [01:34<00:18,
  1.80it/s]
evaluating Epoch:  84%|[32m████████▎ [0m| 167/200 [01:34<00:18,
  1.79it/s]
evaluating Epoch:  84%|[32m████████▍ [0m| 168/200 [01:35<00:18,
  1.78it/s]
evaluating Epoch:  84%|[32m████████▍ [0m| 169/200 [01:35<00:17,
  1.77it/s]
evaluating Epoch:  85%|[32m████████▌ [0m| 170/200 [01:36<00:16,
  1.79it/s]
evaluating Epoch:  86%|[32m████████▌ [0m| 171/200 [01:36<00:16,
  1.78it/s]
evaluating Epoch:  86%|[32m████████▌ [0m| 172/200 [01:37<00:15,
  1.78it/s]
evaluating Epoch:  86%|[32m████████▋ [0m| 173/200 [01:38<00:15,
  1.79it/s]
evaluating Epoch:  87%|[32m████████▋ [0m| 174/200 [01:38<00:14,
  1.78it/s]
evaluating Epoch:  88%|[32m████████▊ [0m| 175/200 [01:39<00:14,
  1.78it/s]
evaluating Epoch:  88%|[32m████████▊ [0m| 176/200 [01:39<00:13,
  1.80it/s]
evaluating Epoch:  88%|[32m████████▊ [0m| 177/200 [01:40<00:12,
  1.80it/s]
evaluating Epoch:  89%|[32m████████▉ [0m| 178/200 [01:40<00:12,
  1.81it/s]
evaluating Epoch:  90%|[32m████████▉ [0m| 179/200 [01:41<00:11,
  1.80it/s]
evaluating Epoch:  90%|[32m█████████ [0m| 180/200 [01:41<00:11,
  1.80it/s]
evaluating Epoch:  90%|[32m█████████ [0m| 181/200 [01:42<00:10,
  1.80it/s]
evaluating Epoch:  91%|[32m█████████ [0m| 182/200 [01:43<00:10,
  1.79it/s]
evaluating Epoch:  92%|[32m█████████▏[0m| 183/200 [01:43<00:09,
  1.81it/s]
evaluating Epoch:  92%|[32m█████████▏[0m| 184/200 [01:44<00:08,
  1.78it/s]
evaluating Epoch:  92%|[32m█████████▎[0m| 185/200 [01:44<00:08,
  1.79it/s]
evaluating Epoch:  93%|[32m█████████▎[0m| 186/200 [01:45<00:07,
  1.79it/s]
evaluating Epoch:  94%|[32m█████████▎[0m| 187/200 [01:45<00:07,
  1.78it/s]
evaluating Epoch:  94%|[32m█████████▍[0m| 188/200 [01:46<00:06,
  1.80it/s]
evaluating Epoch:  94%|[32m█████████▍[0m| 189/200 [01:46<00:06,
  1.80it/s]
evaluating Epoch:  95%|[32m█████████▌[0m| 190/200 [01:47<00:05,
  1.81it/s]
evaluating Epoch:  96%|[32m█████████▌[0m| 191/200 [01:48<00:04,
  1.81it/s]
evaluating Epoch:  96%|[32m█████████▌[0m| 192/200 [01:48<00:04,
  1.79it/s]
evaluating Epoch:  96%|[32m█████████▋[0m| 193/200 [01:49<00:03,
  1.78it/s]
evaluating Epoch:  97%|[32m█████████▋[0m| 194/200 [01:49<00:03,
  1.77it/s]
evaluating Epoch:  98%|[32m█████████▊[0m| 195/200 [01:50<00:02,
  1.78it/s]
evaluating Epoch:  98%|[32m█████████▊[0m| 196/200 [01:50<00:02,
  1.77it/s]
evaluating Epoch:  98%|[32m█████████▊[0m| 197/200 [01:51<00:01,
  1.76it/s]
evaluating Epoch:  99%|[32m█████████▉[0m| 198/200 [01:52<00:01,
  1.75it/s]
evaluating Epoch: 100%|[32m█████████▉[0m| 199/200 [01:52<00:00,
  1.74it/s]
evaluating Epoch: 100%|[32m██████████[0m| 200/200 [01:53<00:00,
  1.75it/s]
evaluating Epoch: 100%|[32m██████████[0m| 200/200 [01:53<00:00,
  1.76it/s]
 eval_ppl=tensor(1.0906,
 device='cuda:0') eval_epoch_loss=tensor(0.0868,
 device='cuda:0')
we are about to save the PEFT modules
PEFT modules are saved in ./output_dir_whole_dataset directory
best eval loss on epoch 1 is 0.08676084131002426
Epoch 2: train_perplexity=1.0752,
 train_epoch_loss=0.0725,
 epcoh time 3094.788047331851s

Training Epoch: 2:   0%|[34m          [0m| 0/249 [00:00<?,
 ?it/s]
Training Epoch: 1/3,
 step 248/249 completed (loss: 0.005063027609139681): : 30876it [53:28,
  9.62it/s]

Training Epoch: 2:   0%|[34m          [0m| 0/249 [00:12<?,
 ?it/s]
Training Epoch: 2/3,
 step 0/249 completed (loss: 0.06119762361049652):   0%|[34m          [0m| 0/249 [00:12<?,
 ?it/s]
Training Epoch: 2/3,
 step 0/249 completed (loss: 0.06119762361049652):   0%|[34m          [0m| 1/249 [00:24<51:19,
 12.42s/it]
Training Epoch: 2/3,
 step 1/249 completed (loss: 0.11625608056783676):   0%|[34m          [0m| 1/249 [00:24<51:19,
 12.42s/it]
Training Epoch: 2/3,
 step 1/249 completed (loss: 0.11625608056783676):   1%|[34m          [0m| 3/249 [00:37<32:04,
  7.82s/it]
Training Epoch: 2/3,
 step 2/249 completed (loss: 0.056413184851408005):   1%|[34m          [0m| 3/249 [00:37<32:04,
  7.82s/it]
Training Epoch: 2/3,
 step 2/249 completed (loss: 0.056413184851408005):   2%|[34m▏         [0m| 6/249 [00:49<22:32,
  5.57s/it]
Training Epoch: 2/3,
 step 3/249 completed (loss: 0.055902693420648575):   2%|[34m▏         [0m| 6/249 [00:49<22:32,
  5.57s/it]
Training Epoch: 2/3,
 step 3/249 completed (loss: 0.055902693420648575):   4%|[34m▍         [0m| 10/249 [01:02<16:53,
  4.24s/it]
Training Epoch: 2/3,
 step 4/249 completed (loss: 0.09592495858669281):   4%|[34m▍         [0m| 10/249 [01:02<16:53,
  4.24s/it] 
Training Epoch: 2/3,
 step 4/249 completed (loss: 0.09592495858669281):   6%|[34m▌         [0m| 15/249 [01:14<13:10,
  3.38s/it]
Training Epoch: 2/3,
 step 5/249 completed (loss: 0.10031989961862564):   6%|[34m▌         [0m| 15/249 [01:14<13:10,
  3.38s/it]
Training Epoch: 2/3,
 step 5/249 completed (loss: 0.10031989961862564):   8%|[34m▊         [0m| 21/249 [01:26<10:33,
  2.78s/it]
Training Epoch: 2/3,
 step 6/249 completed (loss: 0.07067185640335083):   8%|[34m▊         [0m| 21/249 [01:27<10:33,
  2.78s/it]
Training Epoch: 2/3,
 step 6/249 completed (loss: 0.07067185640335083):  11%|[34m█         [0m| 28/249 [01:39<08:38,
  2.35s/it]
Training Epoch: 2/3,
 step 7/249 completed (loss: 0.01660139113664627):  11%|[34m█         [0m| 28/249 [01:39<08:38,
  2.35s/it]
Training Epoch: 2/3,
 step 7/249 completed (loss: 0.01660139113664627):  14%|[34m█▍        [0m| 36/249 [01:51<07:09,
  2.02s/it]
Training Epoch: 2/3,
 step 8/249 completed (loss: 0.11340731382369995):  14%|[34m█▍        [0m| 36/249 [01:51<07:09,
  2.02s/it]
Training Epoch: 2/3,
 step 8/249 completed (loss: 0.11340731382369995):  18%|[34m█▊        [0m| 45/249 [02:04<05:59,
  1.76s/it]
Training Epoch: 2/3,
 step 9/249 completed (loss: 0.03133572265505791):  18%|[34m█▊        [0m| 45/249 [02:04<05:59,
  1.76s/it]
Training Epoch: 2/3,
 step 9/249 completed (loss: 0.03133572265505791):  22%|[34m██▏       [0m| 55/249 [02:16<05:02,
  1.56s/it]
Training Epoch: 2/3,
 step 10/249 completed (loss: 0.044609274715185165):  22%|[34m██▏       [0m| 55/249 [02:16<05:02,
  1.56s/it]
Training Epoch: 2/3,
 step 10/249 completed (loss: 0.044609274715185165):  27%|[34m██▋       [0m| 66/249 [02:29<04:15,
  1.40s/it]
Training Epoch: 2/3,
 step 11/249 completed (loss: 0.03840945288538933):  27%|[34m██▋       [0m| 66/249 [02:29<04:15,
  1.40s/it] 
Training Epoch: 2/3,
 step 11/249 completed (loss: 0.03840945288538933):  31%|[34m███▏      [0m| 78/249 [02:41<03:35,
  1.26s/it]
Training Epoch: 2/3,
 step 12/249 completed (loss: 0.07408442348241806):  31%|[34m███▏      [0m| 78/249 [02:41<03:35,
  1.26s/it]
Training Epoch: 2/3,
 step 12/249 completed (loss: 0.07408442348241806):  37%|[34m███▋      [0m| 91/249 [02:53<03:01,
  1.15s/it]
Training Epoch: 2/3,
 step 13/249 completed (loss: 0.0615663081407547):  37%|[34m███▋      [0m| 91/249 [02:54<03:01,
  1.15s/it] 
Training Epoch: 2/3,
 step 13/249 completed (loss: 0.0615663081407547):  42%|[34m████▏     [0m| 105/249 [03:06<02:32,
  1.06s/it]
Training Epoch: 2/3,
 step 14/249 completed (loss: 0.0771532878279686):  42%|[34m████▏     [0m| 105/249 [03:06<02:32,
  1.06s/it]
Training Epoch: 2/3,
 step 14/249 completed (loss: 0.0771532878279686):  48%|[34m████▊     [0m| 120/249 [03:18<02:05,
  1.02it/s]
Training Epoch: 2/3,
 step 15/249 completed (loss: 0.05835830047726631):  48%|[34m████▊     [0m| 120/249 [03:18<02:05,
  1.02it/s]
Training Epoch: 2/3,
 step 15/249 completed (loss: 0.05835830047726631):  55%|[34m█████▍    [0m| 136/249 [03:31<01:42,
  1.10it/s]
Training Epoch: 2/3,
 step 16/249 completed (loss: 0.07759235799312592):  55%|[34m█████▍    [0m| 136/249 [03:31<01:42,
  1.10it/s]
Training Epoch: 2/3,
 step 16/249 completed (loss: 0.07759235799312592):  61%|[34m██████▏   [0m| 153/249 [03:43<01:21,
  1.18it/s]
Training Epoch: 2/3,
 step 17/249 completed (loss: 0.04230761155486107):  61%|[34m██████▏   [0m| 153/249 [03:43<01:21,
  1.18it/s]
Training Epoch: 2/3,
 step 17/249 completed (loss: 0.04230761155486107):  69%|[34m██████▊   [0m| 171/249 [03:56<01:01,
  1.26it/s]
Training Epoch: 2/3,
 step 18/249 completed (loss: 0.01057218387722969):  69%|[34m██████▊   [0m| 171/249 [03:56<01:01,
  1.26it/s]
Training Epoch: 2/3,
 step 18/249 completed (loss: 0.01057218387722969):  76%|[34m███████▋  [0m| 190/249 [04:08<00:43,
  1.34it/s]
Training Epoch: 2/3,
 step 19/249 completed (loss: 0.02900863066315651):  76%|[34m███████▋  [0m| 190/249 [04:08<00:43,
  1.34it/s]
Training Epoch: 2/3,
 step 19/249 completed (loss: 0.02900863066315651):  84%|[34m████████▍ [0m| 210/249 [04:20<00:27,
  1.42it/s]
Training Epoch: 2/3,
 step 20/249 completed (loss: 0.14312082529067993):  84%|[34m████████▍ [0m| 210/249 [04:21<00:27,
  1.42it/s]
Training Epoch: 2/3,
 step 20/249 completed (loss: 0.14312082529067993):  93%|[34m█████████▎[0m| 231/249 [04:33<00:11,
  1.50it/s]
Training Epoch: 2/3,
 step 21/249 completed (loss: 0.003403738373890519):  93%|[34m█████████▎[0m| 231/249 [04:33<00:11,
  1.50it/s]
Training Epoch: 2/3,
 step 21/249 completed (loss: 0.003403738373890519): : 253it [04:45,
  1.58it/s]                       
Training Epoch: 2/3,
 step 22/249 completed (loss: 0.03332258015871048): : 253it [04:45,
  1.58it/s] 
Training Epoch: 2/3,
 step 22/249 completed (loss: 0.03332258015871048): : 276it [04:58,
  1.66it/s]
Training Epoch: 2/3,
 step 23/249 completed (loss: 0.034766148775815964): : 276it [04:58,
  1.66it/s]
Training Epoch: 2/3,
 step 23/249 completed (loss: 0.034766148775815964): : 300it [05:10,
  1.74it/s]
Training Epoch: 2/3,
 step 24/249 completed (loss: 0.05937938019633293): : 300it [05:10,
  1.74it/s] 
Training Epoch: 2/3,
 step 24/249 completed (loss: 0.05937938019633293): : 325it [05:23,
  1.82it/s]
Training Epoch: 2/3,
 step 25/249 completed (loss: 0.030226197093725204): : 325it [05:23,
  1.82it/s]
Training Epoch: 2/3,
 step 25/249 completed (loss: 0.030226197093725204): : 351it [05:35,
  1.90it/s]
Training Epoch: 2/3,
 step 26/249 completed (loss: 0.016475340351462364): : 351it [05:35,
  1.90it/s]
Training Epoch: 2/3,
 step 26/249 completed (loss: 0.016475340351462364): : 378it [05:47,
  1.99it/s]
Training Epoch: 2/3,
 step 27/249 completed (loss: 0.04466924071311951): : 378it [05:48,
  1.99it/s] 
Training Epoch: 2/3,
 step 27/249 completed (loss: 0.04466924071311951): : 406it [06:00,
  2.07it/s]
Training Epoch: 2/3,
 step 28/249 completed (loss: 0.051577016711235046): : 406it [06:00,
  2.07it/s]
Training Epoch: 2/3,
 step 28/249 completed (loss: 0.051577016711235046): : 435it [06:12,
  2.15it/s]
Training Epoch: 2/3,
 step 29/249 completed (loss: 0.048793233931064606): : 435it [06:12,
  2.15it/s]
Training Epoch: 2/3,
 step 29/249 completed (loss: 0.048793233931064606): : 465it [06:25,
  2.22it/s]
Training Epoch: 2/3,
 step 30/249 completed (loss: 0.07261583209037781): : 465it [06:25,
  2.22it/s] 
Training Epoch: 2/3,
 step 30/249 completed (loss: 0.07261583209037781): : 496it [06:37,
  2.30it/s]
Training Epoch: 2/3,
 step 31/249 completed (loss: 0.07848160713911057): : 496it [06:37,
  2.30it/s]
Training Epoch: 2/3,
 step 31/249 completed (loss: 0.07848160713911057): : 528it [06:50,
  2.39it/s]
Training Epoch: 2/3,
 step 32/249 completed (loss: 0.006224038079380989): : 528it [06:50,
  2.39it/s]
Training Epoch: 2/3,
 step 32/249 completed (loss: 0.006224038079380989): : 561it [07:02,
  2.47it/s]
Training Epoch: 2/3,
 step 33/249 completed (loss: 0.12084489315748215): : 561it [07:02,
  2.47it/s] 
Training Epoch: 2/3,
 step 33/249 completed (loss: 0.12084489315748215): : 595it [07:14,
  2.55it/s]
Training Epoch: 2/3,
 step 34/249 completed (loss: 0.0008405948174186051): : 595it [07:15,
  2.55it/s]
Training Epoch: 2/3,
 step 34/249 completed (loss: 0.0008405948174186051): : 630it [07:27,
  2.63it/s]
Training Epoch: 2/3,
 step 35/249 completed (loss: 0.022756721824407578): : 630it [07:27,
  2.63it/s] 
Training Epoch: 2/3,
 step 35/249 completed (loss: 0.022756721824407578): : 666it [07:39,
  2.71it/s]
Training Epoch: 2/3,
 step 36/249 completed (loss: 0.05295862630009651): : 666it [07:39,
  2.71it/s] 
Training Epoch: 2/3,
 step 36/249 completed (loss: 0.05295862630009651): : 703it [07:52,
  2.79it/s]
Training Epoch: 2/3,
 step 37/249 completed (loss: 0.015907609835267067): : 703it [07:52,
  2.79it/s]
Training Epoch: 2/3,
 step 37/249 completed (loss: 0.015907609835267067): : 741it [08:04,
  2.87it/s]
Training Epoch: 2/3,
 step 38/249 completed (loss: 0.04415249451994896): : 741it [08:04,
  2.87it/s] 
Training Epoch: 2/3,
 step 38/249 completed (loss: 0.04415249451994896): : 780it [08:17,
  2.95it/s]
Training Epoch: 2/3,
 step 39/249 completed (loss: 0.062269218266010284): : 780it [08:17,
  2.95it/s]
Training Epoch: 2/3,
 step 39/249 completed (loss: 0.062269218266010284): : 820it [08:29,
  3.03it/s]
Training Epoch: 2/3,
 step 40/249 completed (loss: 0.0019829219672828913): : 820it [08:29,
  3.03it/s]
Training Epoch: 2/3,
 step 40/249 completed (loss: 0.0019829219672828913): : 861it [08:41,
  3.11it/s]
Training Epoch: 2/3,
 step 41/249 completed (loss: 0.022184228524565697): : 861it [08:42,
  3.11it/s] 
Training Epoch: 2/3,
 step 41/249 completed (loss: 0.022184228524565697): : 903it [08:54,
  3.19it/s]
Training Epoch: 2/3,
 step 42/249 completed (loss: 0.022081874310970306): : 903it [08:54,
  3.19it/s]
Training Epoch: 2/3,
 step 42/249 completed (loss: 0.022081874310970306): : 946it [09:06,
  3.27it/s]
Training Epoch: 2/3,
 step 43/249 completed (loss: 0.12201903015375137): : 946it [09:06,
  3.27it/s] 
Training Epoch: 2/3,
 step 43/249 completed (loss: 0.12201903015375137): : 990it [09:19,
  3.35it/s]
Training Epoch: 2/3,
 step 44/249 completed (loss: 0.21864092350006104): : 990it [09:19,
  3.35it/s]
Training Epoch: 2/3,
 step 44/249 completed (loss: 0.21864092350006104): : 1035it [09:31,
  3.43it/s]
Training Epoch: 2/3,
 step 45/249 completed (loss: 0.003511563641950488): : 1035it [09:31,
  3.43it/s]
Training Epoch: 2/3,
 step 45/249 completed (loss: 0.003511563641950488): : 1081it [09:44,
  3.51it/s]
Training Epoch: 2/3,
 step 46/249 completed (loss: 0.03656700253486633): : 1081it [09:44,
  3.51it/s] 
Training Epoch: 2/3,
 step 46/249 completed (loss: 0.03656700253486633): : 1128it [09:56,
  3.59it/s]
Training Epoch: 2/3,
 step 47/249 completed (loss: 0.11827158182859421): : 1128it [09:56,
  3.59it/s]
Training Epoch: 2/3,
 step 47/249 completed (loss: 0.11827158182859421): : 1176it [10:08,
  3.68it/s]
Training Epoch: 2/3,
 step 48/249 completed (loss: 0.06193530559539795): : 1176it [10:09,
  3.68it/s]
Training Epoch: 2/3,
 step 48/249 completed (loss: 0.06193530559539795): : 1225it [10:21,
  3.76it/s]
Training Epoch: 2/3,
 step 49/249 completed (loss: 0.03648001328110695): : 1225it [10:21,
  3.76it/s]
Training Epoch: 2/3,
 step 49/249 completed (loss: 0.03648001328110695): : 1275it [10:33,
  3.84it/s]
Training Epoch: 2/3,
 step 50/249 completed (loss: 0.1382085531949997): : 1275it [10:33,
  3.84it/s] 
Training Epoch: 2/3,
 step 50/249 completed (loss: 0.1382085531949997): : 1326it [10:46,
  3.92it/s]
Training Epoch: 2/3,
 step 51/249 completed (loss: 0.09129103273153305): : 1326it [10:46,
  3.92it/s]
Training Epoch: 2/3,
 step 51/249 completed (loss: 0.09129103273153305): : 1378it [10:58,
  4.00it/s]
Training Epoch: 2/3,
 step 52/249 completed (loss: 0.04474841058254242): : 1378it [10:58,
  4.00it/s]
Training Epoch: 2/3,
 step 52/249 completed (loss: 0.04474841058254242): : 1431it [11:11,
  4.08it/s]
Training Epoch: 2/3,
 step 53/249 completed (loss: 0.017370078712701797): : 1431it [11:11,
  4.08it/s]
Training Epoch: 2/3,
 step 53/249 completed (loss: 0.017370078712701797): : 1485it [11:23,
  4.16it/s]
Training Epoch: 2/3,
 step 54/249 completed (loss: 0.08113802969455719): : 1485it [11:23,
  4.16it/s] 
Training Epoch: 2/3,
 step 54/249 completed (loss: 0.08113802969455719): : 1540it [11:35,
  4.24it/s]
Training Epoch: 2/3,
 step 55/249 completed (loss: 0.07018035650253296): : 1540it [11:36,
  4.24it/s]
Training Epoch: 2/3,
 step 55/249 completed (loss: 0.07018035650253296): : 1596it [11:48,
  4.32it/s]
Training Epoch: 2/3,
 step 56/249 completed (loss: 0.023877711966633797): : 1596it [11:48,
  4.32it/s]
Training Epoch: 2/3,
 step 56/249 completed (loss: 0.023877711966633797): : 1653it [12:00,
  4.40it/s]
Training Epoch: 2/3,
 step 57/249 completed (loss: 0.024790143594145775): : 1653it [12:00,
  4.40it/s]
Training Epoch: 2/3,
 step 57/249 completed (loss: 0.024790143594145775): : 1711it [12:13,
  4.48it/s]
Training Epoch: 2/3,
 step 58/249 completed (loss: 0.041551318019628525): : 1711it [12:13,
  4.48it/s]
Training Epoch: 2/3,
 step 58/249 completed (loss: 0.041551318019628525): : 1770it [12:25,
  4.56it/s]
Training Epoch: 2/3,
 step 59/249 completed (loss: 0.030884692445397377): : 1770it [12:25,
  4.56it/s]
Training Epoch: 2/3,
 step 59/249 completed (loss: 0.030884692445397377): : 1830it [12:37,
  4.64it/s]
Training Epoch: 2/3,
 step 60/249 completed (loss: 0.01733430288732052): : 1830it [12:38,
  4.64it/s] 
Training Epoch: 2/3,
 step 60/249 completed (loss: 0.01733430288732052): : 1891it [12:50,
  4.72it/s]
Training Epoch: 2/3,
 step 61/249 completed (loss: 0.0961872860789299): : 1891it [12:50,
  4.72it/s] 
Training Epoch: 2/3,
 step 61/249 completed (loss: 0.0961872860789299): : 1953it [13:02,
  4.80it/s]
Training Epoch: 2/3,
 step 62/249 completed (loss: 0.02978786826133728): : 1953it [13:03,
  4.80it/s]
Training Epoch: 2/3,
 step 62/249 completed (loss: 0.02978786826133728): : 2016it [13:15,
  4.88it/s]
Training Epoch: 2/3,
 step 63/249 completed (loss: 0.09662491828203201): : 2016it [13:15,
  4.88it/s]
Training Epoch: 2/3,
 step 63/249 completed (loss: 0.09662491828203201): : 2080it [13:27,
  4.96it/s]
Training Epoch: 2/3,
 step 64/249 completed (loss: 0.0208165030926466): : 2080it [13:27,
  4.96it/s] 
Training Epoch: 2/3,
 step 64/249 completed (loss: 0.0208165030926466): : 2145it [13:40,
  5.05it/s]
Training Epoch: 2/3,
 step 65/249 completed (loss: 0.029815178364515305): : 2145it [13:40,
  5.05it/s]
Training Epoch: 2/3,
 step 65/249 completed (loss: 0.029815178364515305): : 2211it [13:52,
  5.13it/s]
Training Epoch: 2/3,
 step 66/249 completed (loss: 0.05446150526404381): : 2211it [13:52,
  5.13it/s] 
Training Epoch: 2/3,
 step 66/249 completed (loss: 0.05446150526404381): : 2278it [14:04,
  5.20it/s]
Training Epoch: 2/3,
 step 67/249 completed (loss: 0.02114158682525158): : 2278it [14:05,
  5.20it/s]
Training Epoch: 2/3,
 step 67/249 completed (loss: 0.02114158682525158): : 2346it [14:17,
  5.28it/s]
Training Epoch: 2/3,
 step 68/249 completed (loss: 0.073371522128582): : 2346it [14:17,
  5.28it/s]  
Training Epoch: 2/3,
 step 68/249 completed (loss: 0.073371522128582): : 2415it [14:29,
  5.36it/s]
Training Epoch: 2/3,
 step 69/249 completed (loss: 0.06669675558805466): : 2415it [14:30,
  5.36it/s]
Training Epoch: 2/3,
 step 69/249 completed (loss: 0.06669675558805466): : 2485it [14:42,
  5.45it/s]
Training Epoch: 2/3,
 step 70/249 completed (loss: 0.0644354447722435): : 2485it [14:42,
  5.45it/s] 
Training Epoch: 2/3,
 step 70/249 completed (loss: 0.0644354447722435): : 2556it [14:54,
  5.52it/s]
Training Epoch: 2/3,
 step 71/249 completed (loss: 0.03395243361592293): : 2556it [14:54,
  5.52it/s]
Training Epoch: 2/3,
 step 71/249 completed (loss: 0.03395243361592293): : 2628it [15:07,
  5.60it/s]
Training Epoch: 2/3,
 step 72/249 completed (loss: 0.062304042279720306): : 2628it [15:07,
  5.60it/s]
Training Epoch: 2/3,
 step 72/249 completed (loss: 0.062304042279720306): : 2701it [15:19,
  5.68it/s]
Training Epoch: 2/3,
 step 73/249 completed (loss: 0.027817560359835625): : 2701it [15:19,
  5.68it/s]
Training Epoch: 2/3,
 step 73/249 completed (loss: 0.027817560359835625): : 2775it [15:31,
  5.76it/s]
Training Epoch: 2/3,
 step 74/249 completed (loss: 0.033001501113176346): : 2775it [15:32,
  5.76it/s]
Training Epoch: 2/3,
 step 74/249 completed (loss: 0.033001501113176346): : 2850it [15:44,
  5.84it/s]
Training Epoch: 2/3,
 step 75/249 completed (loss: 0.0012992313131690025): : 2850it [15:44,
  5.84it/s]
Training Epoch: 2/3,
 step 75/249 completed (loss: 0.0012992313131690025): : 2926it [15:56,
  5.93it/s]
Training Epoch: 2/3,
 step 76/249 completed (loss: 0.022980621084570885): : 2926it [15:57,
  5.93it/s] 
Training Epoch: 2/3,
 step 76/249 completed (loss: 0.022980621084570885): : 3003it [16:09,
  6.01it/s]
Training Epoch: 2/3,
 step 77/249 completed (loss: 0.008361001498997211): : 3003it [16:09,
  6.01it/s]
Training Epoch: 2/3,
 step 77/249 completed (loss: 0.008361001498997211): : 3081it [16:21,
  6.09it/s]
Training Epoch: 2/3,
 step 78/249 completed (loss: 0.08060579746961594): : 3081it [16:21,
  6.09it/s] 
Training Epoch: 2/3,
 step 78/249 completed (loss: 0.08060579746961594): : 3160it [16:34,
  6.17it/s]
Training Epoch: 2/3,
 step 79/249 completed (loss: 0.1847410500049591): : 3160it [16:34,
  6.17it/s] 
Training Epoch: 2/3,
 step 79/249 completed (loss: 0.1847410500049591): : 3240it [16:46,
  6.25it/s]
Training Epoch: 2/3,
 step 80/249 completed (loss: 0.09404516965150833): : 3240it [16:46,
  6.25it/s]
Training Epoch: 2/3,
 step 80/249 completed (loss: 0.09404516965150833): : 3321it [16:58,
  6.33it/s]
Training Epoch: 2/3,
 step 81/249 completed (loss: 0.05212530493736267): : 3321it [16:59,
  6.33it/s]
Training Epoch: 2/3,
 step 81/249 completed (loss: 0.05212530493736267): : 3403it [17:11,
  6.41it/s]
Training Epoch: 2/3,
 step 82/249 completed (loss: 0.04711558669805527): : 3403it [17:11,
  6.41it/s]
Training Epoch: 2/3,
 step 82/249 completed (loss: 0.04711558669805527): : 3486it [17:23,
  6.49it/s]
Training Epoch: 2/3,
 step 83/249 completed (loss: 0.03127512335777283): : 3486it [17:24,
  6.49it/s]
Training Epoch: 2/3,
 step 83/249 completed (loss: 0.03127512335777283): : 3570it [17:36,
  6.57it/s]
Training Epoch: 2/3,
 step 84/249 completed (loss: 0.030038874596357346): : 3570it [17:36,
  6.57it/s]
Training Epoch: 2/3,
 step 84/249 completed (loss: 0.030038874596357346): : 3655it [17:48,
  6.65it/s]
Training Epoch: 2/3,
 step 85/249 completed (loss: 0.1256667822599411): : 3655it [17:48,
  6.65it/s]  
Training Epoch: 2/3,
 step 85/249 completed (loss: 0.1256667822599411): : 3741it [18:01,
  6.73it/s]
Training Epoch: 2/3,
 step 86/249 completed (loss: 0.03197266161441803): : 3741it [18:01,
  6.73it/s]
Training Epoch: 2/3,
 step 86/249 completed (loss: 0.03197266161441803): : 3828it [18:13,
  6.81it/s]
Training Epoch: 2/3,
 step 87/249 completed (loss: 0.07744775712490082): : 3828it [18:13,
  6.81it/s]
Training Epoch: 2/3,
 step 87/249 completed (loss: 0.07744775712490082): : 3916it [18:26,
  6.89it/s]
Training Epoch: 2/3,
 step 88/249 completed (loss: 0.021340806037187576): : 3916it [18:26,
  6.89it/s]
Training Epoch: 2/3,
 step 88/249 completed (loss: 0.021340806037187576): : 4005it [18:38,
  6.97it/s]
Training Epoch: 2/3,
 step 89/249 completed (loss: 0.07750514894723892): : 4005it [18:38,
  6.97it/s] 
Training Epoch: 2/3,
 step 89/249 completed (loss: 0.07750514894723892): : 4095it [18:50,
  7.05it/s]
Training Epoch: 2/3,
 step 90/249 completed (loss: 0.010312803089618683): : 4095it [18:51,
  7.05it/s]
Training Epoch: 2/3,
 step 90/249 completed (loss: 0.010312803089618683): : 4186it [19:03,
  7.12it/s]
Training Epoch: 2/3,
 step 91/249 completed (loss: 0.0370040200650692): : 4186it [19:03,
  7.12it/s]  
Training Epoch: 2/3,
 step 91/249 completed (loss: 0.0370040200650692): : 4278it [19:15,
  7.20it/s]
Training Epoch: 2/3,
 step 92/249 completed (loss: 0.009537499397993088): : 4278it [19:16,
  7.20it/s]
Training Epoch: 2/3,
 step 92/249 completed (loss: 0.009537499397993088): : 4371it [19:28,
  7.29it/s]
Training Epoch: 2/3,
 step 93/249 completed (loss: 0.016817517578601837): : 4371it [19:28,
  7.29it/s]
Training Epoch: 2/3,
 step 93/249 completed (loss: 0.016817517578601837): : 4465it [19:40,
  7.37it/s]
Training Epoch: 2/3,
 step 94/249 completed (loss: 0.02357347309589386): : 4465it [19:40,
  7.37it/s] 
Training Epoch: 2/3,
 step 94/249 completed (loss: 0.02357347309589386): : 4560it [19:53,
  7.45it/s]
Training Epoch: 2/3,
 step 95/249 completed (loss: 0.04292788729071617): : 4560it [19:53,
  7.45it/s]
Training Epoch: 2/3,
 step 95/249 completed (loss: 0.04292788729071617): : 4656it [20:05,
  7.53it/s]
Training Epoch: 2/3,
 step 96/249 completed (loss: 0.11760813742876053): : 4656it [20:05,
  7.53it/s]
Training Epoch: 2/3,
 step 96/249 completed (loss: 0.11760813742876053): : 4753it [20:17,
  7.61it/s]
Training Epoch: 2/3,
 step 97/249 completed (loss: 0.02398555539548397): : 4753it [20:18,
  7.61it/s]
Training Epoch: 2/3,
 step 97/249 completed (loss: 0.02398555539548397): : 4851it [20:30,
  7.70it/s]
Training Epoch: 2/3,
 step 98/249 completed (loss: 0.007791509851813316): : 4851it [20:30,
  7.70it/s]
Training Epoch: 2/3,
 step 98/249 completed (loss: 0.007791509851813316): : 4950it [20:42,
  7.78it/s]
Training Epoch: 2/3,
 step 99/249 completed (loss: 0.052009183913469315): : 4950it [20:43,
  7.78it/s]
Training Epoch: 2/3,
 step 99/249 completed (loss: 0.052009183913469315): : 5050it [20:55,
  7.86it/s]
Training Epoch: 2/3,
 step 100/249 completed (loss: 0.047314610332250595): : 5050it [20:55,
  7.86it/s]
Training Epoch: 2/3,
 step 100/249 completed (loss: 0.047314610332250595): : 5151it [21:07,
  7.94it/s]
Training Epoch: 2/3,
 step 101/249 completed (loss: 0.014249215833842754): : 5151it [21:07,
  7.94it/s]
Training Epoch: 2/3,
 step 101/249 completed (loss: 0.014249215833842754): : 5253it [21:20,
  8.02it/s]
Training Epoch: 2/3,
 step 102/249 completed (loss: 0.037651773542165756): : 5253it [21:20,
  8.02it/s]
Training Epoch: 2/3,
 step 102/249 completed (loss: 0.037651773542165756): : 5356it [21:32,
  8.10it/s]
Training Epoch: 2/3,
 step 103/249 completed (loss: 0.08596786856651306): : 5356it [21:32,
  8.10it/s] 
Training Epoch: 2/3,
 step 103/249 completed (loss: 0.08596786856651306): : 5460it [21:44,
  8.18it/s]
Training Epoch: 2/3,
 step 104/249 completed (loss: 0.0415329784154892): : 5460it [21:45,
  8.18it/s] 
Training Epoch: 2/3,
 step 104/249 completed (loss: 0.0415329784154892): : 5565it [21:57,
  8.26it/s]
Training Epoch: 2/3,
 step 105/249 completed (loss: 0.0934995636343956): : 5565it [21:57,
  8.26it/s]
Training Epoch: 2/3,
 step 105/249 completed (loss: 0.0934995636343956): : 5671it [22:09,
  8.34it/s]
Training Epoch: 2/3,
 step 106/249 completed (loss: 0.008144854567945004): : 5671it [22:09,
  8.34it/s]
Training Epoch: 2/3,
 step 106/249 completed (loss: 0.008144854567945004): : 5778it [22:22,
  8.42it/s]
Training Epoch: 2/3,
 step 107/249 completed (loss: 0.030624084174633026): : 5778it [22:22,
  8.42it/s]
Training Epoch: 2/3,
 step 107/249 completed (loss: 0.030624084174633026): : 5886it [22:34,
  8.50it/s]
Training Epoch: 2/3,
 step 108/249 completed (loss: 0.027418626472353935): : 5886it [22:34,
  8.50it/s]
Training Epoch: 2/3,
 step 108/249 completed (loss: 0.027418626472353935): : 5995it [22:47,
  8.58it/s]
Training Epoch: 2/3,
 step 109/249 completed (loss: 0.08729924261569977): : 5995it [22:47,
  8.58it/s] 
Training Epoch: 2/3,
 step 109/249 completed (loss: 0.08729924261569977): : 6105it [22:59,
  8.66it/s]
Training Epoch: 2/3,
 step 110/249 completed (loss: 0.010160275734961033): : 6105it [22:59,
  8.66it/s]
Training Epoch: 2/3,
 step 110/249 completed (loss: 0.010160275734961033): : 6216it [23:11,
  8.74it/s]
Training Epoch: 2/3,
 step 111/249 completed (loss: 0.07327718287706375): : 6216it [23:12,
  8.74it/s] 
Training Epoch: 2/3,
 step 111/249 completed (loss: 0.07327718287706375): : 6328it [23:24,
  8.81it/s]
Training Epoch: 2/3,
 step 112/249 completed (loss: 0.08516640216112137): : 6328it [23:24,
  8.81it/s]
Training Epoch: 2/3,
 step 112/249 completed (loss: 0.08516640216112137): : 6441it [23:36,
  8.90it/s]
Training Epoch: 2/3,
 step 113/249 completed (loss: 0.08404592424631119): : 6441it [23:37,
  8.90it/s]
Training Epoch: 2/3,
 step 113/249 completed (loss: 0.08404592424631119): : 6555it [23:49,
  8.97it/s]
Training Epoch: 2/3,
 step 114/249 completed (loss: 0.10376731306314468): : 6555it [23:49,
  8.97it/s]
Training Epoch: 2/3,
 step 114/249 completed (loss: 0.10376731306314468): : 6670it [24:01,
  9.06it/s]
Training Epoch: 2/3,
 step 115/249 completed (loss: 0.045830994844436646): : 6670it [24:01,
  9.06it/s]
Training Epoch: 2/3,
 step 115/249 completed (loss: 0.045830994844436646): : 6786it [24:14,
  9.14it/s]
Training Epoch: 2/3,
 step 116/249 completed (loss: 0.036571916192770004): : 6786it [24:14,
  9.14it/s]
Training Epoch: 2/3,
 step 116/249 completed (loss: 0.036571916192770004): : 6903it [24:26,
  9.22it/s]
Training Epoch: 2/3,
 step 117/249 completed (loss: 0.043571699410676956): : 6903it [24:26,
  9.22it/s]
Training Epoch: 2/3,
 step 117/249 completed (loss: 0.043571699410676956): : 7021it [24:39,
  9.31it/s]
Training Epoch: 2/3,
 step 118/249 completed (loss: 0.023956298828125): : 7021it [24:39,
  9.31it/s]   
Training Epoch: 2/3,
 step 118/249 completed (loss: 0.023956298828125): : 7140it [24:51,
  9.39it/s]
Training Epoch: 2/3,
 step 119/249 completed (loss: 0.11237198859453201): : 7140it [24:51,
  9.39it/s]
Training Epoch: 2/3,
 step 119/249 completed (loss: 0.11237198859453201): : 7260it [25:03,
  9.47it/s]
Training Epoch: 2/3,
 step 120/249 completed (loss: 0.017025316134095192): : 7260it [25:04,
  9.47it/s]
Training Epoch: 2/3,
 step 120/249 completed (loss: 0.017025316134095192): : 7381it [25:16,
  9.55it/s]
Training Epoch: 2/3,
 step 121/249 completed (loss: 0.002844288945198059): : 7381it [25:16,
  9.55it/s]
Training Epoch: 2/3,
 step 121/249 completed (loss: 0.002844288945198059): : 7503it [25:28,
  9.63it/s]
Training Epoch: 2/3,
 step 122/249 completed (loss: 0.03797156363725662): : 7503it [25:28,
  9.63it/s] 
Training Epoch: 2/3,
 step 122/249 completed (loss: 0.03797156363725662): : 7626it [25:41,
  9.71it/s]
Training Epoch: 2/3,
 step 123/249 completed (loss: 0.035551875829696655): : 7626it [25:41,
  9.71it/s]
Training Epoch: 2/3,
 step 123/249 completed (loss: 0.035551875829696655): : 7750it [25:53,
  9.79it/s]
Training Epoch: 2/3,
 step 124/249 completed (loss: 0.028547067195177078): : 7750it [25:53,
  9.79it/s]
Training Epoch: 2/3,
 step 124/249 completed (loss: 0.028547067195177078): : 7875it [26:06,
  9.86it/s]
Training Epoch: 2/3,
 step 125/249 completed (loss: 0.005293446592986584): : 7875it [26:06,
  9.86it/s]
Training Epoch: 2/3,
 step 125/249 completed (loss: 0.005293446592986584): : 7875it [26:16,
  9.86it/s]
Training Epoch: 2/3,
 step 125/249 completed (loss: 0.005293446592986584): : 8001it [26:18,
  9.95it/s]
Training Epoch: 2/3,
 step 126/249 completed (loss: 0.0267050638794899): : 8001it [26:18,
  9.95it/s]  
Training Epoch: 2/3,
 step 126/249 completed (loss: 0.0267050638794899): : 8001it [26:28,
  9.95it/s]
Training Epoch: 2/3,
 step 126/249 completed (loss: 0.0267050638794899): : 8128it [26:30,
 10.03it/s]
Training Epoch: 2/3,
 step 127/249 completed (loss: 0.009095661342144012): : 8128it [26:31,
 10.03it/s]
Training Epoch: 2/3,
 step 127/249 completed (loss: 0.009095661342144012): : 8256it [26:43,
 10.11it/s]
Training Epoch: 2/3,
 step 128/249 completed (loss: 0.010716746561229229): : 8256it [26:43,
 10.11it/s]
Training Epoch: 2/3,
 step 128/249 completed (loss: 0.010716746561229229): : 8385it [26:55,
 10.19it/s]
Training Epoch: 2/3,
 step 129/249 completed (loss: 0.05347464606165886): : 8385it [26:55,
 10.19it/s] 
Training Epoch: 2/3,
 step 129/249 completed (loss: 0.05347464606165886): : 8385it [27:06,
 10.19it/s]
Training Epoch: 2/3,
 step 129/249 completed (loss: 0.05347464606165886): : 8515it [27:08,
 10.27it/s]
Training Epoch: 2/3,
 step 130/249 completed (loss: 0.0001833106070989743): : 8515it [27:08,
 10.27it/s]
Training Epoch: 2/3,
 step 130/249 completed (loss: 0.0001833106070989743): : 8515it [27:18,
 10.27it/s]
Training Epoch: 2/3,
 step 130/249 completed (loss: 0.0001833106070989743): : 8646it [27:20,
 10.35it/s]
Training Epoch: 2/3,
 step 131/249 completed (loss: 0.05327264964580536): : 8646it [27:20,
 10.35it/s]  
Training Epoch: 2/3,
 step 131/249 completed (loss: 0.05327264964580536): : 8778it [27:33,
 10.43it/s]
Training Epoch: 2/3,
 step 132/249 completed (loss: 0.01801718957722187): : 8778it [27:33,
 10.43it/s]
Training Epoch: 2/3,
 step 132/249 completed (loss: 0.01801718957722187): : 8911it [27:45,
 10.50it/s]
Training Epoch: 2/3,
 step 133/249 completed (loss: 0.056386422365903854): : 8911it [27:45,
 10.50it/s]
Training Epoch: 2/3,
 step 133/249 completed (loss: 0.056386422365903854): : 8911it [27:56,
 10.50it/s]
Training Epoch: 2/3,
 step 133/249 completed (loss: 0.056386422365903854): : 9045it [27:57,
 10.59it/s]
Training Epoch: 2/3,
 step 134/249 completed (loss: 0.019718287512660027): : 9045it [27:58,
 10.59it/s]
Training Epoch: 2/3,
 step 134/249 completed (loss: 0.019718287512660027): : 9045it [28:08,
 10.59it/s]
Training Epoch: 2/3,
 step 134/249 completed (loss: 0.019718287512660027): : 9180it [28:10,
 10.67it/s]
Training Epoch: 2/3,
 step 135/249 completed (loss: 0.003806527005508542): : 9180it [28:10,
 10.67it/s]
Training Epoch: 2/3,
 step 135/249 completed (loss: 0.003806527005508542): : 9316it [28:22,
 10.75it/s]
Training Epoch: 2/3,
 step 136/249 completed (loss: 0.010753236711025238): : 9316it [28:22,
 10.75it/s]
Training Epoch: 2/3,
 step 136/249 completed (loss: 0.010753236711025238): : 9453it [28:35,
 10.83it/s]
Training Epoch: 2/3,
 step 137/249 completed (loss: 0.07329091429710388): : 9453it [28:35,
 10.83it/s] 
Training Epoch: 2/3,
 step 137/249 completed (loss: 0.07329091429710388): : 9453it [28:46,
 10.83it/s]
Training Epoch: 2/3,
 step 137/249 completed (loss: 0.07329091429710388): : 9591it [28:47,
 10.91it/s]
Training Epoch: 2/3,
 step 138/249 completed (loss: 0.07265323400497437): : 9591it [28:47,
 10.91it/s]
Training Epoch: 2/3,
 step 138/249 completed (loss: 0.07265323400497437): : 9591it [28:58,
 10.91it/s]
Training Epoch: 2/3,
 step 138/249 completed (loss: 0.07265323400497437): : 9730it [29:00,
 10.99it/s]
Training Epoch: 2/3,
 step 139/249 completed (loss: 0.015506197698414326): : 9730it [29:00,
 10.99it/s]
Training Epoch: 2/3,
 step 139/249 completed (loss: 0.015506197698414326): : 9870it [29:12,
 11.08it/s]
Training Epoch: 2/3,
 step 140/249 completed (loss: 0.051240358501672745): : 9870it [29:12,
 11.08it/s]
Training Epoch: 2/3,
 step 140/249 completed (loss: 0.051240358501672745): : 10011it [29:24,
 11.16it/s]
Training Epoch: 2/3,
 step 141/249 completed (loss: 0.009092600084841251): : 10011it [29:25,
 11.16it/s]
Training Epoch: 2/3,
 step 141/249 completed (loss: 0.009092600084841251): : 10011it [29:36,
 11.16it/s]
Training Epoch: 2/3,
 step 141/249 completed (loss: 0.009092600084841251): : 10153it [29:37,
 11.24it/s]
Training Epoch: 2/3,
 step 142/249 completed (loss: 0.043936122208833694): : 10153it [29:37,
 11.24it/s]
Training Epoch: 2/3,
 step 142/249 completed (loss: 0.043936122208833694): : 10153it [29:48,
 11.24it/s]
Training Epoch: 2/3,
 step 142/249 completed (loss: 0.043936122208833694): : 10296it [29:49,
 11.31it/s]
Training Epoch: 2/3,
 step 143/249 completed (loss: 0.0025369729846715927): : 10296it [29:50,
 11.31it/s]
Training Epoch: 2/3,
 step 143/249 completed (loss: 0.0025369729846715927): : 10440it [30:02,
 11.39it/s]
Training Epoch: 2/3,
 step 144/249 completed (loss: 0.03099404089152813): : 10440it [30:02,
 11.39it/s]  
Training Epoch: 2/3,
 step 144/249 completed (loss: 0.03099404089152813): : 10585it [30:14,
 11.49it/s]
Training Epoch: 2/3,
 step 145/249 completed (loss: 0.08425968885421753): : 10585it [30:14,
 11.49it/s]
Training Epoch: 2/3,
 step 145/249 completed (loss: 0.08425968885421753): : 10585it [30:26,
 11.49it/s]
Training Epoch: 2/3,
 step 145/249 completed (loss: 0.08425968885421753): : 10731it [30:27,
 11.56it/s]
Training Epoch: 2/3,
 step 146/249 completed (loss: 0.06417720019817352): : 10731it [30:27,
 11.56it/s]
Training Epoch: 2/3,
 step 146/249 completed (loss: 0.06417720019817352): : 10731it [30:38,
 11.56it/s]
Training Epoch: 2/3,
 step 146/249 completed (loss: 0.06417720019817352): : 10878it [30:39,
 11.64it/s]
Training Epoch: 2/3,
 step 147/249 completed (loss: 0.05805763602256775): : 10878it [30:39,
 11.64it/s]
Training Epoch: 2/3,
 step 147/249 completed (loss: 0.05805763602256775): : 11026it [30:51,
 11.72it/s]
Training Epoch: 2/3,
 step 148/249 completed (loss: 0.039573416113853455): : 11026it [30:52,
 11.72it/s]
Training Epoch: 2/3,
 step 148/249 completed (loss: 0.039573416113853455): : 11175it [31:04,
 11.79it/s]
Training Epoch: 2/3,
 step 149/249 completed (loss: 0.11594444513320923): : 11175it [31:04,
 11.79it/s] 
Training Epoch: 2/3,
 step 149/249 completed (loss: 0.11594444513320923): : 11175it [31:16,
 11.79it/s]
Training Epoch: 2/3,
 step 149/249 completed (loss: 0.11594444513320923): : 11325it [31:16,
 11.87it/s]
Training Epoch: 2/3,
 step 150/249 completed (loss: 0.041038669645786285): : 11325it [31:17,
 11.87it/s]
Training Epoch: 2/3,
 step 150/249 completed (loss: 0.041038669645786285): : 11325it [31:28,
 11.87it/s]
Training Epoch: 2/3,
 step 150/249 completed (loss: 0.041038669645786285): : 11476it [31:29,
 11.95it/s]
Training Epoch: 2/3,
 step 151/249 completed (loss: 0.04751099646091461): : 11476it [31:29,
 11.95it/s] 
Training Epoch: 2/3,
 step 151/249 completed (loss: 0.04751099646091461): : 11628it [31:41,
 12.04it/s]
Training Epoch: 2/3,
 step 152/249 completed (loss: 0.030335279181599617): : 11628it [31:41,
 12.04it/s]
Training Epoch: 2/3,
 step 152/249 completed (loss: 0.030335279181599617): : 11781it [31:54,
 12.11it/s]
Training Epoch: 2/3,
 step 153/249 completed (loss: 0.15368932485580444): : 11781it [31:54,
 12.11it/s] 
Training Epoch: 2/3,
 step 153/249 completed (loss: 0.15368932485580444): : 11935it [32:06,
 12.19it/s]
Training Epoch: 2/3,
 step 154/249 completed (loss: 0.015552184544503689): : 11935it [32:06,
 12.19it/s]
Training Epoch: 2/3,
 step 154/249 completed (loss: 0.015552184544503689): : 11935it [32:16,
 12.19it/s]
Training Epoch: 2/3,
 step 154/249 completed (loss: 0.015552184544503689): : 12090it [32:19,
 12.28it/s]
Training Epoch: 2/3,
 step 155/249 completed (loss: 0.10263919830322266): : 12090it [32:19,
 12.28it/s] 
Training Epoch: 2/3,
 step 155/249 completed (loss: 0.10263919830322266): : 12246it [32:31,
 12.36it/s]
Training Epoch: 2/3,
 step 156/249 completed (loss: 0.02658839151263237): : 12246it [32:31,
 12.36it/s]
Training Epoch: 2/3,
 step 156/249 completed (loss: 0.02658839151263237): : 12403it [32:43,
 12.44it/s]
Training Epoch: 2/3,
 step 157/249 completed (loss: 0.009899592027068138): : 12403it [32:44,
 12.44it/s]
Training Epoch: 2/3,
 step 157/249 completed (loss: 0.009899592027068138): : 12561it [32:56,
 12.52it/s]
Training Epoch: 2/3,
 step 158/249 completed (loss: 0.005912257358431816): : 12561it [32:56,
 12.52it/s]
Training Epoch: 2/3,
 step 158/249 completed (loss: 0.005912257358431816): : 12561it [33:06,
 12.52it/s]
Training Epoch: 2/3,
 step 158/249 completed (loss: 0.005912257358431816): : 12720it [33:08,
 12.60it/s]
Training Epoch: 2/3,
 step 159/249 completed (loss: 0.0013409790117293596): : 12720it [33:08,
 12.60it/s]
Training Epoch: 2/3,
 step 159/249 completed (loss: 0.0013409790117293596): : 12880it [33:21,
 12.68it/s]
Training Epoch: 2/3,
 step 160/249 completed (loss: 0.01277084555476904): : 12880it [33:21,
 12.68it/s]  
Training Epoch: 2/3,
 step 160/249 completed (loss: 0.01277084555476904): : 13041it [33:33,
 12.76it/s]
Training Epoch: 2/3,
 step 161/249 completed (loss: 0.06118588522076607): : 13041it [33:33,
 12.76it/s]
Training Epoch: 2/3,
 step 161/249 completed (loss: 0.06118588522076607): : 13203it [33:46,
 12.83it/s]
Training Epoch: 2/3,
 step 162/249 completed (loss: 0.037560611963272095): : 13203it [33:46,
 12.83it/s]
Training Epoch: 2/3,
 step 162/249 completed (loss: 0.037560611963272095): : 13203it [33:56,
 12.83it/s]
Training Epoch: 2/3,
 step 162/249 completed (loss: 0.037560611963272095): : 13366it [33:58,
 12.92it/s]
Training Epoch: 2/3,
 step 163/249 completed (loss: 0.007384270429611206): : 13366it [33:58,
 12.92it/s]
Training Epoch: 2/3,
 step 163/249 completed (loss: 0.007384270429611206): : 13366it [34:08,
 12.92it/s]
Training Epoch: 2/3,
 step 163/249 completed (loss: 0.007384270429611206): : 13530it [34:10,
 13.00it/s]
Training Epoch: 2/3,
 step 164/249 completed (loss: 0.007583663333207369): : 13530it [34:11,
 13.00it/s]
Training Epoch: 2/3,
 step 164/249 completed (loss: 0.007583663333207369): : 13695it [34:23,
 13.09it/s]
Training Epoch: 2/3,
 step 165/249 completed (loss: 0.005656632129102945): : 13695it [34:23,
 13.09it/s]
Training Epoch: 2/3,
 step 165/249 completed (loss: 0.005656632129102945): : 13861it [34:35,
 13.17it/s]
Training Epoch: 2/3,
 step 166/249 completed (loss: 0.01440271083265543): : 13861it [34:35,
 13.17it/s] 
Training Epoch: 2/3,
 step 166/249 completed (loss: 0.01440271083265543): : 13861it [34:46,
 13.17it/s]
Training Epoch: 2/3,
 step 166/249 completed (loss: 0.01440271083265543): : 14028it [34:48,
 13.26it/s]
Training Epoch: 2/3,
 step 167/249 completed (loss: 0.010259673930704594): : 14028it [34:48,
 13.26it/s]
Training Epoch: 2/3,
 step 167/249 completed (loss: 0.010259673930704594): : 14028it [34:58,
 13.26it/s]
Training Epoch: 2/3,
 step 167/249 completed (loss: 0.010259673930704594): : 14196it [35:00,
 13.33it/s]
Training Epoch: 2/3,
 step 168/249 completed (loss: 0.05449983477592468): : 14196it [35:00,
 13.33it/s] 
Training Epoch: 2/3,
 step 168/249 completed (loss: 0.05449983477592468): : 14365it [35:13,
 13.41it/s]
Training Epoch: 2/3,
 step 169/249 completed (loss: 0.005906353238970041): : 14365it [35:13,
 13.41it/s]
Training Epoch: 2/3,
 step 169/249 completed (loss: 0.005906353238970041): : 14535it [35:25,
 13.48it/s]
Training Epoch: 2/3,
 step 170/249 completed (loss: 0.006027690134942532): : 14535it [35:25,
 13.48it/s]
Training Epoch: 2/3,
 step 170/249 completed (loss: 0.006027690134942532): : 14535it [35:36,
 13.48it/s]
Training Epoch: 2/3,
 step 170/249 completed (loss: 0.006027690134942532): : 14706it [35:37,
 13.56it/s]
Training Epoch: 2/3,
 step 171/249 completed (loss: 0.0011710404651239514): : 14706it [35:38,
 13.56it/s]
Training Epoch: 2/3,
 step 171/249 completed (loss: 0.0011710404651239514): : 14706it [35:48,
 13.56it/s]
Training Epoch: 2/3,
 step 171/249 completed (loss: 0.0011710404651239514): : 14878it [35:50,
 13.64it/s]
Training Epoch: 2/3,
 step 172/249 completed (loss: 0.019526172429323196): : 14878it [35:50,
 13.64it/s] 
Training Epoch: 2/3,
 step 172/249 completed (loss: 0.019526172429323196): : 15051it [36:02,
 13.72it/s]
Training Epoch: 2/3,
 step 173/249 completed (loss: 0.00792071782052517): : 15051it [36:03,
 13.72it/s] 
Training Epoch: 2/3,
 step 173/249 completed (loss: 0.00792071782052517): : 15225it [36:15,
 13.80it/s]
Training Epoch: 2/3,
 step 174/249 completed (loss: 0.02530761994421482): : 15225it [36:15,
 13.80it/s]
Training Epoch: 2/3,
 step 174/249 completed (loss: 0.02530761994421482): : 15225it [36:26,
 13.80it/s]
Training Epoch: 2/3,
 step 174/249 completed (loss: 0.02530761994421482): : 15400it [36:27,
 13.89it/s]
Training Epoch: 2/3,
 step 175/249 completed (loss: 0.11597757786512375): : 15400it [36:27,
 13.89it/s]
Training Epoch: 2/3,
 step 175/249 completed (loss: 0.11597757786512375): : 15400it [36:38,
 13.89it/s]
Training Epoch: 2/3,
 step 175/249 completed (loss: 0.11597757786512375): : 15576it [36:40,
 13.94it/s]
Training Epoch: 2/3,
 step 176/249 completed (loss: 0.0971132442355156): : 15576it [36:40,
 13.94it/s] 
Training Epoch: 2/3,
 step 176/249 completed (loss: 0.0971132442355156): : 15753it [36:52,
 14.03it/s]
Training Epoch: 2/3,
 step 177/249 completed (loss: 0.008020822890102863): : 15753it [36:52,
 14.03it/s]
Training Epoch: 2/3,
 step 177/249 completed (loss: 0.008020822890102863): : 15931it [37:05,
 14.12it/s]
Training Epoch: 2/3,
 step 178/249 completed (loss: 0.00011392467422410846): : 15931it [37:05,
 14.12it/s]
Training Epoch: 2/3,
 step 178/249 completed (loss: 0.00011392467422410846): : 15931it [37:16,
 14.12it/s]
Training Epoch: 2/3,
 step 178/249 completed (loss: 0.00011392467422410846): : 16110it [37:17,
 14.20it/s]
Training Epoch: 2/3,
 step 179/249 completed (loss: 0.0002695858711376786): : 16110it [37:17,
 14.20it/s] 
Training Epoch: 2/3,
 step 179/249 completed (loss: 0.0002695858711376786): : 16110it [37:28,
 14.20it/s]
Training Epoch: 2/3,
 step 179/249 completed (loss: 0.0002695858711376786): : 16290it [37:29,
 14.29it/s]
Training Epoch: 2/3,
 step 180/249 completed (loss: 0.08293367922306061): : 16290it [37:30,
 14.29it/s]  
Training Epoch: 2/3,
 step 180/249 completed (loss: 0.08293367922306061): : 16471it [37:42,
 14.37it/s]
Training Epoch: 2/3,
 step 181/249 completed (loss: 0.13662059605121613): : 16471it [37:42,
 14.37it/s]
Training Epoch: 2/3,
 step 181/249 completed (loss: 0.13662059605121613): : 16653it [37:54,
 14.46it/s]
Training Epoch: 2/3,
 step 182/249 completed (loss: 0.12779200077056885): : 16653it [37:54,
 14.46it/s]
Training Epoch: 2/3,
 step 182/249 completed (loss: 0.12779200077056885): : 16653it [38:06,
 14.46it/s]
Training Epoch: 2/3,
 step 182/249 completed (loss: 0.12779200077056885): : 16836it [38:07,
 14.53it/s]
Training Epoch: 2/3,
 step 183/249 completed (loss: 0.1093950942158699): : 16836it [38:07,
 14.53it/s] 
Training Epoch: 2/3,
 step 183/249 completed (loss: 0.1093950942158699): : 16836it [38:18,
 14.53it/s]
Training Epoch: 2/3,
 step 183/249 completed (loss: 0.1093950942158699): : 17020it [38:19,
 14.62it/s]
Training Epoch: 2/3,
 step 184/249 completed (loss: 0.05791996046900749): : 17020it [38:19,
 14.62it/s]
Training Epoch: 2/3,
 step 184/249 completed (loss: 0.05791996046900749): : 17205it [38:32,
 14.70it/s]
Training Epoch: 2/3,
 step 185/249 completed (loss: 0.048690151423215866): : 17205it [38:32,
 14.70it/s]
Training Epoch: 2/3,
 step 185/249 completed (loss: 0.048690151423215866): : 17391it [38:44,
 14.77it/s]
Training Epoch: 2/3,
 step 186/249 completed (loss: 0.047043636441230774): : 17391it [38:44,
 14.77it/s]
Training Epoch: 2/3,
 step 186/249 completed (loss: 0.047043636441230774): : 17391it [38:56,
 14.77it/s]
Training Epoch: 2/3,
 step 186/249 completed (loss: 0.047043636441230774): : 17578it [38:56,
 14.85it/s]
Training Epoch: 2/3,
 step 187/249 completed (loss: 0.03844964876770973): : 17578it [38:57,
 14.85it/s] 
Training Epoch: 2/3,
 step 187/249 completed (loss: 0.03844964876770973): : 17578it [39:08,
 14.85it/s]
Training Epoch: 2/3,
 step 187/249 completed (loss: 0.03844964876770973): : 17766it [39:09,
 14.94it/s]
Training Epoch: 2/3,
 step 188/249 completed (loss: 0.027466362342238426): : 17766it [39:09,
 14.94it/s]
Training Epoch: 2/3,
 step 188/249 completed (loss: 0.027466362342238426): : 17955it [39:21,
 15.02it/s]
Training Epoch: 2/3,
 step 189/249 completed (loss: 0.029787791892886162): : 17955it [39:21,
 15.02it/s]
Training Epoch: 2/3,
 step 189/249 completed (loss: 0.029787791892886162): : 18145it [39:34,
 15.10it/s]
Training Epoch: 2/3,
 step 190/249 completed (loss: 0.017009591683745384): : 18145it [39:34,
 15.10it/s]
Training Epoch: 2/3,
 step 190/249 completed (loss: 0.017009591683745384): : 18336it [39:46,
 15.19it/s]
Training Epoch: 2/3,
 step 191/249 completed (loss: 0.03454029560089111): : 18336it [39:46,
 15.19it/s] 
Training Epoch: 2/3,
 step 191/249 completed (loss: 0.03454029560089111): : 18336it [39:56,
 15.19it/s]
Training Epoch: 2/3,
 step 191/249 completed (loss: 0.03454029560089111): : 18528it [39:59,
 15.26it/s]
Training Epoch: 2/3,
 step 192/249 completed (loss: 0.07508037239313126): : 18528it [39:59,
 15.26it/s]
Training Epoch: 2/3,
 step 192/249 completed (loss: 0.07508037239313126): : 18721it [40:11,
 15.34it/s]
Training Epoch: 2/3,
 step 193/249 completed (loss: 0.020208584144711494): : 18721it [40:11,
 15.34it/s]
Training Epoch: 2/3,
 step 193/249 completed (loss: 0.020208584144711494): : 18915it [40:23,
 15.42it/s]
Training Epoch: 2/3,
 step 194/249 completed (loss: 0.11212456971406937): : 18915it [40:24,
 15.42it/s] 
Training Epoch: 2/3,
 step 194/249 completed (loss: 0.11212456971406937): : 19110it [40:36,
 15.51it/s]
Training Epoch: 2/3,
 step 195/249 completed (loss: 0.06777158379554749): : 19110it [40:36,
 15.51it/s]
Training Epoch: 2/3,
 step 195/249 completed (loss: 0.06777158379554749): : 19110it [40:46,
 15.51it/s]
Training Epoch: 2/3,
 step 195/249 completed (loss: 0.06777158379554749): : 19306it [40:48,
 15.59it/s]
Training Epoch: 2/3,
 step 196/249 completed (loss: 0.023952998220920563): : 19306it [40:48,
 15.59it/s]
Training Epoch: 2/3,
 step 196/249 completed (loss: 0.023952998220920563): : 19503it [41:01,
 15.66it/s]
Training Epoch: 2/3,
 step 197/249 completed (loss: 0.022147560492157936): : 19503it [41:01,
 15.66it/s]
Training Epoch: 2/3,
 step 197/249 completed (loss: 0.022147560492157936): : 19701it [41:13,
 15.75it/s]
Training Epoch: 2/3,
 step 198/249 completed (loss: 0.11685691028833389): : 19701it [41:13,
 15.75it/s] 
Training Epoch: 2/3,
 step 198/249 completed (loss: 0.11685691028833389): : 19900it [41:26,
 15.83it/s]
Training Epoch: 2/3,
 step 199/249 completed (loss: 0.04852201044559479): : 19900it [41:26,
 15.83it/s]
Training Epoch: 2/3,
 step 199/249 completed (loss: 0.04852201044559479): : 19900it [41:36,
 15.83it/s]
Training Epoch: 2/3,
 step 199/249 completed (loss: 0.04852201044559479): : 20100it [41:38,
 15.90it/s]
Training Epoch: 2/3,
 step 200/249 completed (loss: 0.025364579632878304): : 20100it [41:38,
 15.90it/s]
Training Epoch: 2/3,
 step 200/249 completed (loss: 0.025364579632878304): : 20100it [41:48,
 15.90it/s]
Training Epoch: 2/3,
 step 200/249 completed (loss: 0.025364579632878304): : 20301it [41:50,
 15.98it/s]
Training Epoch: 2/3,
 step 201/249 completed (loss: 0.01821776106953621): : 20301it [41:51,
 15.98it/s] 
Training Epoch: 2/3,
 step 201/249 completed (loss: 0.01821776106953621): : 20503it [42:03,
 16.06it/s]
Training Epoch: 2/3,
 step 202/249 completed (loss: 0.04241609200835228): : 20503it [42:03,
 16.06it/s]
Training Epoch: 2/3,
 step 202/249 completed (loss: 0.04241609200835228): : 20706it [42:15,
 16.15it/s]
Training Epoch: 2/3,
 step 203/249 completed (loss: 0.011579152196645737): : 20706it [42:15,
 16.15it/s]
Training Epoch: 2/3,
 step 203/249 completed (loss: 0.011579152196645737): : 20706it [42:26,
 16.15it/s]
Training Epoch: 2/3,
 step 203/249 completed (loss: 0.011579152196645737): : 20910it [42:28,
 16.23it/s]
Training Epoch: 2/3,
 step 204/249 completed (loss: 0.011106234043836594): : 20910it [42:28,
 16.23it/s]
Training Epoch: 2/3,
 step 204/249 completed (loss: 0.011106234043836594): : 20910it [42:38,
 16.23it/s]
Training Epoch: 2/3,
 step 204/249 completed (loss: 0.011106234043836594): : 21115it [42:40,
 16.31it/s]
Training Epoch: 2/3,
 step 205/249 completed (loss: 0.19553954899311066): : 21115it [42:40,
 16.31it/s] 
Training Epoch: 2/3,
 step 205/249 completed (loss: 0.19553954899311066): : 21321it [42:53,
 16.40it/s]
Training Epoch: 2/3,
 step 206/249 completed (loss: 0.0010166887659579515): : 21321it [42:53,
 16.40it/s]
Training Epoch: 2/3,
 step 206/249 completed (loss: 0.0010166887659579515): : 21528it [43:05,
 16.47it/s]
Training Epoch: 2/3,
 step 207/249 completed (loss: 0.03175140172243118): : 21528it [43:05,
 16.47it/s]  
Training Epoch: 2/3,
 step 207/249 completed (loss: 0.03175140172243118): : 21528it [43:16,
 16.47it/s]
Training Epoch: 2/3,
 step 207/249 completed (loss: 0.03175140172243118): : 21736it [43:17,
 16.55it/s]
Training Epoch: 2/3,
 step 208/249 completed (loss: 0.03255198150873184): : 21736it [43:18,
 16.55it/s]
Training Epoch: 2/3,
 step 208/249 completed (loss: 0.03255198150873184): : 21736it [43:28,
 16.55it/s]
Training Epoch: 2/3,
 step 208/249 completed (loss: 0.03255198150873184): : 21945it [43:30,
 16.63it/s]
Training Epoch: 2/3,
 step 209/249 completed (loss: 0.04433628171682358): : 21945it [43:30,
 16.63it/s]
Training Epoch: 2/3,
 step 209/249 completed (loss: 0.04433628171682358): : 22155it [43:42,
 16.72it/s]
Training Epoch: 2/3,
 step 210/249 completed (loss: 0.02232496254146099): : 22155it [43:42,
 16.72it/s]
Training Epoch: 2/3,
 step 210/249 completed (loss: 0.02232496254146099): : 22366it [43:55,
 16.80it/s]
Training Epoch: 2/3,
 step 211/249 completed (loss: 0.0300032626837492): : 22366it [43:55,
 16.80it/s] 
Training Epoch: 2/3,
 step 211/249 completed (loss: 0.0300032626837492): : 22366it [44:06,
 16.80it/s]
Training Epoch: 2/3,
 step 211/249 completed (loss: 0.0300032626837492): : 22578it [44:07,
 16.87it/s]
Training Epoch: 2/3,
 step 212/249 completed (loss: 0.09994754940271378): : 22578it [44:07,
 16.87it/s]
Training Epoch: 2/3,
 step 212/249 completed (loss: 0.09994754940271378): : 22578it [44:18,
 16.87it/s]
Training Epoch: 2/3,
 step 212/249 completed (loss: 0.09994754940271378): : 22791it [44:20,
 16.95it/s]
Training Epoch: 2/3,
 step 213/249 completed (loss: 0.04596998170018196): : 22791it [44:20,
 16.95it/s]
Training Epoch: 2/3,
 step 213/249 completed (loss: 0.04596998170018196): : 23005it [44:32,
 17.02it/s]
Training Epoch: 2/3,
 step 214/249 completed (loss: 0.01818654127418995): : 23005it [44:32,
 17.02it/s]
Training Epoch: 2/3,
 step 214/249 completed (loss: 0.01818654127418995): : 23220it [44:44,
 17.10it/s]
Training Epoch: 2/3,
 step 215/249 completed (loss: 0.032227251678705215): : 23220it [44:45,
 17.10it/s]
Training Epoch: 2/3,
 step 215/249 completed (loss: 0.032227251678705215): : 23220it [44:56,
 17.10it/s]
Training Epoch: 2/3,
 step 215/249 completed (loss: 0.032227251678705215): : 23436it [44:57,
 17.18it/s]
Training Epoch: 2/3,
 step 216/249 completed (loss: 0.034961096942424774): : 23436it [44:57,
 17.18it/s]
Training Epoch: 2/3,
 step 216/249 completed (loss: 0.034961096942424774): : 23436it [45:08,
 17.18it/s]
Training Epoch: 2/3,
 step 216/249 completed (loss: 0.034961096942424774): : 23653it [45:09,
 17.27it/s]
Training Epoch: 2/3,
 step 217/249 completed (loss: 0.04193532094359398): : 23653it [45:09,
 17.27it/s] 
Training Epoch: 2/3,
 step 217/249 completed (loss: 0.04193532094359398): : 23871it [45:22,
 17.35it/s]
Training Epoch: 2/3,
 step 218/249 completed (loss: 0.027638742700219154): : 23871it [45:22,
 17.35it/s]
Training Epoch: 2/3,
 step 218/249 completed (loss: 0.027638742700219154): : 24090it [45:34,
 17.42it/s]
Training Epoch: 2/3,
 step 219/249 completed (loss: 0.07888109982013702): : 24090it [45:34,
 17.42it/s] 
Training Epoch: 2/3,
 step 219/249 completed (loss: 0.07888109982013702): : 24090it [45:46,
 17.42it/s]
Training Epoch: 2/3,
 step 219/249 completed (loss: 0.07888109982013702): : 24310it [45:47,
 17.50it/s]
Training Epoch: 2/3,
 step 220/249 completed (loss: 0.17575418949127197): : 24310it [45:47,
 17.50it/s]
Training Epoch: 2/3,
 step 220/249 completed (loss: 0.17575418949127197): : 24310it [45:58,
 17.50it/s]
Training Epoch: 2/3,
 step 220/249 completed (loss: 0.17575418949127197): : 24531it [45:59,
 17.58it/s]
Training Epoch: 2/3,
 step 221/249 completed (loss: 0.00036174882552586496): : 24531it [45:59,
 17.58it/s]
Training Epoch: 2/3,
 step 221/249 completed (loss: 0.00036174882552586496): : 24753it [46:11,
 17.66it/s]
Training Epoch: 2/3,
 step 222/249 completed (loss: 0.06633605062961578): : 24753it [46:12,
 17.66it/s]   
Training Epoch: 2/3,
 step 222/249 completed (loss: 0.06633605062961578): : 24976it [46:24,
 17.74it/s]
Training Epoch: 2/3,
 step 223/249 completed (loss: 0.029251620173454285): : 24976it [46:24,
 17.74it/s]
Training Epoch: 2/3,
 step 223/249 completed (loss: 0.029251620173454285): : 24976it [46:36,
 17.74it/s]
Training Epoch: 2/3,
 step 223/249 completed (loss: 0.029251620173454285): : 25200it [46:36,
 17.83it/s]
Training Epoch: 2/3,
 step 224/249 completed (loss: 0.04756319895386696): : 25200it [46:37,
 17.83it/s] 
Training Epoch: 2/3,
 step 224/249 completed (loss: 0.04756319895386696): : 25200it [46:48,
 17.83it/s]
Training Epoch: 2/3,
 step 224/249 completed (loss: 0.04756319895386696): : 25425it [46:49,
 17.92it/s]
Training Epoch: 2/3,
 step 225/249 completed (loss: 0.005611775908619165): : 25425it [46:49,
 17.92it/s]
Training Epoch: 2/3,
 step 225/249 completed (loss: 0.005611775908619165): : 25651it [47:01,
 17.99it/s]
Training Epoch: 2/3,
 step 226/249 completed (loss: 0.10052505880594254): : 25651it [47:01,
 17.99it/s] 
Training Epoch: 2/3,
 step 226/249 completed (loss: 0.10052505880594254): : 25878it [47:14,
 18.07it/s]
Training Epoch: 2/3,
 step 227/249 completed (loss: 0.06584325432777405): : 25878it [47:14,
 18.07it/s]
Training Epoch: 2/3,
 step 227/249 completed (loss: 0.06584325432777405): : 26106it [47:26,
 18.15it/s]
Training Epoch: 2/3,
 step 228/249 completed (loss: 0.04318951815366745): : 26106it [47:26,
 18.15it/s]
Training Epoch: 2/3,
 step 228/249 completed (loss: 0.04318951815366745): : 26106it [47:36,
 18.15it/s]
Training Epoch: 2/3,
 step 228/249 completed (loss: 0.04318951815366745): : 26335it [47:38,
 18.23it/s]
Training Epoch: 2/3,
 step 229/249 completed (loss: 0.03207726031541824): : 26335it [47:39,
 18.23it/s]
Training Epoch: 2/3,
 step 229/249 completed (loss: 0.03207726031541824): : 26565it [47:51,
 18.32it/s]
Training Epoch: 2/3,
 step 230/249 completed (loss: 0.029189951717853546): : 26565it [47:51,
 18.32it/s]
Training Epoch: 2/3,
 step 230/249 completed (loss: 0.029189951717853546): : 26796it [48:03,
 18.40it/s]
Training Epoch: 2/3,
 step 231/249 completed (loss: 0.017255334183573723): : 26796it [48:04,
 18.40it/s]
Training Epoch: 2/3,
 step 231/249 completed (loss: 0.017255334183573723): : 27028it [48:16,
 18.49it/s]
Training Epoch: 2/3,
 step 232/249 completed (loss: 0.07570602744817734): : 27028it [48:16,
 18.49it/s] 
Training Epoch: 2/3,
 step 232/249 completed (loss: 0.07570602744817734): : 27028it [48:26,
 18.49it/s]
Training Epoch: 2/3,
 step 232/249 completed (loss: 0.07570602744817734): : 27261it [48:28,
 18.56it/s]
Training Epoch: 2/3,
 step 233/249 completed (loss: 0.006512622814625502): : 27261it [48:28,
 18.56it/s]
Training Epoch: 2/3,
 step 233/249 completed (loss: 0.006512622814625502): : 27261it [48:38,
 18.56it/s]
Training Epoch: 2/3,
 step 233/249 completed (loss: 0.006512622814625502): : 27495it [48:41,
 18.64it/s]
Training Epoch: 2/3,
 step 234/249 completed (loss: 0.026034213602542877): : 27495it [48:41,
 18.64it/s]
Training Epoch: 2/3,
 step 234/249 completed (loss: 0.026034213602542877): : 27730it [48:53,
 18.73it/s]
Training Epoch: 2/3,
 step 235/249 completed (loss: 0.04741587117314339): : 27730it [48:53,
 18.73it/s] 
Training Epoch: 2/3,
 step 235/249 completed (loss: 0.04741587117314339): : 27966it [49:05,
 18.81it/s]
Training Epoch: 2/3,
 step 236/249 completed (loss: 0.025432806462049484): : 27966it [49:06,
 18.81it/s]
Training Epoch: 2/3,
 step 236/249 completed (loss: 0.025432806462049484): : 27966it [49:16,
 18.81it/s]
Training Epoch: 2/3,
 step 236/249 completed (loss: 0.025432806462049484): : 28203it [49:18,
 18.88it/s]
Training Epoch: 2/3,
 step 237/249 completed (loss: 0.07019501179456711): : 28203it [49:18,
 18.88it/s] 
Training Epoch: 2/3,
 step 237/249 completed (loss: 0.07019501179456711): : 28203it [49:28,
 18.88it/s]
Training Epoch: 2/3,
 step 237/249 completed (loss: 0.07019501179456711): : 28441it [49:30,
 18.97it/s]
Training Epoch: 2/3,
 step 238/249 completed (loss: 0.04183678701519966): : 28441it [49:30,
 18.97it/s]
Training Epoch: 2/3,
 step 238/249 completed (loss: 0.04183678701519966): : 28680it [49:43,
 19.03it/s]
Training Epoch: 2/3,
 step 239/249 completed (loss: 0.030129609629511833): : 28680it [49:43,
 19.03it/s]
Training Epoch: 2/3,
 step 239/249 completed (loss: 0.030129609629511833): : 28920it [49:55,
 19.10it/s]
Training Epoch: 2/3,
 step 240/249 completed (loss: 0.09308574348688126): : 28920it [49:55,
 19.10it/s] 
Training Epoch: 2/3,
 step 240/249 completed (loss: 0.09308574348688126): : 28920it [50:06,
 19.10it/s]
Training Epoch: 2/3,
 step 240/249 completed (loss: 0.09308574348688126): : 29161it [50:08,
 19.18it/s]
Training Epoch: 2/3,
 step 241/249 completed (loss: 0.09848537296056747): : 29161it [50:08,
 19.18it/s]
Training Epoch: 2/3,
 step 241/249 completed (loss: 0.09848537296056747): : 29161it [50:18,
 19.18it/s]
Training Epoch: 2/3,
 step 241/249 completed (loss: 0.09848537296056747): : 29403it [50:20,
 19.27it/s]
Training Epoch: 2/3,
 step 242/249 completed (loss: 0.00432176049798727): : 29403it [50:20,
 19.27it/s]
Training Epoch: 2/3,
 step 242/249 completed (loss: 0.00432176049798727): : 29646it [50:33,
 19.36it/s]
Training Epoch: 2/3,
 step 243/249 completed (loss: 0.0007978196372278035): : 29646it [50:33,
 19.36it/s]
Training Epoch: 2/3,
 step 243/249 completed (loss: 0.0007978196372278035): : 29890it [50:45,
 19.44it/s]
Training Epoch: 2/3,
 step 244/249 completed (loss: 0.06084407493472099): : 29890it [50:45,
 19.44it/s]  
Training Epoch: 2/3,
 step 244/249 completed (loss: 0.06084407493472099): : 29890it [50:56,
 19.44it/s]
Training Epoch: 2/3,
 step 244/249 completed (loss: 0.06084407493472099): : 30135it [50:57,
 19.53it/s]
Training Epoch: 2/3,
 step 245/249 completed (loss: 0.06872227787971497): : 30135it [50:58,
 19.53it/s]
Training Epoch: 2/3,
 step 245/249 completed (loss: 0.06872227787971497): : 30135it [51:08,
 19.53it/s]
Training Epoch: 2/3,
 step 245/249 completed (loss: 0.06872227787971497): : 30381it [51:10,
 19.61it/s]
Training Epoch: 2/3,
 step 246/249 completed (loss: 0.03800562024116516): : 30381it [51:10,
 19.61it/s]
Training Epoch: 2/3,
 step 246/249 completed (loss: 0.03800562024116516): : 30628it [51:22,
 19.69it/s]
Training Epoch: 2/3,
 step 247/249 completed (loss: 0.012384330853819847): : 30628it [51:22,
 19.69it/s]
Training Epoch: 2/3,
 step 247/249 completed (loss: 0.012384330853819847): : 30876it [51:35,
 19.78it/s]
Training Epoch: 2/3,
 step 248/249 completed (loss: 0.0004284728202037513): : 30876it [51:35,
 19.78it/s]Max CUDA memory allocated was 9 GB
Max CUDA memory reserved was 12 GB
Peak active CUDA memory was 9 GB
Cuda Malloc retires : 0
CPU Total Peak Memory consumed during the train (max): 10 GB


evaluating Epoch:   0%|[32m          [0m| 0/200 [00:00<?,
 ?it/s][A

evaluating Epoch:   0%|[32m          [0m| 1/200 [00:00<02:13,
  1.49it/s][A

evaluating Epoch:   1%|[32m          [0m| 2/200 [00:01<02:01,
  1.63it/s][A

evaluating Epoch:   2%|[32m▏         [0m| 3/200 [00:01<01:56,
  1.69it/s][A

evaluating Epoch:   2%|[32m▏         [0m| 4/200 [00:02<01:54,
  1.72it/s][A

evaluating Epoch:   2%|[32m▎         [0m| 5/200 [00:02<01:53,
  1.72it/s][A

evaluating Epoch:   3%|[32m▎         [0m| 6/200 [00:03<01:52,
  1.73it/s][A

evaluating Epoch:   4%|[32m▎         [0m| 7/200 [00:04<01:51,
  1.73it/s][A

evaluating Epoch:   4%|[32m▍         [0m| 8/200 [00:04<01:49,
  1.76it/s][A

evaluating Epoch:   4%|[32m▍         [0m| 9/200 [00:05<01:48,
  1.77it/s][A

evaluating Epoch:   5%|[32m▌         [0m| 10/200 [00:05<01:46,
  1.78it/s][A

evaluating Epoch:   6%|[32m▌         [0m| 11/200 [00:06<01:47,
  1.76it/s][A

evaluating Epoch:   6%|[32m▌         [0m| 12/200 [00:06<01:44,
  1.79it/s][A

evaluating Epoch:   6%|[32m▋         [0m| 13/200 [00:07<01:43,
  1.81it/s][A

evaluating Epoch:   7%|[32m▋         [0m| 14/200 [00:07<01:43,
  1.80it/s][A

evaluating Epoch:   8%|[32m▊         [0m| 15/200 [00:08<01:43,
  1.79it/s][A

evaluating Epoch:   8%|[32m▊         [0m| 16/200 [00:09<01:42,
  1.80it/s][A

evaluating Epoch:   8%|[32m▊         [0m| 17/200 [00:09<01:41,
  1.80it/s][A

evaluating Epoch:   9%|[32m▉         [0m| 18/200 [00:10<01:41,
  1.80it/s][A

evaluating Epoch:  10%|[32m▉         [0m| 19/200 [00:10<01:42,
  1.77it/s][A
Training Epoch: 2/3,
 step 248/249 completed (loss: 0.0004284728202037513): : 30876it [51:46,
 19.78it/s]

evaluating Epoch:  10%|[32m█         [0m| 20/200 [00:11<01:42,
  1.76it/s][A

evaluating Epoch:  10%|[32m█         [0m| 21/200 [00:11<01:42,
  1.75it/s][A

evaluating Epoch:  11%|[32m█         [0m| 22/200 [00:12<01:41,
  1.75it/s][A

evaluating Epoch:  12%|[32m█▏        [0m| 23/200 [00:13<01:41,
  1.75it/s][A

evaluating Epoch:  12%|[32m█▏        [0m| 24/200 [00:13<01:41,
  1.74it/s][A

evaluating Epoch:  12%|[32m█▎        [0m| 25/200 [00:14<01:40,
  1.74it/s][A

evaluating Epoch:  13%|[32m█▎        [0m| 26/200 [00:14<01:39,
  1.75it/s][A

evaluating Epoch:  14%|[32m█▎        [0m| 27/200 [00:15<01:37,
  1.78it/s][A

evaluating Epoch:  14%|[32m█▍        [0m| 28/200 [00:15<01:37,
  1.77it/s][A

evaluating Epoch:  14%|[32m█▍        [0m| 29/200 [00:16<01:36,
  1.77it/s][A

evaluating Epoch:  15%|[32m█▌        [0m| 30/200 [00:17<01:35,
  1.78it/s][A

evaluating Epoch:  16%|[32m█▌        [0m| 31/200 [00:17<01:36,
  1.76it/s][A

evaluating Epoch:  16%|[32m█▌        [0m| 32/200 [00:18<01:35,
  1.76it/s][A

evaluating Epoch:  16%|[32m█▋        [0m| 33/200 [00:18<01:34,
  1.76it/s][A

evaluating Epoch:  17%|[32m█▋        [0m| 34/200 [00:19<01:32,
  1.79it/s][A

evaluating Epoch:  18%|[32m█▊        [0m| 35/200 [00:19<01:33,
  1.77it/s][A

evaluating Epoch:  18%|[32m█▊        [0m| 36/200 [00:20<01:31,
  1.78it/s][A

evaluating Epoch:  18%|[32m█▊        [0m| 37/200 [00:20<01:31,
  1.79it/s][A

evaluating Epoch:  19%|[32m█▉        [0m| 38/200 [00:21<01:31,
  1.77it/s][A

evaluating Epoch:  20%|[32m█▉        [0m| 39/200 [00:22<01:31,
  1.76it/s][A

evaluating Epoch:  20%|[32m██        [0m| 40/200 [00:22<01:30,
  1.76it/s][A

evaluating Epoch:  20%|[32m██        [0m| 41/200 [00:23<01:30,
  1.75it/s][A

evaluating Epoch:  21%|[32m██        [0m| 42/200 [00:23<01:30,
  1.75it/s][A

evaluating Epoch:  22%|[32m██▏       [0m| 43/200 [00:24<01:29,
  1.76it/s][A

evaluating Epoch:  22%|[32m██▏       [0m| 44/200 [00:25<01:29,
  1.75it/s][A

evaluating Epoch:  22%|[32m██▎       [0m| 45/200 [00:25<01:27,
  1.77it/s][A

evaluating Epoch:  23%|[32m██▎       [0m| 46/200 [00:26<01:27,
  1.76it/s][A

evaluating Epoch:  24%|[32m██▎       [0m| 47/200 [00:26<01:25,
  1.78it/s][A

evaluating Epoch:  24%|[32m██▍       [0m| 48/200 [00:27<01:24,
  1.80it/s][A

evaluating Epoch:  24%|[32m██▍       [0m| 49/200 [00:27<01:25,
  1.78it/s][A

evaluating Epoch:  25%|[32m██▌       [0m| 50/200 [00:28<01:24,
  1.77it/s][A

evaluating Epoch:  26%|[32m██▌       [0m| 51/200 [00:28<01:24,
  1.77it/s][A

evaluating Epoch:  26%|[32m██▌       [0m| 52/200 [00:29<01:24,
  1.76it/s][A

evaluating Epoch:  26%|[32m██▋       [0m| 53/200 [00:30<01:23,
  1.76it/s][A

evaluating Epoch:  27%|[32m██▋       [0m| 54/200 [00:30<01:21,
  1.78it/s][A

evaluating Epoch:  28%|[32m██▊       [0m| 55/200 [00:31<01:21,
  1.78it/s][A

evaluating Epoch:  28%|[32m██▊       [0m| 56/200 [00:31<01:21,
  1.77it/s][A

evaluating Epoch:  28%|[32m██▊       [0m| 57/200 [00:32<01:20,
  1.77it/s][A

evaluating Epoch:  29%|[32m██▉       [0m| 58/200 [00:32<01:20,
  1.76it/s][A

evaluating Epoch:  30%|[32m██▉       [0m| 59/200 [00:33<01:19,
  1.77it/s][A

evaluating Epoch:  30%|[32m███       [0m| 60/200 [00:34<01:19,
  1.75it/s][A

evaluating Epoch:  30%|[32m███       [0m| 61/200 [00:34<01:18,
  1.76it/s][A

evaluating Epoch:  31%|[32m███       [0m| 62/200 [00:35<01:17,
  1.77it/s][A

evaluating Epoch:  32%|[32m███▏      [0m| 63/200 [00:35<01:17,
  1.76it/s][A

evaluating Epoch:  32%|[32m███▏      [0m| 64/200 [00:36<01:17,
  1.76it/s][A

evaluating Epoch:  32%|[32m███▎      [0m| 65/200 [00:36<01:16,
  1.76it/s][A

evaluating Epoch:  33%|[32m███▎      [0m| 66/200 [00:37<01:15,
  1.77it/s][A

evaluating Epoch:  34%|[32m███▎      [0m| 67/200 [00:38<01:15,
  1.75it/s][A

evaluating Epoch:  34%|[32m███▍      [0m| 68/200 [00:38<01:15,
  1.75it/s][A

evaluating Epoch:  34%|[32m███▍      [0m| 69/200 [00:39<01:14,
  1.77it/s][A

evaluating Epoch:  35%|[32m███▌      [0m| 70/200 [00:39<01:12,
  1.78it/s][A

evaluating Epoch:  36%|[32m███▌      [0m| 71/200 [00:40<01:12,
  1.79it/s][A

evaluating Epoch:  36%|[32m███▌      [0m| 72/200 [00:40<01:11,
  1.78it/s][A

evaluating Epoch:  36%|[32m███▋      [0m| 73/200 [00:41<01:10,
  1.80it/s][A

evaluating Epoch:  37%|[32m███▋      [0m| 74/200 [00:41<01:10,
  1.78it/s][A

evaluating Epoch:  38%|[32m███▊      [0m| 75/200 [00:42<01:10,
  1.78it/s][A

evaluating Epoch:  38%|[32m███▊      [0m| 76/200 [00:43<01:10,
  1.76it/s][A

evaluating Epoch:  38%|[32m███▊      [0m| 77/200 [00:43<01:09,
  1.78it/s][A

evaluating Epoch:  39%|[32m███▉      [0m| 78/200 [00:44<01:08,
  1.79it/s][A

evaluating Epoch:  40%|[32m███▉      [0m| 79/200 [00:44<01:08,
  1.77it/s][A

evaluating Epoch:  40%|[32m████      [0m| 80/200 [00:45<01:07,
  1.78it/s][A

evaluating Epoch:  40%|[32m████      [0m| 81/200 [00:45<01:07,
  1.77it/s][A

evaluating Epoch:  41%|[32m████      [0m| 82/200 [00:46<01:05,
  1.80it/s][A

evaluating Epoch:  42%|[32m████▏     [0m| 83/200 [00:46<01:05,
  1.78it/s][A

evaluating Epoch:  42%|[32m████▏     [0m| 84/200 [00:47<01:05,
  1.77it/s][A

evaluating Epoch:  42%|[32m████▎     [0m| 85/200 [00:48<01:05,
  1.76it/s][A

evaluating Epoch:  43%|[32m████▎     [0m| 86/200 [00:48<01:04,
  1.76it/s][A

evaluating Epoch:  44%|[32m████▎     [0m| 87/200 [00:49<01:03,
  1.78it/s][A

evaluating Epoch:  44%|[32m████▍     [0m| 88/200 [00:49<01:03,
  1.76it/s][A

evaluating Epoch:  44%|[32m████▍     [0m| 89/200 [00:50<01:02,
  1.77it/s][A

evaluating Epoch:  45%|[32m████▌     [0m| 90/200 [00:50<01:02,
  1.77it/s][A

evaluating Epoch:  46%|[32m████▌     [0m| 91/200 [00:51<01:01,
  1.76it/s][A

evaluating Epoch:  46%|[32m████▌     [0m| 92/200 [00:52<01:01,
  1.75it/s][A

evaluating Epoch:  46%|[32m████▋     [0m| 93/200 [00:52<01:01,
  1.74it/s][A

evaluating Epoch:  47%|[32m████▋     [0m| 94/200 [00:53<01:00,
  1.74it/s][A

evaluating Epoch:  48%|[32m████▊     [0m| 95/200 [00:53<00:59,
  1.77it/s][A

evaluating Epoch:  48%|[32m████▊     [0m| 96/200 [00:54<00:58,
  1.78it/s][A

evaluating Epoch:  48%|[32m████▊     [0m| 97/200 [00:54<00:57,
  1.78it/s][A

evaluating Epoch:  49%|[32m████▉     [0m| 98/200 [00:55<00:58,
  1.75it/s][A

evaluating Epoch:  50%|[32m████▉     [0m| 99/200 [00:56<00:57,
  1.75it/s][A

evaluating Epoch:  50%|[32m█████     [0m| 100/200 [00:56<00:56,
  1.77it/s][A

evaluating Epoch:  50%|[32m█████     [0m| 101/200 [00:57<00:55,
  1.78it/s][A

evaluating Epoch:  51%|[32m█████     [0m| 102/200 [00:57<00:54,
  1.78it/s][A

evaluating Epoch:  52%|[32m█████▏    [0m| 103/200 [00:58<00:54,
  1.76it/s][A

evaluating Epoch:  52%|[32m█████▏    [0m| 104/200 [00:58<00:54,
  1.77it/s][A

evaluating Epoch:  52%|[32m█████▎    [0m| 105/200 [00:59<00:53,
  1.77it/s][A

evaluating Epoch:  53%|[32m█████▎    [0m| 106/200 [01:00<00:53,
  1.75it/s][A

evaluating Epoch:  54%|[32m█████▎    [0m| 107/200 [01:00<00:52,
  1.77it/s][A

evaluating Epoch:  54%|[32m█████▍    [0m| 108/200 [01:01<00:51,
  1.77it/s][A

evaluating Epoch:  55%|[32m█████▍    [0m| 109/200 [01:01<00:51,
  1.77it/s][A

evaluating Epoch:  55%|[32m█████▌    [0m| 110/200 [01:02<00:50,
  1.80it/s][A

evaluating Epoch:  56%|[32m█████▌    [0m| 111/200 [01:02<00:49,
  1.79it/s][A

evaluating Epoch:  56%|[32m█████▌    [0m| 112/200 [01:03<00:49,
  1.78it/s][A

evaluating Epoch:  56%|[32m█████▋    [0m| 113/200 [01:03<00:48,
  1.80it/s][A

evaluating Epoch:  57%|[32m█████▋    [0m| 114/200 [01:04<00:47,
  1.81it/s][A

evaluating Epoch:  57%|[32m█████▊    [0m| 115/200 [01:05<00:46,
  1.81it/s][A

evaluating Epoch:  58%|[32m█████▊    [0m| 116/200 [01:05<00:46,
  1.79it/s][A

evaluating Epoch:  58%|[32m█████▊    [0m| 117/200 [01:06<00:46,
  1.78it/s][A

evaluating Epoch:  59%|[32m█████▉    [0m| 118/200 [01:06<00:45,
  1.79it/s][A

evaluating Epoch:  60%|[32m█████▉    [0m| 119/200 [01:07<00:44,
  1.81it/s][A

evaluating Epoch:  60%|[32m██████    [0m| 120/200 [01:07<00:44,
  1.78it/s][A

evaluating Epoch:  60%|[32m██████    [0m| 121/200 [01:08<00:44,
  1.77it/s][A

evaluating Epoch:  61%|[32m██████    [0m| 122/200 [01:08<00:44,
  1.76it/s][A

evaluating Epoch:  62%|[32m██████▏   [0m| 123/200 [01:09<00:43,
  1.78it/s][A

evaluating Epoch:  62%|[32m██████▏   [0m| 124/200 [01:10<00:43,
  1.76it/s][A

evaluating Epoch:  62%|[32m██████▎   [0m| 125/200 [01:10<00:42,
  1.76it/s][A

evaluating Epoch:  63%|[32m██████▎   [0m| 126/200 [01:11<00:42,
  1.75it/s][A

evaluating Epoch:  64%|[32m██████▎   [0m| 127/200 [01:11<00:41,
  1.75it/s][A

evaluating Epoch:  64%|[32m██████▍   [0m| 128/200 [01:12<00:41,
  1.74it/s][A

evaluating Epoch:  64%|[32m██████▍   [0m| 129/200 [01:12<00:40,
  1.77it/s][A

evaluating Epoch:  65%|[32m██████▌   [0m| 130/200 [01:13<00:39,
  1.76it/s][A

evaluating Epoch:  66%|[32m██████▌   [0m| 131/200 [01:14<00:38,
  1.78it/s][A

evaluating Epoch:  66%|[32m██████▌   [0m| 132/200 [01:14<00:38,
  1.77it/s][A

evaluating Epoch:  66%|[32m██████▋   [0m| 133/200 [01:15<00:37,
  1.79it/s][A

evaluating Epoch:  67%|[32m██████▋   [0m| 134/200 [01:15<00:37,
  1.78it/s][A

evaluating Epoch:  68%|[32m██████▊   [0m| 135/200 [01:16<00:36,
  1.79it/s][A

evaluating Epoch:  68%|[32m██████▊   [0m| 136/200 [01:16<00:35,
  1.78it/s][A

evaluating Epoch:  68%|[32m██████▊   [0m| 137/200 [01:17<00:35,
  1.77it/s][A

evaluating Epoch:  69%|[32m██████▉   [0m| 138/200 [01:18<00:35,
  1.76it/s][A

evaluating Epoch:  70%|[32m██████▉   [0m| 139/200 [01:18<00:34,
  1.75it/s][A

evaluating Epoch:  70%|[32m███████   [0m| 140/200 [01:19<00:33,
  1.77it/s][A

evaluating Epoch:  70%|[32m███████   [0m| 141/200 [01:19<00:33,
  1.76it/s][A

evaluating Epoch:  71%|[32m███████   [0m| 142/200 [01:20<00:32,
  1.76it/s][A

evaluating Epoch:  72%|[32m███████▏  [0m| 143/200 [01:20<00:32,
  1.76it/s][A

evaluating Epoch:  72%|[32m███████▏  [0m| 144/200 [01:21<00:31,
  1.79it/s][A

evaluating Epoch:  72%|[32m███████▎  [0m| 145/200 [01:21<00:30,
  1.80it/s][A

evaluating Epoch:  73%|[32m███████▎  [0m| 146/200 [01:22<00:30,
  1.79it/s][A

evaluating Epoch:  74%|[32m███████▎  [0m| 147/200 [01:23<00:29,
  1.80it/s][A

evaluating Epoch:  74%|[32m███████▍  [0m| 148/200 [01:23<00:29,
  1.78it/s][A

evaluating Epoch:  74%|[32m███████▍  [0m| 149/200 [01:24<00:28,
  1.78it/s][A

evaluating Epoch:  75%|[32m███████▌  [0m| 150/200 [01:24<00:28,
  1.76it/s][A

evaluating Epoch:  76%|[32m███████▌  [0m| 151/200 [01:25<00:27,
  1.76it/s][A

evaluating Epoch:  76%|[32m███████▌  [0m| 152/200 [01:25<00:27,
  1.77it/s][A

evaluating Epoch:  76%|[32m███████▋  [0m| 153/200 [01:26<00:26,
  1.76it/s][A

evaluating Epoch:  77%|[32m███████▋  [0m| 154/200 [01:27<00:26,
  1.75it/s][A

evaluating Epoch:  78%|[32m███████▊  [0m| 155/200 [01:27<00:25,
  1.76it/s][A

evaluating Epoch:  78%|[32m███████▊  [0m| 156/200 [01:28<00:24,
  1.77it/s][A

evaluating Epoch:  78%|[32m███████▊  [0m| 157/200 [01:28<00:24,
  1.78it/s][A

evaluating Epoch:  79%|[32m███████▉  [0m| 158/200 [01:29<00:23,
  1.78it/s][A

evaluating Epoch:  80%|[32m███████▉  [0m| 159/200 [01:29<00:22,
  1.79it/s][A

evaluating Epoch:  80%|[32m████████  [0m| 160/200 [01:30<00:22,
  1.78it/s][A

evaluating Epoch:  80%|[32m████████  [0m| 161/200 [01:31<00:21,
  1.78it/s][A

evaluating Epoch:  81%|[32m████████  [0m| 162/200 [01:31<00:21,
  1.81it/s][A

evaluating Epoch:  82%|[32m████████▏ [0m| 163/200 [01:32<00:20,
  1.81it/s][A

evaluating Epoch:  82%|[32m████████▏ [0m| 164/200 [01:32<00:19,
  1.81it/s][A

evaluating Epoch:  82%|[32m████████▎ [0m| 165/200 [01:33<00:19,
  1.81it/s][A

evaluating Epoch:  83%|[32m████████▎ [0m| 166/200 [01:33<00:18,
  1.81it/s][A

evaluating Epoch:  84%|[32m████████▎ [0m| 167/200 [01:34<00:18,
  1.79it/s][A

evaluating Epoch:  84%|[32m████████▍ [0m| 168/200 [01:34<00:17,
  1.78it/s][A

evaluating Epoch:  84%|[32m████████▍ [0m| 169/200 [01:35<00:17,
  1.78it/s][A

evaluating Epoch:  85%|[32m████████▌ [0m| 170/200 [01:36<00:16,
  1.79it/s][A

evaluating Epoch:  86%|[32m████████▌ [0m| 171/200 [01:36<00:16,
  1.79it/s][A

evaluating Epoch:  86%|[32m████████▌ [0m| 172/200 [01:37<00:15,
  1.82it/s][A

evaluating Epoch:  86%|[32m████████▋ [0m| 173/200 [01:37<00:14,
  1.83it/s][A

evaluating Epoch:  87%|[32m████████▋ [0m| 174/200 [01:38<00:14,
  1.82it/s][A

evaluating Epoch:  88%|[32m████████▊ [0m| 175/200 [01:38<00:13,
  1.83it/s][A

evaluating Epoch:  88%|[32m████████▊ [0m| 176/200 [01:39<00:12,
  1.85it/s][A

evaluating Epoch:  88%|[32m████████▊ [0m| 177/200 [01:39<00:12,
  1.84it/s][A

evaluating Epoch:  89%|[32m████████▉ [0m| 178/200 [01:40<00:11,
  1.85it/s][A

evaluating Epoch:  90%|[32m████████▉ [0m| 179/200 [01:40<00:11,
  1.83it/s][A

evaluating Epoch:  90%|[32m█████████ [0m| 180/200 [01:41<00:10,
  1.84it/s][A

evaluating Epoch:  90%|[32m█████████ [0m| 181/200 [01:41<00:10,
  1.85it/s][A

evaluating Epoch:  91%|[32m█████████ [0m| 182/200 [01:42<00:09,
  1.82it/s][A

evaluating Epoch:  92%|[32m█████████▏[0m| 183/200 [01:43<00:09,
  1.83it/s][A

evaluating Epoch:  92%|[32m█████████▏[0m| 184/200 [01:43<00:08,
  1.82it/s][A

evaluating Epoch:  92%|[32m█████████▎[0m| 185/200 [01:44<00:08,
  1.83it/s][A

evaluating Epoch:  93%|[32m█████████▎[0m| 186/200 [01:44<00:07,
  1.81it/s][A

evaluating Epoch:  94%|[32m█████████▎[0m| 187/200 [01:45<00:07,
  1.80it/s][A

evaluating Epoch:  94%|[32m█████████▍[0m| 188/200 [01:45<00:06,
  1.81it/s][A

evaluating Epoch:  94%|[32m█████████▍[0m| 189/200 [01:46<00:06,
  1.81it/s][A

evaluating Epoch:  95%|[32m█████████▌[0m| 190/200 [01:46<00:05,
  1.81it/s][A

evaluating Epoch:  96%|[32m█████████▌[0m| 191/200 [01:47<00:04,
  1.82it/s][A

evaluating Epoch:  96%|[32m█████████▌[0m| 192/200 [01:48<00:04,
  1.81it/s][A

evaluating Epoch:  96%|[32m█████████▋[0m| 193/200 [01:48<00:03,
  1.81it/s][A

evaluating Epoch:  97%|[32m█████████▋[0m| 194/200 [01:49<00:03,
  1.79it/s][A

evaluating Epoch:  98%|[32m█████████▊[0m| 195/200 [01:49<00:02,
  1.81it/s][A

evaluating Epoch:  98%|[32m█████████▊[0m| 196/200 [01:50<00:02,
  1.82it/s][A

evaluating Epoch:  98%|[32m█████████▊[0m| 197/200 [01:50<00:01,
  1.81it/s][A

evaluating Epoch:  99%|[32m█████████▉[0m| 198/200 [01:51<00:01,
  1.79it/s][A

evaluating Epoch: 100%|[32m█████████▉[0m| 199/200 [01:51<00:00,
  1.80it/s][A

evaluating Epoch: 100%|[32m██████████[0m| 200/200 [01:52<00:00,
  1.80it/s][A
evaluating Epoch: 100%|[32m██████████[0m| 200/200 [01:52<00:00,
  1.78it/s]

Training Epoch: 2/3,
 step 248/249 completed (loss: 0.0004284728202037513): : 30876it [53:28,
  9.62it/s]
 eval_ppl=tensor(1.0522,
 device='cuda:0') eval_epoch_loss=tensor(0.0508,
 device='cuda:0')
we are about to save the PEFT modules
PEFT modules are saved in ./output_dir_whole_dataset directory
best eval loss on epoch 2 is 0.05084925889968872
Epoch 3: train_perplexity=1.0486,
 train_epoch_loss=0.0475,
 epcoh time 3095.8347171559s
Key: avg_train_prep,
 Value: 1.1313245296478271
Key: avg_train_loss,
 Value: 0.11970707029104233
Key: avg_eval_prep,
 Value: 1.101104974746704
Key: avg_eval_loss,
 Value: 0.09549128264188766
Key: avg_epoch_time,
 Value: 3096.7982691169404
Key: avg_checkpoint_time,
 Value: 0.037331069043527045

  0%|          | 0/1 [00:00<?,
 ?it/s]
100%|██████████| 1/1 [00:01<00:00,
  1.65s/it]
100%|██████████| 1/1 [00:01<00:00,
  1.65s/it]

Training config is:  {'__module__': 'llama_recipes.configs.training',
 '__annotations__': {'model_name': <class 'str'>,
 'enable_fsdp': <class 'bool'>,
 'low_cpu_fsdp': <class 'bool'>,
 'run_validation': <class 'bool'>,
 'batch_size_training': <class 'int'>,
 'gradient_accumulation_steps': <class 'int'>,
 'num_epochs': <class 'int'>,
 'num_workers_dataloader': <class 'int'>,
 'lr': <class 'float'>,
 'weight_decay': <class 'float'>,
 'gamma': <class 'float'>,
 'seed': <class 'int'>,
 'use_fp16': <class 'bool'>,
 'mixed_precision': <class 'bool'>,
 'val_batch_size': <class 'int'>,
 'peft_method': <class 'str'>,
 'use_peft': <class 'bool'>,
 'output_dir': <class 'str'>,
 'freeze_layers': <class 'bool'>,
 'num_freeze_layers': <class 'int'>,
 'quantization': <class 'bool'>,
 'one_gpu': <class 'bool'>,
 'save_model': <class 'bool'>,
 'dist_checkpoint_root_folder': <class 'str'>,
 'dist_checkpoint_folder': <class 'str'>,
 'save_optimizer': <class 'bool'>,
 'use_fast_kernels': <class 'bool'>},
 'model_name': 'meta-llama/Llama-2-7b-hf',
 'enable_fsdp': False,
 'low_cpu_fsdp': False,
 'run_validation': True,
 'batch_size_training': 8,
 'gradient_accumulation_steps': 1,
 'num_epochs': 3,
 'num_workers_dataloader': 1,
 'lr': 0.0001,
 'weight_decay': 0.0,
 'gamma': 0.85,
 'seed': 42,
 'use_fp16': False,
 'mixed_precision': True,
 'val_batch_size': 1,
 'dataset': 'custom_dataset',
 'peft_method': 'lora',
 'use_peft': True,
 'output_dir': './random_check',
 'freeze_layers': False,
 'num_freeze_layers': 1,
 'quantization': True,
 'one_gpu': False,
 'save_model': True,
 'dist_checkpoint_root_folder': 'PATH/to/save/FSDP/model',
 'dist_checkpoint_folder': 'fine-tuned',
 'save_optimizer': False,
 'use_fast_kernels': False,
 '__dict__': <attribute '__dict__' of 'train_config' objects>,
 '__weakref__': <attribute '__weakref__' of 'train_config' objects>,
 '__doc__': "train_config(model_name: str = 'PATH/to/LLAMA/7B',
 enable_fsdp: bool = False,
 low_cpu_fsdp: bool = False,
 run_validation: bool = True,
 batch_size_training: int = 4,
 gradient_accumulation_steps: int = 1,
 num_epochs: int = 3,
 num_workers_dataloader: int = 1,
 lr: float = 0.0001,
 weight_decay: float = 0.0,
 gamma: float = 0.85,
 seed: int = 42,
 use_fp16: bool = False,
 mixed_precision: bool = True,
 val_batch_size: int = 1,
 peft_method: str = 'lora',
 use_peft: bool = False,
 output_dir: str = 'PATH/to/save/PEFT/model',
 freeze_layers: bool = False,
 num_freeze_layers: int = 1,
 quantization: bool = False,
 one_gpu: bool = False,
 save_model: bool = True,
 dist_checkpoint_root_folder: str = 'PATH/to/save/FSDP/model',
 dist_checkpoint_folder: str = 'fine-tuned',
 save_optimizer: bool = False,
 use_fast_kernels: bool = False)",
 '__dataclass_params__': _DataclassParams(init=True,
repr=True,
eq=True,
order=False,
unsafe_hash=False,
frozen=False),
 '__dataclass_fields__': {'model_name': Field(name='model_name',
type=<class 'str'>,
default='PATH/to/LLAMA/7B',
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'enable_fsdp': Field(name='enable_fsdp',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'low_cpu_fsdp': Field(name='low_cpu_fsdp',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'run_validation': Field(name='run_validation',
type=<class 'bool'>,
default=True,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'batch_size_training': Field(name='batch_size_training',
type=<class 'int'>,
default=4,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'gradient_accumulation_steps': Field(name='gradient_accumulation_steps',
type=<class 'int'>,
default=1,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'num_epochs': Field(name='num_epochs',
type=<class 'int'>,
default=3,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'num_workers_dataloader': Field(name='num_workers_dataloader',
type=<class 'int'>,
default=1,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'lr': Field(name='lr',
type=<class 'float'>,
default=0.0001,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'weight_decay': Field(name='weight_decay',
type=<class 'float'>,
default=0.0,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'gamma': Field(name='gamma',
type=<class 'float'>,
default=0.85,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'seed': Field(name='seed',
type=<class 'int'>,
default=42,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'use_fp16': Field(name='use_fp16',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'mixed_precision': Field(name='mixed_precision',
type=<class 'bool'>,
default=True,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'val_batch_size': Field(name='val_batch_size',
type=<class 'int'>,
default=1,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'peft_method': Field(name='peft_method',
type=<class 'str'>,
default='lora',
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'use_peft': Field(name='use_peft',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'output_dir': Field(name='output_dir',
type=<class 'str'>,
default='PATH/to/save/PEFT/model',
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'freeze_layers': Field(name='freeze_layers',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'num_freeze_layers': Field(name='num_freeze_layers',
type=<class 'int'>,
default=1,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'quantization': Field(name='quantization',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'one_gpu': Field(name='one_gpu',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'save_model': Field(name='save_model',
type=<class 'bool'>,
default=True,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'dist_checkpoint_root_folder': Field(name='dist_checkpoint_root_folder',
type=<class 'str'>,
default='PATH/to/save/FSDP/model',
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'dist_checkpoint_folder': Field(name='dist_checkpoint_folder',
type=<class 'str'>,
default='fine-tuned',
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'save_optimizer': Field(name='save_optimizer',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'use_fast_kernels': Field(name='use_fast_kernels',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD)},
 '__init__': <function __create_fn__.<locals>.__init__ at 0x7ff5e557f820>,
 '__repr__': <function __create_fn__.<locals>.__repr__ at 0x7ff5e55799d0>,
 '__eq__': <function __create_fn__.<locals>.__eq__ at 0x7ff5e557f700>,
 '__hash__': None}
,




FSDP config is:  {'__module__': 'llama_recipes.configs.fsdp',
 '__annotations__': {'mixed_precision': <class 'bool'>,
 'use_fp16': <class 'bool'>,
 'sharding_strategy': <enum 'ShardingStrategy'>,
 'checkpoint_type': <enum 'StateDictType'>,
 'fsdp_activation_checkpointing': <class 'bool'>,
 'pure_bf16': <class 'bool'>,
 'optimizer': <class 'str'>},
 'mixed_precision': True,
 'use_fp16': False,
 'sharding_strategy': <ShardingStrategy.FULL_SHARD: 1>,
 'checkpoint_type': <StateDictType.SHARDED_STATE_DICT: 3>,
 'fsdp_activation_checkpointing': True,
 'pure_bf16': False,
 'optimizer': 'AdamW',
 '__dict__': <attribute '__dict__' of 'fsdp_config' objects>,
 '__weakref__': <attribute '__weakref__' of 'fsdp_config' objects>,
 '__doc__': "fsdp_config(mixed_precision: bool = True,
 use_fp16: bool = False,
 sharding_strategy: torch.distributed.fsdp.api.ShardingStrategy = <ShardingStrategy.FULL_SHARD: 1>,
 checkpoint_type: torch.distributed.fsdp.api.StateDictType = <StateDictType.SHARDED_STATE_DICT: 3>,
 fsdp_activation_checkpointing: bool = True,
 pure_bf16: bool = False,
 optimizer: str = 'AdamW')",
 '__dataclass_params__': _DataclassParams(init=True,
repr=True,
eq=True,
order=False,
unsafe_hash=False,
frozen=False),
 '__dataclass_fields__': {'mixed_precision': Field(name='mixed_precision',
type=<class 'bool'>,
default=True,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'use_fp16': Field(name='use_fp16',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'sharding_strategy': Field(name='sharding_strategy',
type=<enum 'ShardingStrategy'>,
default=<ShardingStrategy.FULL_SHARD: 1>,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'checkpoint_type': Field(name='checkpoint_type',
type=<enum 'StateDictType'>,
default=<StateDictType.SHARDED_STATE_DICT: 3>,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'fsdp_activation_checkpointing': Field(name='fsdp_activation_checkpointing',
type=<class 'bool'>,
default=True,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'pure_bf16': Field(name='pure_bf16',
type=<class 'bool'>,
default=False,
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD),
 'optimizer': Field(name='optimizer',
type=<class 'str'>,
default='AdamW',
default_factory=<dataclasses._MISSING_TYPE object at 0x7ff6e490d6d0>,
init=True,
repr=True,
hash=None,
compare=True,
metadata=mappingproxy({}),
_field_type=_FIELD)},
 '__init__': <function __create_fn__.<locals>.__init__ at 0x7ff5e557f4c0>,
 '__repr__': <function __create_fn__.<locals>.__repr__ at 0x7ff5e557f310>,
 '__eq__': <function __create_fn__.<locals>.__eq__ at 0x7ff5e557f5e0>,
 '__hash__': None}
/home/anmol/anaconda3/envs/wizard_coder/lib/python3.8/site-packages/torch/cuda/memory.py:329: FutureWarning: torch.cuda.reset_max_memory_allocated now calls torch.cuda.reset_peak_memory_stats,
 which resets /all/ peak memory stats.
