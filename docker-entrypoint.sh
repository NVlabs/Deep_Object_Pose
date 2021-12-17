cd /workspace/dope/scripts/nvisii_data_gen
python generate_dataset.py

cd /workspace/dope/scripts/train2
python -m torch.distributed.launch --nproc_per_node=1 train.py train
