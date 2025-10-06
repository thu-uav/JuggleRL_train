CUDA_VISIBLE_DEVICES=1 python ../train.py headless=true \
    total_frames=2000000000 \
    task=Hover \
    task.drone_model=air \
    task.action_transform=PIDrate_FM \
    task.sim.dt=0.02 \
    task.env.num_envs=4096 \
    eval_interval=-1 \
    save_interval=200 \
    wandb.run_name=new_hover_sim2real \
    # policy_checkpoint_path="/home/chenyinuo/OmniDrones/scripts/shell/wandb/run-20250222_105703-7q6b6d1k/files/checkpoint_839122944.pt"\
    # wandb.mode=disabled # debug