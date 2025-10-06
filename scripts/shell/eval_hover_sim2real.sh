CUDA_VISIBLE_DEVICES=0 python ../train.py headless=true \
    total_frames=500000000 \
    task=Hover \
    task.drone_model=air \
    task.action_transform=PIDrate_FM \
    task.sim.dt=0.02 \
    task.env.num_envs=1 \
    eval_interval=30 \
    save_interval=30 \
    only_eval=true \
    only_eval_one_traj=true \
    eval_type=log_all \
    policy_checkpoint_path="wandb/checkpoint_final.pt" \
    wandb.run_name=eval_new_hover \
    # wandb.mode=disabled # debug
    # policy_checkpoint_path="checkpoint_19791872.pt"\