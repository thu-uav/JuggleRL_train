CUDA_VISIBLE_DEVICES=7 python ../train.py headless=true \
    total_frames=20000000000 \
    task=Track2023 \
    task.drone_model=air \
    task.action_transform=PIDrate_FM \
    task.sim.dt=0.02 \
    task.env.num_envs=1 \
    eval_interval=200 \
    save_interval=200 \
    wandb.run_name=changed_param \
    only_eval=true \
    only_eval_one_traj=false \
    task.use_eval=1 \
    policy_checkpoint_path="wandb/checkpoint_1625554944.pt" \
    wandb.run_name=track_eval \
    # eval_type=log_all \

    # wandb.mode=disabled # debug
    # policy_checkpoint_path="checkpoint_19791872.pt"\