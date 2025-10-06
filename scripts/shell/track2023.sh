CUDA_VISIBLE_DEVICES=7 python ../train.py headless=true \
    total_frames=20000000000 \
    task=Track2023 \
    task.drone_model=air \
    task.action_transform=PIDrate_FM \
    task.sim.dt=0.02 \
    task.env.num_envs=8192 \
    eval_interval=-1 \
    save_interval=200 \
    wandb.run_name=changed_param \
    algo.critic_input=state \
    # wandb.mode=disabled # debug
    # policy_checkpoint_path="checkpoint_19791872.pt"\