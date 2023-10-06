## check all envs work
#for policy in DoubleGum
#do
#  for env in BipedalWalker-v3 cheetah-run Ant-v4 metaworld_button-press-v2
#  do
#    eval "\
#    python main_cont.py --seed -1 \
#    --env ${env} \
#    --policy ${policy} \
#    --max-timesteps 100 --start-timesteps 10 --folder testdelete \
#    "
#  done
#done
#
## check all baselines work
#for policy in MoG-DDPG DDPG TD3 QR-DDPG SAC XQL IQL SACLite
#do
#  for env in cheetah-run
#  do
#    eval "\
#    python main_cont.py --seed -1 \
#    --env ${env} \
#    --policy ${policy} \
#    --max-timesteps 100 --start-timesteps 10 --folder testdelete \
#    "
#  done
#done

for policy in DoubleGum DDQN DQN DuellingDDQN DuellingDQN
do
  for env in CartPole-v1
  do
    eval "\
    python main_disc.py --seed -1 \
    --env ${env} \
    --policy ${policy} \
    --max-timesteps 100 --start-timesteps 10 --folder testdelete \
    "
  done
done


#BipedalWalker-v3 BipedalWalkerHardcore-v3 \
#  humanoid-run acrobot-swingup finger-turn_hard cheetah-run quadruped-run dog-run hopper-hop swimmer-swimmer15 fish-swim reacher-hard walker-run \
#  Walker2d-v4 Hopper-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4 \
#  metaworld_button-press-v2 metaworld_door-open-v2 metaworld_drawer-close-v2 metaworld_drawer-open-v2 metaworld_hammer-v2 \
#  metaworld_peg-insert-side-v2 metaworld_pick-place-v2 metaworld_push-v2 metaworld_reach-v2 metaworld_window-open-v2 \
#  metaworld_window-close-v2 metaworld_basketball-v2 metaworld_dial-turn-v2 metaworld_sweep-into-v2 metaworld_assembly-v2
