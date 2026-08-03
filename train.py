import os
import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


def train(args):
    configure_logger()

    # Time-slicing orchestrator client setup
    sampler_client = None
    trainer_client = None
    job_id = getattr(args, "timeslice_job_id", None) or os.getenv("TIMESLICE_JOB_ID", "slime-job-default")

    if getattr(args, "enable_timeslice", False):
        from timeslice import OrchestratorClient
        addr = getattr(args, "timeslice_orchestrator_addr", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
        sampler_group = getattr(args, "timeslice_sampler_group", "samplers")
        trainer_group = getattr(args, "timeslice_trainer_group", "trainers")

        print(f"[TimeSlice] Initializing OrchestratorClient (addr={addr}, job_id={job_id})...")
        sampler_client = OrchestratorClient(target=addr, job_id=job_id, group_id=sampler_group)
        trainer_client = OrchestratorClient(target=addr, job_id=job_id, group_id=trainer_group)

    if trainer_client is not None:
        print("[TimeSlice] Acquiring Trainer GPU Grant before placement group allocation...")
        trainer_client.acquire()

    def _create_rollout_group():
        if sampler_client is not None:
            print("[TimeSlice] Acquiring Sampler GPU Grant in rollout thread before allocation...")
            sampler_client.acquire()
        return create_placement_groups(args, role="rollout")

    if getattr(args, "enable_timeslice", False):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_actor = executor.submit(create_placement_groups, args, role="actor")
            f_rollout = executor.submit(_create_rollout_group)
            pgs = f_actor.result()
            pgs_rollout = f_rollout.result()
        pgs.update(pgs_rollout)
    else:
        pgs = create_placement_groups(args)

    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    if trainer_client is not None:
        print("[TimeSlice] Yielding Trainer GPU Grant after initial weight sync.")
        trainer_client.release()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if sampler_client:
            print(f"[TimeSlice] Yielding Sampler GPU Grant for job {job_id}...")
            sampler_client.release()

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if trainer_client:
            print(f"[TimeSlice] Acquiring Trainer GPU Grant for job {job_id}...")
            trainer_client.acquire()

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            save(rollout_id)

        offload_train(actor_trains_this_step)

        if sampler_client:
            print(f"[TimeSlice] Acquiring Sampler GPU Grant for weight update (job {job_id})...")
            sampler_client.acquire()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if trainer_client:
            print(f"[TimeSlice] Yielding Trainer GPU Grant after weight update for job {job_id}...")
            trainer_client.release()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    if sampler_client:
        print(f"[TimeSlice] Releasing final Sampler GPU Grant for job {job_id}...")
        sampler_client.release()

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
