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
        sampler_group = getattr(args, "timeslice_sampler_group", "group-slime-sampler")
        trainer_group = getattr(args, "timeslice_trainer_group", "group-slime-trainer")

        print(f"[TimeSlice] Initializing OrchestratorClient (addr={addr}, job_id={job_id})...")
        sampler_client = OrchestratorClient(target=addr, job_id=job_id, group_id=sampler_group)
        trainer_client = OrchestratorClient(target=addr, job_id=job_id, group_id=trainer_group)

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    if sampler_client is not None:
        print("[TimeSlice] Acquiring lock for Sampler (SGLang) initialization...")
        sampler_client.acquire()

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    if sampler_client is not None:
        # Note: create_rollout_manager internally calls rollout_manager.offload()
        # synchronously if args.offload_rollout is True. Calling it again here
        # would trigger a double-pause crash in torch_memory_saver.
        print("[TimeSlice] Releasing lock for Sampler initialization.")
        sampler_client.release()

    # create the actor and critic models
    if trainer_client is not None:
        print("[TimeSlice] Acquiring lock for Trainer (Megatron) initialization...")
        trainer_client.acquire()

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if trainer_client is not None:
        print("[TimeSlice] Releasing lock for Trainer initialization.")
        trainer_client.release()

    # Initial weight sync. SGLang and Megatron are currently offloaded.
    # We must acquire both locks because we will wake them up.
    #
    # DEADLOCK PREVENTION: We must ALWAYS acquire locks in the same order:
    # Trainer first, then Sampler.
    # During the training loop, the job naturally holds Trainer (for training)
    # and then acquires Sampler (for weight update). To avoid circular wait
    # deadlocks, this startup sync must also follow Trainer-first order.
    #
    # NOTE: We do not need both locks between rollout and training because
    # rollout data is buffered in Host RAM (CPU memory) after SGLang offloads,
    # breaking the direct GPU-to-GPU dependency during data loading.
    if trainer_client is not None:
        print("[TimeSlice] Acquiring Trainer GPU Grant for initial weight sync...")
        trainer_client.acquire()
    if sampler_client is not None:
        print("[TimeSlice] Acquiring Sampler GPU Grant for initial weight sync...")
        sampler_client.acquire()

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    # This will internally wake up and sleep Megatron
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        # Onload KV cache so SGLang is fully ready for rollout
        ray.get(rollout_manager.onload_kv.remote())

    # Megatron is now offloaded. SGLang is onloaded.
    # Release Trainer lock, but KEEP Sampler lock for rollout generation!
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

        # ---------------------------------------------------------
        # Phase 1: Rollout Generation (Rollout GPU Group)
        # ---------------------------------------------------------
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if sampler_client:
            print(f"[TimeSlice] Yielding Sampler GPU Grant for job {job_id}...")
            sampler_client.release()

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        # ---------------------------------------------------------
        # Phase 2: Megatron-LM Policy Training (Trainer GPU Group)
        # ---------------------------------------------------------
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

        # Acquire Sampler lock before onloading SGLang weights.
        # This follows the global Trainer-first order (we already hold Trainer lock from training).
        if sampler_client:
            print(f"[TimeSlice] Acquiring Sampler GPU Grant for weight update (job {job_id})...")
            sampler_client.acquire()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())

        # Update weights (internally wakes up and sleeps Megatron Trainer)
        # We must hold Trainer lock here because Trainer is active during transfer.
        actor_model.update_weights()

        if args.offload_rollout:
            # Onload KV cache so SGLang is fully ready for rollout in the next epoch
            ray.get(rollout_manager.onload_kv.remote())

        # Trainer is offloaded. Release Trainer lock.
        # KEEP Sampler lock active to protect SGLang memory on the GPU!
        if trainer_client:
            print(f"[TimeSlice] Yielding Trainer GPU Grant for job {job_id}...")
            trainer_client.release()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    # After loop cleanup: SGLang is currently onloaded and Sampler lock is held.
    # We must release it before exiting.
    if sampler_client:
        print("[TimeSlice] Releasing Sampler GPU Grant after training completion.")
        sampler_client.release()

    if sampler_client:
        sampler_client.close()
    if trainer_client:
        trainer_client.close()

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
