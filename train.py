import atexit
import os
import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import RewardConvergenceDetector, should_run_periodic_action


def train(args):
    configure_logger()

    # Time-slicing orchestrator client setup
    sampler_client = None
    trainer_client = None
    job_id = getattr(args, "timeslice_job_id", None) or os.getenv("TIMESLICE_JOB_ID", "slime-job-default")

    if getattr(args, "enable_timeslice", False):
        from timeslice import TimeSliceOrchestratorClient as OrchestratorClient
        addr = getattr(args, "timeslice_orchestrator_addr", "timeslice-acceleratororchestrator.timeslice-system.svc.cluster.local:50051")
        sampler_group = getattr(args, "timeslice_sampler_group", "samplers")
        trainer_group = getattr(args, "timeslice_trainer_group", "trainers")

        print(f"[TimeSlice] Initializing OrchestratorClient (addr={addr}, job_id={job_id})...")
        sampler_client = OrchestratorClient(target=addr, job_id=job_id, group_id=sampler_group)
        trainer_client = OrchestratorClient(target=addr, job_id=job_id, group_id=trainer_group)

        # Safety net: never exit (early stop, exception, sys.exit) while holding a
        # GPU grant, or every other job in the group blocks forever in acquire().
        # release() is idempotent, so releasing an already-yielded grant is a no-op.
        def _release_grants_at_exit():
            for role, client in (("Sampler", sampler_client), ("Trainer", trainer_client)):
                try:
                    client.release()
                except Exception as e:
                    print(f"[TimeSlice] Failed to release {role} GPU Grant at exit: {e}")

        atexit.register(_release_grants_at_exit)

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

    def save(rollout_id, force_sync=False):
        force_sync = force_sync or rollout_id == args.num_rollout - 1
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if actor_trains_this_step:
            actor_model.save_model(
                rollout_id,
                force_sync=force_sync,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=force_sync,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    convergence_detector = None
    if args.early_stop_window is not None:
        convergence_detector = RewardConvergenceDetector(
            args.early_stop_window, args.early_stop_threshold, min_reward=args.early_stop_min_reward
        )

    def check_convergence(rollout_id):
        """Feed this step's mean reward to the detector; True means stop training."""
        if convergence_detector is None:
            return False
        reward_mean = ray.get(rollout_manager.get_rollout_reward_mean.remote(rollout_id))
        if not convergence_detector.update(reward_mean):
            return False
        min_msg = f" (mean {convergence_detector.last_mean:.4f} >= {args.early_stop_min_reward})" if args.early_stop_min_reward is not None else ""
        print(
            f"[Early Stop] Reward converged at rollout {rollout_id}: "
            f"std {convergence_detector.last_std:.6f} < {args.early_stop_threshold}{min_msg} "
            f"over the last {args.early_stop_window} steps."
        )
        return True

    if args.async_off_policy_limit is not None:
        staleness_limit = args.async_off_policy_limit
        next_rollout_id = args.start_rollout_id
        curr_train_id = args.start_rollout_id
        pending_rollouts = []  # List of (rollout_id, ObjectRef)

        print(f"[Async Pipeline] Starting async RL loop (start={args.start_rollout_id}, total={args.num_rollout}, staleness_limit={staleness_limit})...")

        while curr_train_id < args.num_rollout:
            # 1. Producer: submit rollout generation up to staleness_limit
            while next_rollout_id < args.num_rollout and (next_rollout_id - curr_train_id) <= staleness_limit:
                if args.eval_interval is not None and next_rollout_id == 0 and not args.skip_eval_before_train:
                    ray.get(rollout_manager.eval.remote(next_rollout_id))

                if sampler_client:
                    print(f"[TimeSlice] Acquiring Sampler GPU Grant for async rollout {next_rollout_id} (job {job_id})...")
                    sampler_client.acquire()

                if args.offload_rollout:
                    ray.get(rollout_manager.onload_weights.remote())
                    ray.get(rollout_manager.onload_kv.remote())

                print(f"[Async Pipeline] Submitting rollout {next_rollout_id} for generation...")
                f_gen = rollout_manager.generate.remote(next_rollout_id)
                pending_rollouts.append((next_rollout_id, f_gen))
                next_rollout_id += 1

                if (next_rollout_id - curr_train_id) > staleness_limit or next_rollout_id >= args.num_rollout:
                    break

            # 2. Consumer: wait for next ready batch
            if not pending_rollouts:
                if trainer_client:
                    print(f"[TimeSlice] Yielding Trainer GPU Grant while pipeline is empty (job {job_id})...")
                    trainer_client.release()
                continue

            r_id, f_gen = pending_rollouts.pop(0)
            rollout_data_ref = ray.get(f_gen)

            if args.offload_rollout:
                ray.get(rollout_manager.offload.remote())

            if sampler_client:
                print(f"[TimeSlice] Yielding Sampler GPU Grant after rollout {r_id} generation (job {job_id})...")
                sampler_client.release()

            # 3. Train on the batch
            actor_trains_this_step = (not args.use_critic) or r_id >= args.num_critic_only_steps

            if trainer_client:
                print(f"[TimeSlice] Acquiring Trainer GPU Grant for training rollout {r_id} (job {job_id})...")
                trainer_client.acquire()

            print(f"[Async Pipeline] Training on rollout {r_id}...")
            if args.use_critic:
                value_refs = critic_model.async_train(r_id, rollout_data_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(r_id, rollout_data_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(r_id, rollout_data_ref))

            converged = check_convergence(r_id)

            if should_run_periodic_action(r_id, args.save_interval, num_rollout_per_epoch, args.num_rollout) or (
                converged and args.save_interval is not None
            ):
                save(r_id, force_sync=converged)

            offload_train(actor_trains_this_step)

            # 4. Update weights (skip on final rollout to eliminate terminal stalls)
            if not converged and curr_train_id < args.num_rollout - 1:
                if sampler_client:
                    print(f"[TimeSlice] Acquiring Sampler GPU Grant for weight update after rollout {r_id} (job {job_id})...")
                    sampler_client.acquire()

                if args.offload_rollout:
                    ray.get(rollout_manager.onload_weights.remote())
                actor_model.update_weights()

                if args.offload_rollout:
                    ray.get(rollout_manager.onload_kv.remote())

                if sampler_client:
                    sampler_client.release()

            if trainer_client:
                print(f"[TimeSlice] Yielding Trainer GPU Grant after training rollout {r_id} (job {job_id})...")
                trainer_client.release()

            if not converged and should_run_periodic_action(r_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(r_id))

            curr_train_id += 1

            if converged:
                # Drain in-flight generations before shutdown. The trainer grant is
                # already released above; re-acquire only the sampler grant so the
                # paused engines can finish (never hold both grants here, keeping
                # the global trainer->sampler acquisition order).
                if pending_rollouts:
                    if sampler_client:
                        print(f"[TimeSlice] Acquiring Sampler GPU Grant to drain pending rollouts (job {job_id})...")
                        sampler_client.acquire()
                    for p_id, p_gen in pending_rollouts:
                        print(f"[Early Stop] Draining pending rollout {p_id} (result discarded)...")
                        ray.get(p_gen)
                    pending_rollouts.clear()
                break

        if sampler_client:
            print(f"[TimeSlice] Releasing final Sampler GPU Grant for job {job_id}...")
            sampler_client.release()

        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)
        return

    # train loop (synchronous fallback)
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

        converged = check_convergence(rollout_id)

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout) or (
            converged and args.save_interval is not None
        ):
            save(rollout_id, force_sync=converged)

        offload_train(actor_trains_this_step)

        # Skip weight broadcast on the final iteration (no more rollouts will be generated)
        if not converged and rollout_id < args.num_rollout - 1:
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
        else:
            if trainer_client:
                print(f"[TimeSlice] Yielding final Trainer GPU Grant after training completion for job {job_id}...")
                trainer_client.release()

        if not converged and should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        if converged:
            break

    if sampler_client:
        print(f"[TimeSlice] Releasing final Sampler GPU Grant for job {job_id}...")
        sampler_client.release()

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
