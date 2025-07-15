import os
import sys
import time
import getpass
import pathlib
from typing import List, Optional, Dict
import importlib.util

import ray
import ray.exceptions
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup
from ray.job_submission import JobStatus as RayJobStatus

import realhf.base.logging as logging
from arealite.api.cli_args import (
    ClusterSpecConfig,
    LauncherConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
)
from arealite.api.io_struct import AllocationMode, AllocationType
from arealite.utils.launcher import (
    get_env_vars,
    wait_sglang_server_addrs
)
from realhf.base import logging, name_resolve
from realhf.scheduler.client import JobException

logger = logging.getLogger("RayLauncher")

RAY_WAIT_CHECK_TIME_INTERVAL = 5 # seconds


def run_func(file_path, function_name, *args, **kwargs):
    # Convert the file path to a module name
    module_name = file_path.replace('/', '_').replace('.', '_')
    
    # Load the module from file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Get the function and execute it
    function = getattr(module, function_name)
    return function(*args, **kwargs)


class RayLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        # job_name to ray future
        self.jobs = {}

    @property
    def run_name(self):
        return f"{self.experiment_name}_{self.trial_name}"

    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")
    
    def submit(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        gpus: int,
        cpus: int,
        mem: int,  # MB
        env_vars: Optional[Dict] = None,
        placement_group: Optional[PlacementGroup] = None,
        bundle_index: Optional[int] = None,
        *args, # arguments to pass to the function
        **kwargs, # keyword arguments to pass to the function
    ):
        runtime_env = RuntimeEnv(
            env_vars = env_vars or dict(),
        )
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement_group,
            bundle_index=bundle_index,
        ) if placement_group is not None else "DEFAULT"
        future = ray.remote(
            num_cpus=gpus,
            num_gpus=cpus,
            memory=mem*1024*1024,  # Convert MB to bytes
            runtime_env=runtime_env,
            scheduling_strategy=scheduling_strategy
        )(run_func).remote(
            file_path, func_name, *args, **kwargs
        )
        self.jobs[job_name] = future
        return future
    
    def submit_array(
        self,
        job_name: str,
        file_path: str,
        func_name: str,
        count: int,
        nodes: int,
        list_args: List[List],
        gpus_per_task: int,
        cpus_per_task: int,
        mem_per_task: int,  # MB
        list_kwargs: List[Dict] | None = None,
        env_vars: Optional[Dict] = None,
        amend_torch_dist_env: bool = False,
    ):
        """ Submit an array of jobs to Ray with ray placement groups.
        """

        if count % nodes != 0:
            raise ValueError(
                f"Count {count} is not divisible by nodes {nodes}. "
                "Please ensure that count is a multiple of nodes."
            )
        assert len(list_args) == count, (
            f"Length of list_args {len(list_args)} does not match count {count}."
        )
        if list_kwargs is not None:
            assert len(list_kwargs) == count, (
                f"Length of list_kwargs {len(list_kwargs)} does not match count {count}."
            )

        tasks_per_node = count // nodes
        gpus_per_node = gpus_per_task * tasks_per_node
        cpus_per_node = cpus_per_task * tasks_per_node
        mem_per_node = mem_per_task * tasks_per_node
        
        placement_group = ray.util.placement_group(
            bundles=[
                {
                    "CPU": cpus_per_node,
                    "GPU": gpus_per_node,
                    "memory": mem_per_node * 1024 * 1024,  # Convert MB to bytes
                }
            ] * nodes,
            strategy="STRICT_SPREAD",
        )
        ray.get(placement_group.ready())
        
        futures = []
        for i in enumerate(list_args):
            args = list_args[i]
            kwargs = list_kwargs[i] if list_kwargs is not None else {}

            # manage environment variables
            env_vars = env_vars or {}
            assert (
                "CUDA_VISIBLE_DEVICES" not in env_vars
            ), "CUDA_VISIBLE_DEVICES should be automatically resolved by Launcher instead of manually assigned."
            
            gpu_id_start = (i % tasks_per_node) * gpus_per_task
            gpu_id_end = ((i % tasks_per_node) + 1) * gpus_per_task
            node_id = i // tasks_per_node
            _env_vars = {
                **env_vars,
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(x) for x in range(gpu_id_start, gpu_id_end)
                ),
            }

            if amend_torch_dist_env:
                assert gpus_per_task == 1
                _env_vars.update({
                    "RANK": str(i),
                    "WORLD_SIZE": str(count),
                    "LOCAL_RANK": str(i % tasks_per_node),
                })
            
            future = self.submit(
                job_name=f"{job_name}:{i}",
                file_path=file_path,
                func_name=func_name,
                gpus=gpus_per_task,
                cpus=cpus_per_task,
                mem=mem_per_task,
                env_vars=_env_vars,
                placement_group=placement_group,
                bundle_index=node_id,
                *args,
                **kwargs,
            )
            futures.append(future)
    
        return futures

    def stop(self, job_name: str, force: bool = False):
        """ Stop a job by name. """
        if job_name in self.jobs:
            future = self.jobs[job_name]
            try:
                ray.cancel(future, force=force)
            except Exception as e:
                logger.error(f"Failed to cancel job {job_name}: {e}")
                return
            self.jobs.pop(job_name, None)
            logger.info(f"Job {job_name} stopped.")
        else:
            logger.warning(f"Job {job_name} not found in running jobs.")

    def stop_all(self, force: bool = False):
        """ Stop all jobs. """
        for job_name in list(self.jobs.keys()):
            self.stop(job_name, force=force)
        logger.info("All jobs stopped.")
        self.jobs.clear()

    def wait(self):
        """ Check every SCHEDULER_WAIT_CHECK_TIME_INTERVAL seconds for the status of all jobs.
        If any jobs failed terminate all jobs, and return.
        If any jobs completed, remove them from self.jobs.
        If all jobs are completed, return.
        """
        while self.jobs:
            completed_jobs = []
            for job_name, future in list(self.jobs.items()):
                try:
                    r = ray.get(future, timeout=0.1)
                    logger.info(f"Job {job_name} completed with result: {r}")
                    completed_jobs.append(job_name)
                except ray.exceptions.GetTimeoutError:
                    continue
                except ray.exceptions.RayTaskError as e:
                    logger.error(f"Job {job_name} failed with error: {e}, stopping all jobs.")
                    self.stop_all(force=True)
                    return
            for job_name in completed_jobs:
                self.jobs.pop(job_name, None)
                logger.info(f"Job {job_name} completed. Removed.")
            time.sleep(RAY_WAIT_CHECK_TIME_INTERVAL)


if __name__ == "__main__":
    # usage: python -m arealite.launcher.ray <entry_point> --config <config_path> [<additional_args>] launcher.ray.main_func_name=<main_func_name_in_entry_point>
    ray.init()
    config, config_file = parse_cli_args(sys.argv[2:])

    config.launcher = to_structured_cfg(config.launcher, LauncherConfig)
    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
    name_resolve.reconfigure(config.cluster.name_resolve)

    n_nodes = config.n_nodes
    n_gpus_per_node = config.n_gpus_per_node
    if n_gpus_per_node < config.cluster.n_gpus_per_node:
        raise ValueError(
            f"Slurm Launcher requires at least {config.cluster.n_gpus_per_node} (#GPUs per node) GPU. For usecases of less GPUs, use LocalLauncher instead."
        )
    elif n_gpus_per_node > config.cluster.n_gpus_per_node:
        raise ValueError(
            f"#GPU per node required by experiment ({n_gpus_per_node}) is larger than #GPU per node in the cluster  ({config.cluster.n_gpus_per_node})."
        )

    launcher = RayLauncher(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=config.cluster.fileroot,
    )
    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    sglang_cmds = []
    sglang_addrs = []
    n_sglang_nodes = 0
    if allocation_mode.type_ == AllocationType.DECOUPLED_SGLANG:
        # Launcher should launch SGLang servers according to allocation mode.
        config.sglang = to_structured_cfg(config.sglang, SGLangConfig)
        assert (
            allocation_mode.gen_pp_size == 1
        ), "Pipeline generation in SGLang is not supported for now."
        assert (
            allocation_mode.gen_tp_size <= config.cluster.n_gpus_per_node
        ), "Currently only support SGLang TP size less <= #GPUs per node."
        sglang_world_size = allocation_mode.gen_world_size
        sglang_tp_size = allocation_mode.gen_tp_size
        n_sglang_servers = allocation_mode.gen_dp_size
        n_sglang_nodes = allocation_mode.gen_world_size // n_gpus_per_node
        n_sglang_servers_per_node = config.cluster.n_gpus_per_node // sglang_tp_size

        base_seed = config.sglang.random_seed

        sglang_args_list = [
            sys.argv[2:] + [
                f"sglang.random_seed={base_seed + i}"
            ]  for i in range(n_sglang_servers)
        ]
        sglang_entry_point = pathlib.Path(__file__).resolve().joinpath("sglang_server.py")
        sglang_main_func_name = "main_sglang_server"
        launcher.submit_array(
            job_name="llm_server",
            file_path=sglang_entry_point,
            func_name=sglang_main_func_name,
            count=n_sglang_servers,
            nodes=n_sglang_nodes,
            list_args=sglang_args_list,
            gpus_per_task=sglang_tp_size,
            cpus_per_task=config.launcher.inference_server_cpus_per_gpu * sglang_tp_size,
            mem_per_task=config.launcher.inference_server_mem_per_gpu * sglang_tp_size,
            env_vars=get_env_vars(
                config.cluster.cluster_name,
                config.launcher.inference_server_env_vars,
            ),
        )

        # Get SGLang slurm nodes, find the hosts
        sglang_addrs = wait_sglang_server_addrs(
            config.experiment_name,
            config.trial_name,
            n_sglang_servers,
        )

    trainer_n_nodes = n_nodes - n_sglang_nodes
    trainer_entry_point = sys.argv[1]
    trainer_main_func_name = config.launcher.ray.main_func_name
    n_trainer_processes = trainer_n_nodes * config.cluster.n_gpus_per_node
    trainer_args_list = [
        sys.argv[2:] for _ in range(n_trainer_processes)
    ]
    if not config.server_only:
        # launch trainers
        launcher.submit_array(
            job_name="trainer",
            file_path=trainer_entry_point,
            func_name=trainer_main_func_name,
            count=trainer_n_nodes * config.cluster.n_gpus_per_node,
            nodes=trainer_n_nodes,
            list_args=trainer_args_list,
            gpus_per_task=1,
            cpus_per_task=config.launcher.trainer_cpus_per_gpu,
            mem_per_task=config.launcher.trainer_mem_per_gpu,
            env_vars=dict(
                **get_env_vars(
                    config.cluster.cluster_name,
                    config.launcher.trainer_env_vars,
                ),
                AREAL_LLM_SERVER_ADDRS=",".join(sglang_addrs),
            ),
            amend_torch_dist_env=True,
        )

    try:
        launcher.wait()
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        launcher.stop_all(force=True)
        raise e