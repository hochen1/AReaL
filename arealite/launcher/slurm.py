from typing import Tuple, List, Optional, Literal, Dict
import subprocess
import time
import re
import os
import getpass
import argparse
import sys

from omegaconf import OmegaConf

from arealite.api.cli_args import SGLangConfig, ClusterSpecConfig, parse_cli_args, to_structured_cfg
from arealite.api.io_struct import AllocationMode, AllocationType
from arealite.launcher.utils import find_config, find_and_amend_config
import realhf.base.logging as logging
from realhf.scheduler.client import (
    JobException,
    JobInfo,
    JobState
)

logger = logging.getLogger("SlurmLauncher")


SQUEUE_FIELDS = [
    "JobID",
    "State",
    "SubmitTime",
    "StartTime",
    "Name",
    "NodeList",
    "UserName",
    "MaxCPUs",
    "cpus-per-task",
    "NumTasks",
    "tres-alloc",
]
STATUS_MAPPING = {
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "PENDING": JobState.PENDING,
    "CANCELLED": JobState.CANCELLED,
    "FAILED": JobState.FAILED,
    "COMPLETED": JobState.COMPLETED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "DEADLINE": JobState.COMPLETED,
    "TIMEOUT": JobState.COMPLETED,
}


def cancel_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[str]] = None,
    signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL",
):
    assert (
        slurm_names is not None or slurm_ids is not None
    ), "Must specify slurm_names or slurm_ids."
    assert not (
        slurm_names and slurm_ids
    ), "Cannot specify both slurm_names and slurm_ids."
    cmd = ["scancel", "-s", signal]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    elif slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    subprocess.check_call(cmd)
    logger.info(
        f"Cancelled Slurm job with signal {signal}: "
        f"slurm identifiers {slurm_names if slurm_ids is None else slurm_ids}"
    )

def query_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[str]] = None,
    status: str = "all",
    delimiter: str = "__PSI__",
) -> List[JobInfo]:
    squeue_format = f":.{delimiter},".join(SQUEUE_FIELDS)
    cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    if slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    output = (
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    )
    rs = []
    for line in output.split("\n")[1:]:
        job_id, state, submit_time, start_time, slurm_name, nodelist, *_ = line.split(
            delimiter
        )
        rs.append(
            JobInfo(
                name=slurm_name,
                state=STATUS_MAPPING[state],
                host=nodelist,
                submit_time=submit_time,
                start_time=start_time,
                slurm_id=job_id.strip(),
            )
        )
    return rs


SCHEDULING_RETRY_INTERVAL_SECONDS = 30
SCHEDULING_TIMEOUT_MAX_SECONDS = 3600 * 24
SCHEDULER_WAIT_CHECK_TIME_INTERVAL = 5


SBATCH_SCRIPT_TEMPLATE = """
#!/bin/bash
{sbatch_options}

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo nodes=$nodes

nodes_array=($nodes)
echo node_array=$nodes_array

head_node=$\{nodes_array[0]\}
echo head_node=$head_node

# Getting the head node IP address
head_node_ip=$(srun --mpi=pmi2 --nodes=1 --ntasks=1 --nodelist="$head_node" hostname --ip-address)
echo head_node_ip=$head_node_ip

# srun commands
{srun_cmds}

wait
"""

SRUN_CMD_TEMPLATE = """
srun --mpi=pmi2 -K -l --chdir $PWD --nodes={nodes} --ntasks={ntasks} --gres:gpu:{n_gpus_per_node} --cpus-per-task={cpus_per_task} \\
    --mem-per-cpu={mem_per_cpu}M {apptainer_name} exec {apptainer_options} --bind {container_mounts} \\
    {container_env_strings} {container_image} {cmd} &

"""

LOCAL_CACHE_DIR = "/tmp/arealite"
PYTORCH_KERNEL_CACHE_PATH = (
    f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/torch/kernels"
)
TRITON_CACHE_PATH = f"{LOCAL_CACHE_DIR}/.cache/{getpass.getuser()}/triton"
os.makedirs(PYTORCH_KERNEL_CACHE_PATH, exist_ok=True)
os.makedirs(TRITON_CACHE_PATH, exist_ok=True)
BASE_ENVIRONS = {
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "OMP_NUM_THREADS": str(min(os.cpu_count(), 32)),
}
NA132_ENVIRONS = {
    "NCCL_SOCKET_IFNAME": "bond0",
    "NCCL_NET_PLUGIN": "",
    "NCCL_IB_GID_INDEX": "3",
    "NCCL_IB_TIMEOUT": "2",
    "NCCL_IB_RETRY_CNT": "7",
    "NCCL_IB_SL": "5",
    "NCCL_IB_TC": "136",
    "NCCL_IB_HCA": "mlx5_bond",
    "NCCL_IB_QPS_PER_CONNECTION": "8",
    "NCCL_SET_THREAD_NAME": "1",
    "NCCL_DEBUG_SUBSYS": "INIT,TUNING,GRAPH",
}

def get_env_vars(cluster_name: str):
    """Returns the environment variables for the cluster."""
    if cluster_name == "na132":
        return {**BASE_ENVIRONS, **NA132_ENVIRONS}
    else:
        return BASE_ENVIRONS


class SlurmLauncher:
    def __init__(self, experiment_name: str, trial_name: str, fileroot: str):
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.fileroot = fileroot

        # actual slurm job name -> JobInfo
        self.jobs: Dict[str, JobInfo] = {} 

    @property
    def run_name(self) -> str:
        """Returns the run name of this launcher."""
        return f"{self.experiment_name}_{self.trial_name}"
    
    def slurm_name(self, job_name: str) -> str:
        """Returns the slurm name of a job."""
        return f"{self.experiment_name}_{self.trial_name}:{job_name}"
    
    	
    def log_path_of(self, job_name: str) -> str:
        log_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(log_path, exist_ok=True)
        return os.path.join(log_path, f"{job_name}.log")
    
    def sbatch_path_of(self, job_name: str) -> str:
        sbatch_path = f"{self.fileroot}/sbatch/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(sbatch_path, exist_ok=True)
        return os.path.join(sbatch_path, f"{job_name}.sh")

    
    def submit(self, job_name, cmd, **kwargs):
        """Submits and launch a job with SBATCH.

        Args:
            cmd (str or List[str]): The core command to be executed.
        """
        return self.submit_array(job_name, cmd, count=1, **kwargs)

    def submit_array(
        self, 
        job_name: str, 
        cmd: List[str] | str,
        count: int,
        nodes: int,
        n_gpus_per_node: int,
        cpus_per_task: int,
        mem_per_task: int, # MB
        container_image: str,
        container_mounts: Optional[str] = None,
        env_vars: Optional[Dict] = None,
        nodelist: Optional[str] = None,
        exclude: Optional[str] = None,
        apptainer_name: Optional[str] = "singularity",
        apptainer_options: Optional[Tuple[str]] = (
            "--no-home", "--writable-tmpfs", "--nv", "--pid" 
        ),
    ):
        """Submits and launch a job array with SBATCH. 
        Note that a job array has one (unique) slurm name, and one (unique) slurm id.

        Args:
            job_name (str): The job name of the job array. The actual slurm name will be
                `<experiment_name>_<trial_name>:<job_name>`. 
            cmd (str or List[str]): The core command to be executed.
            count (int): The number of jobs in the array.
        """
        assert self.slurm_name(job_name) not in self.jobs, (
            f"Job {self.slurm_name(job_name)} is already submitted. "
            "Please use a different job name or stop the existing job."
        )
        if isinstance(cmd, str):
            cmd = [cmd]
        assert len(cmd) == count, (
            f"Command length {len(cmd)} does not match the job count {count}. "
            "Please provide a command for each job in the array."
        )
        assert count % nodes == 0, (
            f"Job count {count} must be divisible by the number of nodes {nodes}. "
            "Please adjust the job count or the number of nodes."
        )
        ntasks_per_node = count // nodes
        assert n_gpus_per_node % ntasks_per_node == 0, (
            "GPUs must be evenly distributed across tasks. "
            f"Current #GPUs per node {n_gpus_per_node}, #tasks per node {ntasks_per_node}."
        )

        mem_per_cpu = mem_per_task // cpus_per_task  # MB per CPU

        sbatch_options = [
            f"--job-name={self.slurm_name(job_name)}",
            f"--output={self.fileroot}/{self.run_name}/{job_name}.out",
            "--open-mode=append",
            "--no-requeue",
            f"--nodes={nodes}-{nodes}",
            f"--ntasks-per-node={ntasks_per_node}",
            f"--gres=gpu:{n_gpus_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem-per-cpu={mem_per_cpu}M",
        ]

        if nodelist:
            sbatch_options.append(f"--nodelist={nodelist}")
        if exclude:
            sbatch_options.append(f"--exclude={exclude}")

        sbatch_options_str = "\n".join([f"#SBATCH {opt}" for opt in sbatch_options])


        if env_vars is None:
            env_vars = dict()
        n_gpus_per_task = n_gpus_per_node // ntasks_per_node
        assert "CUDA_VISIBLE_DEVICES" not in env_vars, (
            "CUDA_VISIBLE_DEVICES should be automatically resolved by Launcher instead of manually assigned."
        )

        srun_cmds = []
        for i in range(count):
            # resolve CUDA_VISIBLE_DEVICES for each task
            gpu_id_start = (i % ntasks_per_node) * n_gpus_per_task
            gpu_id_end = ((i % ntasks_per_node) + 1) * n_gpus_per_task
            _env_vars = {
                **env_vars,
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(x) for x in range(gpu_id_start, gpu_id_end)
                ),
            }
            env_string = " ".join("--env {}={}".format(k, v) for k, v in (_env_vars or {}).items())
            # Prepare the command for each job in the array
            job_cmd = cmd[i]
            srun_cmd = SRUN_CMD_TEMPLATE.format(
                nodes=nodes,
                ntasks=ntasks_per_node,
                n_gpus_per_node=n_gpus_per_node,
                cpus_per_task=cpus_per_task,
                mem_per_cpu=mem_per_cpu,
                apptainer_name=apptainer_name,
                apptainer_options=" ".join(apptainer_options),
                container_mounts=container_mounts or "",
                container_env_strings=env_string,
                container_image=container_image,
                cmd=job_cmd,
            )
            srun_cmds.append(srun_cmd)

        srun_cmd = "\n".join(srun_cmds)
        sbatch_script = SBATCH_SCRIPT_TEMPLATE.format(
            sbatch_options=sbatch_options_str, srun_cmds=srun_cmd
        )
        sbatch_file_path = self.sbatch_path_of(f"{job_name}_{i}")
        with open(sbatch_file_path, "w") as f:
            f.write(sbatch_script)

        # Submit the job
        return_code = subprocess.check_call(["sbatch", sbatch_file_path])

        if return_code != 0:
            logger.info(
                f"Failed to submit job {self.slurm_name(job_name)}. "
                f"For debugging, please make sure your sbatch command works "
                f"and check generated sbatch file on {sbatch_file_path}."
            )
    
        self.jobs[self.slurm_name(job_name)] = JobInfo(
            name=self.slurm_name(job_name),
            state=JobState.PENDING,
        )
        self._update_all()

    def stop(self, job_name, signal=None):
        """Stops a running job.

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.
        
        Args:
            job_name: The job name of the job array to stop. 
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`. 
        """
        raise NotImplementedError()

    def stop_all(self, signal=None):
        """Stops all running jobs."""
        raise NotImplementedError()

    def find(self, job_name) -> JobInfo:
        """Gets the status of a job of this job.

        Args:
            job_name: The job name of the job array to find. 
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`. 

        Returns:
            A JobInfo if the job is found, or None otherwise.
        """
        self._update_all()
        return self.jobs.get(self.slurm_name(job_name), None)


    def find_all(self, job_name_regex=".*") -> List[JobInfo]:
        """Finds jobs.

        Args:
            job_name_regex: job name regex.

        Returns:
            A list of found JobInfo.
        """
        self._update_all()
        infos = []
        for r in self.jobs.values():
            job_name = r.name.split(":")[-1] # Extract the job name from slurm name
            if re.fullmatch(job_name_regex, job_name):
                infos.append(r)
        return infos
    
    def _find_job_with_status(
        self,
        status: List[JobState],
    ) -> List[JobInfo]:
        """Finds jobs with the given status.

        Args:
            status: A list of JobState to filter jobs.

        Returns:
            A list of JobInfo with the given status.
        """
        self._update_all()
        return [r for r in self.jobs.values() if r.state in status]


    def wait(
        self,
        timeout=None,
        check_status: Tuple[JobState, ...] = (
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ),
        remove_status: Tuple[JobState, ...] = (JobState.COMPLETED,),
        update=False,
    ):
        """Waits until all jobs submitted via this client instance finish."""
        # begin wait
        deadline = None if timeout is None else time.time() + timeout

        num_jobs_left = len(self.jobs)
        left = set(self.jobs.values())
        logger.info(
            f"Waiting for {num_jobs_left} jobs. Jobs IDs: "
            f"{','.join(sorted([x.slurm_id for x in self.jobs.values()]))}."
        )
        while len(left) > 0:
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {num_jobs_left} jobs. Job ID: "
                    f"{','.join(sorted([x.slurm_id for x in self.jobs.values()]))}."
                )
            self._update_all()
            left = list(self.jobs.values())
            for slurm_info in list(left):
                if slurm_info.slurm_id is None:
                    continue
                if slurm_info.state in check_status:
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=slurm_info.name,
                        host=slurm_info.host,
                        reason=slurm_info.state,
                    )
                if slurm_info.state in remove_status:
                    logger.info(
                        f"Job {slurm_info.name} is {slurm_info.state}. (Removed)"
                    )
                    left.remove(slurm_info)
                    if update:
                        self.jobs.pop(slurm_info.name)
            time.sleep(SCHEDULER_WAIT_CHECK_TIME_INTERVAL)

    def _update_all(self):
        """Updates the status of all jobs. """
        try:
            slurm_infos = query_jobs(list(self.jobs.keys()))
            for slurm_info in slurm_infos:
                if slurm_info.name in self.jobs:
                    self.jobs[slurm_info.name] = slurm_info
        except subprocess.CalledProcessError:
            logger.warning(
                "Calling squeue failed. Check slurm manually if you continue to see this warning."
            )


def slurm_args_parser():
    parser = argparse.ArgumentParser(description="Slurm Launcher for AReaL")
    parser.add_argument("entry_point", type=str, help="The entry point script to run.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "--sglang-server-base-port", type=int, required=False, default=27010, 
        help="Base port for SGLang servers. SGLang servers on the same node will ."
    )
    parser.add_argument("--trainer-port", type=int, required=False, default=27009, help="Pytorch distributed initialization port for trainer.")
    # parser.add_argument("remaining_args", nargs='*', help="Additional arguments to pass to the entry point script.")

    return parser

if __name__ == "__main__":
    # usage: python -m arealite.launcher.slurm <entry_point> --allocation_mode <allocation_mode> <config_path> [<args>] 
    r = parse_cli_args(sys.argv[2:], parser=slurm_args_parser())
    config = r.config
    args = r.additional_args

    cluster_config: ClusterSpecConfig = config.cluster
    n_nodes = config.n_nodes
    n_gpus_per_node = config.n_gpus_per_node
    if n_gpus_per_node < cluster_config.n_gpus_per_node:
        raise ValueError(
            f"Slurm Launcher requires at least {cluster_config.n_gpus_per_node} (#GPUs per node) GPU. For usecases of less GPUs, use LocalLauncher instead."
        )
    elif n_gpus_per_node > cluster_config.n_gpus_per_node:
        raise ValueError(
            f"#GPU per node required by experiment ({n_gpus_per_node}) is larger than #GPU per node in the cluster  ({cluster_config.n_gpus_per_node})."
        )
    
    launcher = SlurmLauncher(
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=cluster_config.fileroot
    )
    allocation_mode = config.allocation_mode
    allocation_mode = AllocationMode.from_str(allocation_mode)
    sglang_cmds = []
    n_sglang_nodes = 0
    if allocation_mode.type_ == AllocationType.DECOUPLED_SGLANG:
        # Launcher should launch SGLang servers according to allocation mode.
        assert isinstance(allocation_mode, str), "Allocation mode should be a string."
        sglang_config = find_and_amend_config(config, "sglang", SGLangConfig)
        assert "gen" in allocation_mode
        assert allocation_mode.gen_pp_size == 1, "Pipeline generation in SGLang is not supported for now."
        assert allocation_mode.gen_tp_size <= cluster_config.n_gpus_per_node, "Currently only support SGLang TP size less <= #GPUs per node."
        sglang_world_size = allocation_mode.gen_world_size 
        sglang_tp_size = allocation_mode.gen_tp_size
        n_sglang_server_instances = allocation_mode.gen_dp_size
        n_sglang_nodes = allocation_mode.gen_world_size // n_gpus_per_node

        model_path = find_config(config, "path")
        assert model_path is not None and isinstance(model_path, str)

        seed = find_config(config, "seed")
        base_port = args.sglang_server_base_port
        for i in range(n_sglang_server_instances):
            base_gpu_id = i * sglang_tp_size % n_gpus_per_node
            n_server_per_node = cluster_config.n_gpus_per_node // sglang_tp_size
            # Since we cannot get port information from slurm, we only ensure ports on the same .
            server_port = base_port + i % n_server_per_node * 2
            dist_port = base_port + i % n_server_per_node * 2 + 1
            dist_init_addr = f"tcp://localhost:{server_port}"
            sglang_cmds.append(
                SGLangConfig.build_cmd(
                    sglang_config,
                    model_path,
                    sglang_tp_size,
                    base_gpu_id=base_gpu_id,
                    seed=seed + i
                )
            )

        # launch sglang servers
        if sglang_cmds:
            launcher.submit_array(
                job_name="sglang-server",
                cmd=sglang_cmds,
                count=n_sglang_nodes,
                nodes=n_sglang_server_instances,
                n_gpus_per_node=cluster_config.n_gpus_per_node,
                cpus_per_task=15 * sglang_tp_size,  # one sglang task occupy the entire node
                mem_per_task=150 * 1024 * sglang_tp_size,  # 150GB per task
                container_image=cluster_config.gpu_infer_image,
                container_mounts=cluster_config.mount,
                env_vars=get_env_vars(cluster_config.cluster_name),
            )
    
    trainer_n_nodes = n_nodes - n_sglang_nodes
    trainer_cmd_template = (
        f"torchrun --nnodes={{nnodes}} --nproc-per-node={{nproc_per_node}} --node-rank {{node_rank}} "
        f"--master-addr $head_node_ip --master-port {args.trainer_port} {' '.join(sys.argv[1:])}"
    )
    
    trainer_cmds = []
    for i in range(trainer_n_nodes):
        # For each trainer node, we launch a trainer with the same command.
        # The node rank is the index of the node in the cluster.
        trainer_cmds.append(
            trainer_cmd_template.format(
                nnodes=trainer_n_nodes,
                nproc_per_node=cluster_config.n_gpus_per_node,
                node_rank=i
            )
        )
    
    # launch trainers 
    launcher.submit_array(
        job_name="trainer",
        cmd=trainer_cmds,
        count=trainer_n_nodes,
        nodes=trainer_n_nodes,
        n_gpus_per_node=cluster_config.n_gpus_per_node,
        cpus_per_task=120, # one trainer task occupy the entire node
        mem_per_task=1200 * 1024,  # 1.2T per task
        container_image=cluster_config.gpu_image,
        container_mounts=cluster_config.mount,
        env_vars=get_env_vars(cluster_config.cluster_name),
    )

    try:
        launcher.wait(
            check_status=(
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.NOT_FOUND,
                JobState.COMPLETED,
            ),
            remove_status=(),
        )
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        launcher.stop_all("SIGTERM")
        raise e

