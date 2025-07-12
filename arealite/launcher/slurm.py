import argparse
import getpass
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, List, Literal, Optional, Tuple

from omegaconf import OmegaConf

import realhf.base.logging as logging
from arealite.api.cli_args import (
    ClusterSpecConfig,
    SGLangConfig,
    parse_cli_args,
    to_structured_cfg,
)
from arealite.api.io_struct import AllocationMode, AllocationType
from arealite.launcher.utils import find_and_amend_config, find_config
from realhf.scheduler.client import JobException, JobInfo, JobState

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
    slurm_ids: Optional[List[int]] = None,
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
        cmd += [",".join(str(s) for s in slurm_ids)]
    subprocess.check_call(cmd)
    logger.info(
        f"Cancelled Slurm job with signal {signal}: "
        f"slurm identifiers {slurm_names if slurm_ids is None else slurm_ids}. CMD: {cmd}"
    )


def query_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[int]] = None,
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
                slurm_id=int(job_id.strip()),
            )
        )
    print(rs)
    return rs


def parse_slurm_nodelist(nodelist: str) -> List[str]:
    return (
        subprocess.check_output(
            [
                "scontrol",
                "show",
                "hostnames",
                nodelist,
            ]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )


def get_slurm_host_ip(node: str):
    try:
        cmd = f"srun --overlap --mpi=pmi2 --immediate=1 --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist={node} hostname --ip-address"
        return subprocess.check_output(cmd.split(" ")).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        logger.warning(f"Get slurm host ip for node {node} failed.")


SCHEDULING_RETRY_INTERVAL_SECONDS = 30
SCHEDULING_TIMEOUT_MAX_SECONDS = 3600 * 24
SCHEDULER_WAIT_CHECK_TIME_INTERVAL = 5


SBATCH_SCRIPT_TEMPLATE = """#!/bin/bash
{sbatch_options}

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo nodes=$nodes

nodes_array=($nodes)
echo node_array=$nodes_array

head_node=${{nodes_array[0]}}
echo head_node=$head_node

# Getting the head node IP address
head_node_ip=$(srun --overlap --mpi=pmi2 --nodes=1 --ntasks=1 -n1 -c1 --mem=10M --nodelist="$head_node" hostname --ip-address)
echo head_node_ip=$head_node_ip

# srun commands
{srun_cmds}

wait
"""

SRUN_CMD_TEMPLATE = """srun --overlap --mpi=pmi2 -K -l --chdir $PWD --nodelist=${{nodes_array[{node_id}]}} \\
    --nodes={nodes} --ntasks={ntasks} --gres=gpu:{n_gpus_per_node} --cpus-per-task={cpus_per_task} \\
    --mem-per-cpu={mem_per_cpu}M {apptainer_name} exec {apptainer_options} --bind {container_mounts} \\
    {container_env_strings} \\
    {container_image} \\
    {cmd} &
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
    "HF_ENDPOINT": "https://hf-mirror.com",
    "PYTHONPATH": pathlib.Path(__file__).resolve().parent.parent.parent,
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
    "NCCL_DEBUG": "WARN",
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

        # slurm_job_id -> JobInfo
        self.jobs: Dict[int, JobInfo] = {}
        self.job_names = []

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
        sbatch_path = f"{self.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}/{self.trial_name}"
        os.makedirs(sbatch_path, exist_ok=True)
        return os.path.join(sbatch_path, f"{job_name}.sh")

    def submit(self, job_name, cmd, **kwargs):
        """Submits and launch a job with SBATCH.

        Args:
            cmd (str or List[str]): The core command to be executed.
        """
        return self.submit_array(job_name, cmd, count=1, **kwargs)

    def find_job_id(self, job_name: str):
        job_name = self.slurm_name(job_name)
        for job_id, job_info in self.jobs.items():
            if job_info.name == job_name:
                return job_id
        return None

    def submit_array(
        self,
        job_name: str,
        cmd: List[str] | str,
        count: int,
        nodes: int,
        n_gpus_per_node: int,
        cpus_per_task: int,
        mem_per_task: int,  # MB
        container_image: str,
        container_mounts: Optional[str] = None,
        env_vars: Optional[Dict] = None,
        nodelist: Optional[str] = None,
        exclude: Optional[str] = None,
        apptainer_name: Optional[str] = "singularity",
        apptainer_options: Optional[Tuple[str, ...]] = (
            "--no-home",
            "--writable-tmpfs",
            "--nv",
            "--pid",
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
        assert job_name not in self.job_names, (
            f"Job {job_name} is already submitted. "
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
        mem_per_node = mem_per_task * count // nodes + 1024 * 100  # FIXME

        sbatch_options = [
            f"--job-name={self.slurm_name(job_name)}",
            f"--output={self.log_path_of(job_name)}",
            # "--open-mode=append", # FIXME
            "--no-requeue",
            f"--nodes={nodes}-{nodes}",
            f"--ntasks-per-node={ntasks_per_node}",
            f"--gres=gpu:{n_gpus_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem={mem_per_node}M",
        ]

        if nodelist:
            sbatch_options.append(f"--nodelist={nodelist}")
        if exclude:
            sbatch_options.append(f"--exclude={exclude}")

        sbatch_options_str = "\n".join([f"#SBATCH {opt}" for opt in sbatch_options])

        if env_vars is None:
            env_vars = dict()
        n_gpus_per_task = n_gpus_per_node // ntasks_per_node
        assert (
            "CUDA_VISIBLE_DEVICES" not in env_vars
        ), "CUDA_VISIBLE_DEVICES should be automatically resolved by Launcher instead of manually assigned."

        srun_cmds = []
        for i in range(count):
            # resolve CUDA_VISIBLE_DEVICES for each task
            gpu_id_start = (i % ntasks_per_node) * n_gpus_per_task
            gpu_id_end = ((i % ntasks_per_node) + 1) * n_gpus_per_task
            node_id = i // ntasks_per_node
            _env_vars = {
                **env_vars,
                "CUDA_VISIBLE_DEVICES": ",".join(
                    str(x) for x in range(gpu_id_start, gpu_id_end)
                ),
            }
            env_string = " ".join(
                "--env {}={}".format(k, v) for k, v in (_env_vars or {}).items()
            )
            # Prepare the command for each job in the array
            job_cmd = cmd[i]
            # FIXME: only for debugging, remove and replace new image
            job_cmd = f'bash -c "pip3 install -U gymnasium torchdata tensordict hf-xet; {job_cmd}"'

            srun_cmd = SRUN_CMD_TEMPLATE.format(
                nodes=1,
                ntasks=1,
                node_id=node_id,
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

        srun_cmds = "\n".join(srun_cmds)
        sbatch_script = SBATCH_SCRIPT_TEMPLATE.format(
            sbatch_options=sbatch_options_str, srun_cmds=srun_cmds
        )
        sbatch_file_path = self.sbatch_path_of(f"{job_name}")
        with open(sbatch_file_path, "w") as f:
            f.write(sbatch_script)

        # Submit the job
        try:
            output = (
                subprocess.check_output(["sbatch", sbatch_file_path])
                .decode("utf-8")
                .strip()
            )
            logger.info(
                f"Submitted Slurm job {self.slurm_name(job_name)} to scheduler. To check the output, run \n\t`tail -f {self.log_path_of(job_name)}`."
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Failed to submit job {self.slurm_name(job_name)}. "
                f"For debugging, please make sure your sbatch command works "
                f"and check generated sbatch file on {sbatch_file_path}."
            )
            logger.error(f"Error message: {e}")
            return

        match = re.search(r"Submitted batch job (\d+)", output)
        slurm_job_id = int(match.group(1)) if match else None
        if slurm_job_id is None:
            logger.warning(
                f"Failed to obtain job id for job {self.slurm_name(job_name)}. "
                f"sbatch output: {output}"
            )
            return

        assert isinstance(slurm_job_id, int)
        self.jobs[slurm_job_id] = JobInfo(
            name=self.slurm_name(job_name),
            state=JobState.PENDING,
            slurm_id=slurm_job_id,
        )
        self._update_all()

    def stop(self, job_name, signal="SIGKILL"):
        """Stops a running job.

        Raises exception if there is no such job, but passes if the job
        has stopped either successfully or not.

        Args:
            job_name: The job name of the job array to stop.
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`.
        """
        job_id = self.find_job_id(job_name)
        if not job_id:
            return
        return cancel_jobs(slurm_ids=[job_id], signal=signal)

    def stop_all(self, signal="SIGKILL"):
        """Stops all running jobs."""
        return cancel_jobs(slurm_ids=list(self.jobs.keys()), signal=signal)

    def find(self, job_name) -> JobInfo | None:
        """Gets the status of a job of this job.

        Args:
            job_name: The job name of the job array to find.
                The actual slurm job name will be `<experiment_name>_<trial_name>:<job_name>`.

        Returns:
            A JobInfo if the job is found, or None otherwise.
        """
        self._update_all()
        job_id = self.find_job_id(job_name)
        return self.jobs[job_id] if job_id else None

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
            job_name = r.name.split(":")[-1]  # Extract the job name from slurm name
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
        left = list(self.jobs.keys())
        logger.info(
            f"Waiting for {num_jobs_left} jobs. Jobs IDs: "
            f"{','.join(sorted([str(x.slurm_id) for x in self.jobs.values()]))}."
        )
        while len(left) > 0:
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {num_jobs_left} jobs. Job ID: "
                    f"{','.join(sorted([str(x.slurm_id) for x in self.jobs.values()]))}."
                )
            self._update_all()
            left = list(self.jobs.keys())
            for slurm_id in list(left):
                slurm_info = self.jobs[slurm_id]
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
                    left.remove(slurm_id)
                    if update:
                        self.jobs.pop(slurm_info.slurm_id)
            time.sleep(SCHEDULER_WAIT_CHECK_TIME_INTERVAL)

    def _update_all(self):
        """Updates the status of all jobs."""
        try:
            slurm_infos = query_jobs(slurm_ids=list(self.jobs.keys()))
            for slurm_info in slurm_infos:
                assert slurm_info.slurm_id is not None
                self.jobs[slurm_info.slurm_id] = slurm_info
        except subprocess.CalledProcessError:
            logger.warning(
                "Calling squeue failed. Check slurm manually if you continue to see this warning."
            )


if __name__ == "__main__":
    # usage: python -m arealite.launcher.slurm <entry_point> <config_path> [<args>]
    config, config_file = parse_cli_args(sys.argv[2:])
    entry_point = sys.argv[1]

    config.cluster = to_structured_cfg(config.cluster, ClusterSpecConfig)
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

    launcher = SlurmLauncher(
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
        base_port = args.sglang_server_base_port
        sglang_ports_on_node = set()
        for i in range(n_sglang_servers):
            base_gpu_id = i * sglang_tp_size % n_gpus_per_node
            # Since we cannot get port information from slurm, we only ensure ports on the same .
            server_port = base_port + i % n_sglang_servers_per_node * 2
            # print(server_port, dist_init_addr)
            config.sglang.random_seed = base_seed + i
            sglang_cmds.append(
                SGLangConfig.build_cmd(
                    config.sglang,
                    sglang_tp_size,
                    base_gpu_id=0,
                    host="localhost",
                    port=server_port,
                    dist_init_addr=None,
                    sglang_version=args.sglang_version,
                )
            )
            sglang_ports_on_node.add(server_port)
        assert len(sglang_ports_on_node) == n_sglang_servers_per_node

        # launch SGLang servers, note that we need to leave some resources on each node
        # to schedule jobs that retrieve node IP. (1 CPU core & 10 MB memory)
        if sglang_cmds:
            launcher.submit_array(
                job_name="sglang-server",
                cmd=sglang_cmds,
                count=n_sglang_servers,
                nodes=n_sglang_nodes,
                n_gpus_per_node=config.cluster.n_gpus_per_node,
                cpus_per_task=15 * sglang_tp_size,
                mem_per_task=150 * 1024 * sglang_tp_size,  # 20GB per task
                container_image=config.cluster.gpu_infer_image,
                container_mounts=config.cluster.mount,
                env_vars=get_env_vars(config.cluster.cluster_name),
            )

        # Get SGLang slurm nodes, find the hosts
        start_time = time.perf_counter()
        while True:
            job_info = launcher.find("sglang-server")
            assert job_info is not None
            print(job_info)
            sglang_hosts = job_info.host
            logger.info(
                f"Waiting for SGLang servers to be scheduled by slurm, time since started = {time.perf_counter() - start_time:.2f}"
            )
            if sglang_hosts:
                print(sglang_hosts)
                sglang_hosts = [
                    get_slurm_host_ip(node)
                    for node in parse_slurm_nodelist(sglang_hosts)
                ]
                print(sglang_hosts)
                for host in sglang_hosts:
                    sglang_addrs.extend(
                        [f"{host}:{port}" for port in sglang_ports_on_node]
                    )
                logger.info(f"Get SGLang addresses: {' '.join(sglang_addrs)}")
                assert len(sglang_addrs) == n_sglang_servers
                break
            time.sleep(10)

    trainer_n_nodes = n_nodes - n_sglang_nodes
    assert r.overrides is not None
    trainer_cmd_template = (
        f"torchrun --nnodes={{nnodes}} --nproc-per-node={{nproc_per_node}} --node-rank {{node_rank}} "
        f"--master-addr $head_node_ip --master-port {args.trainer_port} {entry_point} --config {config_file} {' '.join(r.overrides)}"
    )

    trainer_cmds = []
    for i in range(trainer_n_nodes):
        # For each trainer node, we launch a trainer with the same command.
        # The node rank is the index of the node in the cluster.
        trainer_cmds.append(
            trainer_cmd_template.format(
                nnodes=trainer_n_nodes,
                nproc_per_node=config.cluster.n_gpus_per_node,
                node_rank=i,
            )
        )

    if not config.server_only:
        # launch trainers
        launcher.submit_array(
            job_name="trainer",
            cmd=trainer_cmds,
            count=trainer_n_nodes,
            nodes=trainer_n_nodes,
            n_gpus_per_node=config.cluster.n_gpus_per_node,
            cpus_per_task=120,  # one trainer task occupy the entire node
            mem_per_task=1024 * 1024,  # 1024GB per task
            container_image=config.cluster.gpu_image,
            container_mounts=config.cluster.mount,
            env_vars=dict(
                **get_env_vars(config.cluster.cluster_name),
                AREAL_LLM_SERVER_ADDRS=",".join(sglang_addrs),
            ),
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
        launcher.stop_all("SIGKILL")
        raise e
