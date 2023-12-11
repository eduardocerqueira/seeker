#date: 2023-12-11T16:41:14Z
#url: https://api.github.com/gists/ae4e7c6d14d0e5425b054ecc630debc0
#owner: https://api.github.com/users/matthieufond

#!/bin/python3

import os
import psutil
import sys
import yaml
from dataclasses import dataclass
from typing import Set, Optional, Dict, Any, List
from vfio_isolate import cpuset, nodeset

def is_cgroup_v2():
    return isinstance(cpuset.CPUSet.impl, cpuset.CGroupV2)

SYSTEM_CPUSET = "system.slice" if is_cgroup_v2 else "System"
USER_CPUSET = "user.slice" if is_cgroup_v2 else "User"

@dataclass
class Thread:
    name: str
    pid: int
    process: psutil.Process

@dataclass
class Proc:
    name: str
    pid: int
    process: psutil.Process
    threads: List[Thread]

@dataclass
class Procs:
    procs: List[Proc]

    def thread_pids(self, name: str, parent: str) -> List[int]:
        pids = []

        for proc in self.procs:
            if parent:
                if proc.name == parent:
                    for thread in proc.threads:
                        if name == thread.name:
                            pids.append(thread.pid)
            elif proc.name == name:
                pids.append(proc.pid)

        return pids

    def proc_pids(self, name: str) -> List[int]:
        return [proc.pid for proc in self.procs if proc.name == name]

class CPUSet:
    def __init__(self, name: str, config: Any, parent: Optional["CPUSet"] = None):
        self._name = name
        self._config = config
        self._cpus = CPUSet.parse_cpus(config["cpus"])
        self._tasks = config.get("tasks", [])

        self._path = name

        if parent is not None:
            self._path = "".join([parent._path, "/", self._path]) 

        self._cgroup = cpuset.CPUSet(self._path)
        self._subgroups: Dict[str, CPUSet] = {}

        for subname, subconf in self._config.get("subgroups", {}).items():
            self._subgroups[subname] = CPUSet(subname, subconf, self)

        @staticmethod
        def is_con_enabled(cpuset, controller):
            with cpuset.open("cgroup.controllers", "r") as f:
                for c in f.readline().split(" "):
                    if controller == c.replace("\n", ""):
                        return True
            return False
        
        cpuset.CGroupV2.is_controller_enabled = is_con_enabled

    @staticmethod
    def parse_cpus(yaml_cpu_conf: Any) -> Set[int]:
        cpus = set()
        for elem in yaml_cpu_conf:
            if isinstance(elem, int):
                cpus.add(elem)
            elif isinstance(elem, str):
                rnge = elem.split(",")
                if len(rnge) == 1:
                    cpus.add(int(rnge[0]))
                elif len(rnge) == 2:
                    cpus.update(range(int(rnge[0]), int(rnge[1]) + 1))
                else:
                    raise TypeError("CPUset elements should be either a single number or a range in the form of a start,end string (inclusive)")
            else:
                raise(TypeError("CPUset elements should be either an int or a string describing a range"))
        return cpus

    def ensure_cpuset_controller(self) -> None:
        assert self.exists(), "CPUset must exist to ensure controller"

        cpuset.CGroupV2.ensure_cpuset_controller_enabled(self._cgroup)
        with self._cgroup.open("cgroup.subtree_control", "w") as f:
            f.write("+cpuset +cpu")

    def exists(self):
        try:
            self._cgroup.get_cpus()
        except FileNotFoundError:
            return False
        return True
    
    def apply(self):
        print(f"Applying {self._name}")

        cpu_node_set =  nodeset.CPUNodeSet(self._cpus)

        try:
            self._cgroup.create(cpu_node_set)
        except FileExistsError:
            pass

        if self._name not in [SYSTEM_CPUSET, USER_CPUSET]:
            self.set_threaded()
            self.ensure_cpuset_controller()

        self._cgroup.set_cpus(cpu_node_set)


    def apply_procs(self, procs_db: Procs):
        for task in self._tasks:
            if ":" not in task:
                for pid in procs_db.proc_pids(task):
                    self._cgroup.add_pid(str(pid))

    def apply_threads(self, procs_db: Procs):
        for task in self._tasks:
            if ":" in task:
                thread = task.split(":")
                for tid in procs_db.thread_pids(thread[1], thread[0]):
                    self.add_thread(tid)

    def add_thread(self, tid: int):
        with self._cgroup.open("cgroup.threads", "w") as f:
            f.write(str(tid))

    def set_threaded(self):
        with self._cgroup.open("cgroup.type", "w") as f:
            f.write("threaded")

    def __repr__(self) -> str:
        ret = f"{self._name}, {self._cpus}, {self._tasks}, {[f'{group}' for group in self._subgroups.values()]}"

        return ret

class CPUset_Config:
    def __init__(self, cpusets):
        self._cpusets: Dict[str, CPUSet] = cpusets

    def apply_config(self, procs_db: Procs):
        self._apply(self._cpusets.values())
        self._apply_procs(self._cpusets.values(), procs_db)
        self._apply_threads(self._cpusets.values(), procs_db)

    def _apply(self, cpusets: List[CPUSet]):
        for cpuset in cpusets:
            cpuset.apply()
            self._apply(cpuset._subgroups.values())

    def _apply_procs(self, cpusets: List[CPUSet], procs_db: Procs):
        for cpuset in cpusets:
            cpuset.apply_procs(procs_db)
            self._apply_procs(cpuset._subgroups.values(), procs_db)

    def _apply_threads(self, cpusets: List[CPUSet], procs_db: Procs):
        for cpuset in cpusets:
            cpuset.apply_threads(procs_db)
            self._apply_threads(cpuset._subgroups.values(), procs_db)


def fill_procs() -> Procs:
    processes = psutil.process_iter()
    procs = []

    for proc in processes:
        threads = []
        for th in proc.threads():
            thread_proc = psutil.Process(th.id)
            threads.append(Thread(thread_proc.name(), thread_proc.pid, thread_proc))

        procs.append(Proc(proc.name(), proc.pid, proc, threads))

    return Procs(procs)

def main(args):

    csets = {}

    assert len(args) == 1

    with open(args[0], "r") as file:
        yaml_conf = yaml.load(file)

    for name, config in yaml_conf.items():
        if name == "system":
            for name in [SYSTEM_CPUSET, USER_CPUSET]:
                csets[name] = CPUSet(name, config)
        else:
            csets[name] = CPUSet(name, config)

    cpuset_config = CPUset_Config(csets)

    print(cpuset_config._cpusets)

    procs_db = fill_procs()

    cpuset_config.apply_config(procs_db)

if __name__ == "__main__":
    main(sys.argv[1:])