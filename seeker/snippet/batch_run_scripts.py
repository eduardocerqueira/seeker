#date: 2025-03-12T16:57:34Z
#url: https://api.github.com/gists/b91e481072763d295f7eae51b93b412e
#owner: https://api.github.com/users/zhuyk6

"""
这个脚本的出发点很简单，我有一个脚本 script ，它接受一个数据集 dataset ，然后执行计算。
这个计算可能非常耗时，但是并行性很差，对于大量的 dataset ，我们可以生成大量的 python 进程来执行这个脚本。

但是如果 dataset 太大，导致计算时间太长，我们希望超时就提前终止。
更进一步，如果 datasets 的大小是可以通过名字来判断的，比如 test_data1.txt, test_data2.txt, test_data3.txt，
那么我们可以在运行的时候，先检查一下这个 dataset 的大小，如果它比之前超时的 dataset 大，那就直接跳过这个 dataset。
我们可以通过一个临时文件 timeout.log 来记录这些超时的 dataset。

之所以这么做，是因为 进程池 和线程池 的执行顺序是不确定的，
如果动态的添加新的线程，终止进程、线程会过于复杂，起码 AI 给出的方案都是不能正确工作的。
"""

import argparse
from pathlib import Path
import pprint
import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import signal


def run_script_with_timeout(script_path: str, dataset_path: str, timeout: int, cmp_method):
    """
    运行单个脚本并对超时进行处理
    """
    try:
        path = Path(dataset_path)
        log_file = path.parent / "timeout.log"
        if log_file.exists():
            with log_file.open("r") as f:
                log_data = f.readlines()
            # each line of log_data is a timeout dataset_path
            for line in log_data:
                if cmp_method(line) < cmp_method(dataset_path):
                    # 如果当前数据集规模比日志中的数据集大，则跳过
                    return (dataset_path, -1, None, f"Skipped: {dataset_path}")

        # 使用 subprocess.Popen 启动脚本
        with subprocess.Popen(
            ["uv", "run", "python", script_path, dataset_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # 创建一个新的进程组
        ) as process:
            # 等待进程完成或超时
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return (dataset_path, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                # 如果超时，杀死整个进程组
                os.killpg(process.pid, signal.SIGTERM)

                # write the dataset_path to the log file
                with log_file.open("a") as f:
                    f.write(f"{dataset_path}\n")

                return (dataset_path, -1, None, f"Timeout expired after {timeout} seconds")
    except Exception as e:
        return (dataset_path, -1, None, f"Error: {str(e)}")


def batch_run_scripts(script_path: str, datasets: list[str], timeout: int, max_workers: int, cmp_method):
    """
    批量运行脚本
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个数据集提交任务
        futures = {
            executor.submit(run_script_with_timeout, script_path, dataset, timeout, cmp_method): dataset
            for dataset in datasets
        }
        # 收集结果
        for future in futures:
            results.append(future.result())
    return results


def cmp_test_datasets(s: str) -> int:
    name = Path(s).stem  # name format: test_data{size}.txt
    words = name.split("_")
    n = int(words[1][4:])
    return n


def load_datasets(datasets_dir_test: str, cmp_method) -> list[str]:
    """
    加载数据集目录下的所有数据集
    """
    datasets = []
    for root, dirs, files in os.walk(datasets_dir_test):
        for file in files:
            if file.endswith(".txt") or file.endswith(".qpy"):  # 假设数据集文件以 .txt 结尾
                datasets.append(os.path.join(root, file))

    datasets.sort(key=cmp_method)  # 按文件大小排序
    return datasets


def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Main Script for Benchmarking")

    # 添加位置参数
    parser.add_argument("script_path", type=str, help="Path to the script to benchmark")
    parser.add_argument("datasets_dir", type=str, help="Directory of datasets")

    # 添加可选参数
    parser.add_argument("-t", "--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600)")
    parser.add_argument("-w", "--max_workers", type=int, default=10, help="Maximum number of workers (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    # 解析命令行参数
    args = parser.parse_args()

    # 打印解析到的参数
    if args.verbose:
        print("Main Script for Benchmarking")
        print(f"Script Path: {args.script_path}")
        print(f"Datasets Dir: {args.datasets_dir}")
        print(f"Timeout: {args.timeout} seconds")
        print(f"Max Workers: {args.max_workers}")
        print(f"Verbose: {args.verbose}")
        print("=" * 50)

    datasets = load_datasets(args.datasets_dir, cmp_test_datasets)
    pprint.pprint(datasets)

    # 在这里实现你的 benchmark 逻辑
    results = batch_run_scripts(args.script_path, datasets, args.timeout, args.max_workers, cmp_test_datasets)

    # 打印结果
    for result in results:
        dataset_path, returncode, stdout, stderr = result
        print(f"Dataset: {dataset_path}")
        print(f"Return Code: {returncode}")
        if stdout:
            print("Output:")
            print(stdout)
        if stderr:
            print("Error:")
            print(stderr)
        print("=" * 50)


if __name__ == "__main__":
    main()
