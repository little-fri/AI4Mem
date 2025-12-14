#!/usr/bin/env python3
"""
循环流程：
1) 运行 auto_collect_loop.py 生成/刷新 total_data.csv（启用 --total-log）。
2) 使用 train_model.py 在当前 total_data.csv 上连续训练 N 次（默认 30），支持断点继续。
3) 重复上述流程 M 轮（默认 20）。

用法示例：
    python3 cycle_collect_train.py \
        --cycles 20 \
        --train-repeat 30 \
        --auto-args "--test-cmd ./test --interval 5 --total-sep" \
        --train-extra "" \
        --resume

注意：auto_collect_loop 在启用 --total-log 时会删除已有 total_data.csv，然后重新收集。
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_cmd(cmd, cwd=None):
    print("[cmd]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"命令失败，退出码 {result.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="循环执行：采集 -> 训练多次 -> 重复")
    parser.add_argument('--cycles', type=int, default=20, help='总循环次数')
    parser.add_argument('--train-repeat', type=int, default=30, help='每轮训练重复次数')
    parser.add_argument('--auto-script', default=str(ROOT / 'auto_collect_loop.py'), help='auto_collect_loop.py 路径')
    parser.add_argument('--train-script', default=str(ROOT / 'train_model.py'), help='train_model.py 路径')
    parser.add_argument('--total-path', default=str(ROOT / 'total_data.csv'), help='total_data.csv 路径')
    parser.add_argument('--model-path', default=str(ROOT / 'uvm_prefetch_model.txt'), help='模型保存路径')
    parser.add_argument('--feature-path', default=str(ROOT / 'feature_cols.pkl'), help='特征列保存路径')
    parser.add_argument('--auto-args', default='', help='传给 auto_collect_loop.py 的额外参数字符串，例如 "--test-cmd ./test --interval 5 --total-sep"')
    parser.add_argument('--train-extra', default='', help='传给 train_model.py 的额外参数字符串')
    parser.add_argument('--no-resume', action='store_true', help='禁用断点续训（默认启用续训）')
    args = parser.parse_args()

    auto_base = [sys.executable, args.auto_script, '--total-log', '--total-path', args.total_path]
    # 如果用户在 auto-args 里没带 --total-sep 也没关系，默认不加分隔行
    auto_cmd_extra = shlex.split(args.auto_args)

    train_base = [sys.executable, args.train_script, '--total-path', args.total_path,
                  '--model-path', args.model_path, '--feature-path', args.feature_path]
    if not args.no_resume:
        train_base.append('--resume')
    train_cmd_extra = shlex.split(args.train_extra)

    for cycle in range(1, args.cycles + 1):
        print(f"================ 开始第 {cycle}/{args.cycles} 轮采集 ================")
        run_cmd(auto_base + auto_cmd_extra, cwd=ROOT)

        print(f"================ 开始第 {cycle}/{args.cycles} 轮训练（共 {args.train_repeat} 次） ================")
        for i in range(1, args.train_repeat + 1):
            print(f"-- 训练 {i}/{args.train_repeat}")
            run_cmd(train_base + train_cmd_extra, cwd=ROOT)

    print("全部循环完成。")


if __name__ == '__main__':
    main()
