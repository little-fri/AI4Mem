#!/usr/bin/env python3
"""
自动化循环：
- 以 LD_PRELOAD 注入 data_collector.so 运行测试程序（默认 ./test），采集 UVM 事件。
- 每隔 interval 秒（默认 5s）把最新采集日志拷贝到 data.csv（覆盖写，不生成新文件），
  随后调用 preprocess.py 预处理，再调用 infer.py 进行推理。
- 当测试程序退出时做最后一次处理后结束。

假设 data_collector.so 与本脚本在同一目录，模型默认使用 model_big.pth。
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import shlex
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_SO = os.path.join(ROOT, 'data_collector.so')
UVM_LOG = os.path.join(ROOT, 'uvm_monitor_result.csv')
UVM_LATEST = os.path.join(ROOT, 'uvm_latest.csv')
DATA_CSV = os.path.join(ROOT, 'data.csv')
TOTAL_CSV = os.path.join(ROOT, 'total_data.csv')


def file_contains_uvm(path, max_read=8192):
    try:
        with open(path, 'r', errors='ignore') as f:
            data = f.read(max_read)
            return ('UVM,' in data or 'GPU_PAGE_FAULT' in data or 'MIGRATE_HtoD' in data or 'MIGRATE_DtoH' in data)
    except Exception:
        return False


def find_latest_uvm_file():
    """在工作目录中寻找最新的 uvm_monitor_result*.csv 或 uvm_latest.csv"""
    candidates = []
    pattern = os.path.join(ROOT, 'uvm_monitor_result*.csv')
    candidates.extend(glob.glob(pattern))
    if os.path.exists(UVM_LATEST):
        candidates.append(UVM_LATEST)
    if os.path.exists(UVM_LOG):
        candidates.append(UVM_LOG)
    # 去重
    uniq = list(dict.fromkeys(candidates))
    if not uniq:
        return None
    # 按修改时间倒序
    uniq.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return uniq


def copy_latest_to_data():
    """选取最近的、尽量有 UVM 事件的文件拷贝到 data.csv（覆盖写）。"""
    ordered = find_latest_uvm_file()
    if not ordered:
        return False
    src = None
    # 先找包含 UVM 的最新文件
    for p in ordered:
        if file_contains_uvm(p):
            src = p
            break
    # 如果都没有，就用最新的一个
    if src is None:
        src = ordered[0]
    tmp = DATA_CSV + '.tmp'
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, DATA_CSV)
        print(f"已拷贝 {os.path.basename(src)} -> data.csv")
        return True
    except Exception as e:
        print('拷贝 UVM 日志失败:', e)
        return False


def append_to_total(source_csv, total_csv, separator_ts=None):
    """将当前 data.csv 追加到 total_data.csv，并用分隔行标记时间。"""
    try:
        with open(source_csv, 'r', errors='ignore') as f:
            content = f.read()
        if not content.strip():
            return
        with open(total_csv, 'a') as out:
            if separator_ts:
                out.write(f"# ---- snapshot @ {separator_ts} ----\n")
            out.write(content)
            if not content.endswith('\n'):
                out.write('\n')
    except Exception as e:
        print('追加 total_data.csv 失败:', e)


def prune_rotated_files(keep_latest=True):
    """删除 timestamp 轮转产生的 uvm_monitor_result_*.csv，保留最新（可选）。"""
    files = glob.glob(os.path.join(ROOT, 'uvm_monitor_result_*.csv'))
    if not files:
        return
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    survivors = files[:1] if keep_latest else []
    for f in files:
        if f in survivors:
            continue
        try:
            os.remove(f)
        except Exception:
            pass


def run_preprocess(csv_path, out_path, seq, vocab, horizon):
    cmd = [sys.executable, os.path.join(ROOT, 'preprocess.py'), '--input', csv_path, '--out', out_path,
           '--seq', str(seq), '--vocab', str(vocab), '--horizon', str(horizon)]
    print('  Running preprocess:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def run_infer(model_path, data_path, csv_path, out_path):
    cmd = [sys.executable, os.path.join(ROOT, 'infer.py'), '--model', model_path,
           '--data', data_path, '--csv', csv_path, '--out', out_path]
    print('  Running infer:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=5, help='循环间隔秒数')
    parser.add_argument('--test-cmd', default='./test', help='要运行的测试程序命令（可包含参数）')
    parser.add_argument('--model', default=os.path.join(ROOT, 'model_big.pth'), help='模型路径')
    parser.add_argument('--data-out', default=os.path.join(ROOT, 'train_data_loop.pth'), help='预处理输出路径（每轮覆盖）')
    parser.add_argument('--pred-out', default=os.path.join(ROOT, 'predictions.csv'), help='推理输出 CSV（每轮覆盖）')
    parser.add_argument('--seq', type=int, default=10, help='preprocess 序列长度')
    parser.add_argument('--vocab', type=int, default=4096, help='preprocess 词表大小')
    parser.add_argument('--horizon', type=float, default=2.0, help='preprocess 预测窗口（秒）')
    parser.add_argument('--prune-rotated', action='store_true', help='每轮清理 timestamp 轮转产生的 CSV，仅保留最新或全部删除')
    parser.add_argument('--prune-keep-latest', action='store_true', help='配合 --prune-rotated 使用，保留最新轮转文件，其余删除；默认全部删除')
    parser.add_argument('--total-log', action='store_true', help='启用 total_data.csv 聚合每轮 data.csv 内容')
    parser.add_argument('--total-path', default=TOTAL_CSV, help='total_data.csv 路径（仅在 --total-log 时使用）')
    parser.add_argument('--total-sep', action='store_true', help='在 total_data.csv 中添加时间戳分隔行')
    args = parser.parse_args()

    if not os.path.exists(COLLECTOR_SO):
        print('找不到 data_collector.so 于', COLLECTOR_SO)
        sys.exit(1)

    # 清理旧的 data.csv
    if os.path.exists(DATA_CSV):
        try:
            os.remove(DATA_CSV)
        except Exception:
            pass
    # 清理旧的 total_data.csv（按需）
    if args.total_log and os.path.exists(args.total_path):
        try:
            os.remove(args.total_path)
        except Exception:
            pass

    env = os.environ.copy()
    env['LD_PRELOAD'] = COLLECTOR_SO
    env['UVM_LOG_FILE'] = UVM_LOG
    env['UVM_ROTATE_INTERVAL'] = str(args.interval)  # 让 collector 每 interval 秒轮转

    test_cmd = shlex.split(args.test_cmd)
    print('启动测试程序:', ' '.join(test_cmd))
    proc = subprocess.Popen(test_cmd, env=env, cwd=ROOT)

    last_mtime = 0
    try:
        while True:
            time.sleep(args.interval)

            copied = copy_latest_to_data()
            if not copied:
                print('未找到采集文件，等待下一轮...')
            else:
                if args.prune_rotated:
                    prune_rotated_files(keep_latest=args.prune_keep_latest)
                try:
                    mtime = os.path.getmtime(DATA_CSV)
                except Exception:
                    mtime = 0
                if mtime > last_mtime:
                    print('检测到新的 data.csv，开始预处理和推理')
                    if not file_contains_uvm(DATA_CSV):
                        print('data.csv 暂无 UVM 事件，跳过本轮（mtime 未推进）')
                        # 不更新 last_mtime，等待下一轮可能出现的延迟 flush
                        continue
                    if args.total_log:
                        append_to_total(DATA_CSV, args.total_path, separator_ts=time.strftime('%Y-%m-%d %H:%M:%S') if args.total_sep else None)
                    last_mtime = mtime
                    try:
                        run_preprocess(DATA_CSV, args.data_out, args.seq, args.vocab, args.horizon)
                        run_infer(args.model, args.data_out, DATA_CSV, args.pred_out)
                    except subprocess.CalledProcessError as e:
                        print('子进程失败，返回码', e.returncode)
                else:
                    print('data.csv 未更新，跳过本轮')

            if proc.poll() is not None:
                print('测试程序退出，做最后一次处理后结束')
                # 为了等待 CUPTI flush，最多再尝试 3 次，每次间隔 1s
                for attempt in range(3):
                    copy_latest_to_data()
                    if args.prune_rotated:
                        prune_rotated_files(keep_latest=args.prune_keep_latest)
                    if os.path.exists(DATA_CSV) and file_contains_uvm(DATA_CSV):
                        if args.total_log:
                            append_to_total(DATA_CSV, args.total_path, separator_ts=time.strftime('%Y-%m-%d %H:%M:%S') if args.total_sep else None)
                        try:
                            run_preprocess(DATA_CSV, args.data_out, args.seq, args.vocab, args.horizon)
                            run_infer(args.model, args.data_out, DATA_CSV, args.pred_out)
                        except subprocess.CalledProcessError as e:
                            print('最终处理失败，返回码', e.returncode)
                        break
                    time.sleep(1)
                break
    except KeyboardInterrupt:
        print('收到中断信号，停止测试程序...')
        proc.terminate()
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == '__main__':
    main()
