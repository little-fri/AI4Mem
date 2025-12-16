#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
import shlex
import glob

# ... (之前的配置保持不变) ...
ROOT = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_SO = os.path.join(ROOT, 'data_collector.so')
UVM_LOG = os.path.join(ROOT, 'uvm_monitor_result.csv')
UVM_LATEST = os.path.join(ROOT, 'uvm_latest.csv')
DATA_CSV = os.path.join(ROOT, 'data.csv')
TOTAL_CSV = os.path.join(ROOT, 'total_data.csv')

# === 全局变量：用于跟踪后台训练进程 ===
current_train_proc = None

def file_contains_uvm(path, max_read=8192):
    # ... (保持不变) ...
    try:
        with open(path, 'r', errors='ignore') as f:
            data = f.read(max_read)
            return ('UVM,' in data or 'GPU_PAGE_FAULT' in data or 'MIGRATE' in data)
    except Exception:
        return False

def find_latest_uvm_file():
    # ... (保持不变) ...
    candidates = []
    pattern = os.path.join(ROOT, 'uvm_monitor_result*.csv')
    candidates.extend(glob.glob(pattern))
    if os.path.exists(UVM_LATEST): candidates.append(UVM_LATEST)
    if os.path.exists(UVM_LOG): candidates.append(UVM_LOG)
    uniq = list(dict.fromkeys(candidates))
    if not uniq: return None
    uniq.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return uniq

def copy_latest_to_data():
    # ... (保持不变) ...
    ordered = find_latest_uvm_file()
    if not ordered: return False
    src = None
    for p in ordered:
        if file_contains_uvm(p):
            src = p
            break
    if src is None: src = ordered[0]
    try:
        shutil.copyfile(src, DATA_CSV + '.tmp')
        os.replace(DATA_CSV + '.tmp', DATA_CSV)
       # print(f"已拷贝 {os.path.basename(src)} -> data.csv")
        return True
    except Exception as e:
        print('拷贝 UVM 日志失败:', e)
        return False

def append_to_total(source_csv, total_csv, separator_ts=None):
    # ... (保持不变) ...
    try:
        with open(source_csv, 'r', errors='ignore') as f:
            content = f.read()
        if not content.strip(): return
        with open(total_csv, 'a') as out:
            if separator_ts: out.write(f"# ---- snapshot @ {separator_ts} ----\n")
            out.write(content)
            if not content.endswith('\n'): out.write('\n')
    except Exception as e:
        print('追加 total_data.csv 失败:', e)

def prune_rotated_files(keep_latest=True):
    # ... (保持不变) ...
    files = glob.glob(os.path.join(ROOT, 'uvm_monitor_result_*.csv'))
    if not files: return
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    survivors = files[:1] if keep_latest else []
    for f in files:
        if f in survivors: continue
        try: os.remove(f)
        except Exception: pass

def run_preprocess(csv_path, out_path, seq, vocab, horizon):
    # 强制增加 --fit 参数，确保适应新地址
    cmd = [sys.executable, os.path.join(ROOT, 'preprocess.py'), '--fit']
   # print('  [Sync] Running preprocess:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def run_infer(model_path, data_path, csv_path, out_path):
    cmd = [sys.executable, os.path.join(ROOT, 'infer.py')]
    #print('  [Sync] Running infer:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

# === 新增：后台异步训练函数 ===
def start_train_bg(epoch=30):
    global current_train_proc
    
    # 1. 检查之前的训练是否还在跑
    if current_train_proc is not None:
        ret_code = current_train_proc.poll()
        if ret_code is None:
            # 上一轮还没跑完，跳过本轮训练，防止进程堆积或文件冲突
            print("  [Async] 上一轮训练尚未结束，跳过本次触发。")
            return
        else:
            # 上一轮跑完了，清理掉
            current_train_proc = None

    # 2. 启动新的训练进程
    cmd = [sys.executable, os.path.join(ROOT, 'train_model.py'), '--epoch', str(epoch)]
    print(f"  [Async] 启动后台训练 (Epoch={epoch})...")
    
    # 使用 Popen 启动，不等待它结束
    # stdout/stderr 设为 None 表示直接输出到屏幕，如果想静默可以设为 subprocess.DEVNULL
    try:
        current_train_proc = subprocess.Popen(cmd)
    except Exception as e:
        print(f"  [Error] 启动训练失败: {e}")

def main():
    parser = argparse.ArgumentParser()
    # ... (保持原本的参数定义不变) ...
    parser.add_argument('--interval', type=int, default=5, help='循环间隔秒数')
    parser.add_argument('--test-cmd', default='./test', help='要运行的测试程序命令')
    parser.add_argument('--model', default=os.path.join(ROOT, 'model_big.pth'), help='模型路径')
    parser.add_argument('--data-out', default=os.path.join(ROOT, 'train_data_loop.pth'), help='预处理输出')
    parser.add_argument('--pred-out', default=os.path.join(ROOT, 'predictions.csv'), help='推理输出')
    parser.add_argument('--seq', type=int, default=10, help='序列长度')
    parser.add_argument('--vocab', type=int, default=4096, help='词表大小')
    parser.add_argument('--horizon', type=float, default=2.0, help='预测窗口')
    parser.add_argument('--prune-rotated', action='store_true', help='清理旧日志')
    parser.add_argument('--prune-keep-latest', action='store_true', help='保留最新')
    parser.add_argument('--total-log', action='store_true', help='启用 total_data.csv')
    parser.add_argument('--total-path', default=TOTAL_CSV, help='total_data.csv 路径')
    parser.add_argument('--total-sep', action='store_true', help='分隔符')
    args = parser.parse_args()

    if not os.path.exists(COLLECTOR_SO):
        print('找不到 data_collector.so 于', COLLECTOR_SO)
        sys.exit(1)

    env = os.environ.copy()
    env['LD_PRELOAD'] = COLLECTOR_SO
    env['UVM_LOG_FILE'] = UVM_LOG
    env['UVM_ROTATE_INTERVAL'] = str(args.interval)

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
                   # print('>>> 检测到新数据，开始 AI 流水线 <<<')
                    if not file_contains_uvm(DATA_CSV):
                        print('data.csv 暂无 UVM 事件，跳过本轮')
                        continue
                        
                    if args.total_log:
                        append_to_total(DATA_CSV, args.total_path, separator_ts=time.strftime('%Y-%m-%d %H:%M:%S') if args.total_sep else None)
                    
                    last_mtime = mtime
                    
                    try:
                        # 1. 预处理 (同步阻塞)：必须等数据准备好
                        run_preprocess(DATA_CSV, args.data_out, args.seq, args.vocab, args.horizon)
                        
                        # 2. 推理 (同步阻塞)：使用当前模型立即给出预测
                        run_infer(args.model, args.data_out, DATA_CSV, args.pred_out)
                        
                        # 3. 训练 (异步非阻塞)：后台悄悄更新模型，供下一轮使用
                        # 建议 Epoch 设小一点，比如 1 或 2，保证能快速跟上
                        start_train_bg(epoch=2)
                        
                    except subprocess.CalledProcessError as e:
                        print('流水线步骤失败，返回码', e.returncode)
                else:
                    print('data.csv 未更新，跳过本轮')

            # 检查测试程序是否退出
            if proc.poll() is not None:
                print('测试程序退出，做最后一次处理后结束')
                for attempt in range(3):
                    copy_latest_to_data()
                    if args.prune_rotated:
                        prune_rotated_files(keep_latest=args.prune_keep_latest)
                    if os.path.exists(DATA_CSV) and file_contains_uvm(DATA_CSV):
                        if args.total_log:
                            append_to_total(DATA_CSV, args.total_path, separator_ts=time.strftime('%Y-%m-%d %H:%M:%S') if args.total_sep else None)
                        try:
                            # 退出前，全部做成同步的，确保模型更新并输出最后结果
                            run_preprocess(DATA_CSV, args.data_out, args.seq, args.vocab, args.horizon)
                            
                            # 这里如果是退出前最后一次，建议同步训练一次
                            print("  [Final] 执行最后一次同步训练...")
                            subprocess.run([sys.executable, os.path.join(ROOT, 'train_model.py'), '--epoch', '1'], check=False)
                            
                            run_infer(args.model, args.data_out, DATA_CSV, args.pred_out)
                        except Exception as e:
                            print(f'最终处理异常: {e}')
                        break
                    time.sleep(1)
                break

    except KeyboardInterrupt:
        print('收到中断信号，停止测试程序...')
        proc.terminate()
    finally:
        # 清理后台训练进程
        if current_train_proc is not None and current_train_proc.poll() is None:
            print("正在终止后台训练进程...")
            current_train_proc.terminate()
            current_train_proc.wait()
            
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

if __name__ == '__main__':
    main()