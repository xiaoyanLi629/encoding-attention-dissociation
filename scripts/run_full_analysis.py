"""
完整分析流程运行脚本
Run Full Analysis Pipeline for Cross-Modal Integration Study

每次运行会创建带时间戳的独立结果目录：
runs/run_YYYYMMDD_HHMMSS/
├── unimodal_models/
├── modality_contribution/
├── crossmodal_attention/
├── brain_networks/
└── figures/
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# Add src directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
sys.path.insert(0, SRC_DIR)


def create_run_directory(base_dir):
    """创建带时间戳的运行目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, 'runs', f'run_{timestamp}')
    
    subdirs = [
        'unimodal_models',
        'modality_contribution',
        'crossmodal_attention',
        'brain_networks',
        'figures'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir, timestamp


def run_script(script_path, args_list=None, description=""):
    """运行Python脚本"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"脚本: {os.path.basename(script_path)}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path]
    if args_list:
        cmd.extend(args_list)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✓ 完成! 耗时: {elapsed:.1f}秒")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='运行完整分析流程')
    parser.add_argument('--project_dir', default='/root/autodl-fs/CCN_Competition',
                        help='项目根目录')
    parser.add_argument('--subjects', default='1,2,3,5', help='被试列表')
    parser.add_argument('--skip_training', action='store_true', 
                        help='跳过模型训练步骤')
    parser.add_argument('--only', type=str, default=None,
                        choices=['train', 'contribution', 'attention', 'network', 'figures', 'dissociation_figure'],
                        help='只运行特定步骤')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='指定输出目录（默认创建带时间戳的新目录）')
    
    args = parser.parse_args()
    
    # Use the src directory for analysis scripts
    src_dir = SRC_DIR
    
    # 创建带时间戳的运行目录
    if args.output_dir:
        run_dir = args.output_dir
        os.makedirs(run_dir, exist_ok=True)
        for subdir in ['unimodal_models', 'modality_contribution', 'crossmodal_attention', 'brain_networks', 'figures']:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
        timestamp = "custom"
    else:
        run_dir, timestamp = create_run_directory(args.project_dir)
    
    print("=" * 70)
    print("跨模态整合研究 - 完整分析流程")
    print("Cross-Modal Integration Study - Full Analysis Pipeline")
    print("=" * 70)
    print(f"运行时间戳: {timestamp}")
    print(f"输出目录: {run_dir}")
    print(f"项目目录: {args.project_dir}")
    print(f"被试列表: {args.subjects}")
    print("=" * 70)
    
    # 保存运行配置
    config = {
        'timestamp': timestamp,
        'project_dir': args.project_dir,
        'subjects': args.subjects,
        'output_dir': run_dir,
        'start_time': datetime.now().isoformat()
    }
    
    import json
    with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    steps = [
        {
            'name': 'train',
            'script': os.path.join(src_dir, '01_train_unimodal_models.py'),
            'description': 'Step 1: 训练单模态编码模型',
            'skip': args.skip_training,
            'args': ['--project_dir', args.project_dir, '--subjects', args.subjects,
                    '--output_dir', os.path.join(run_dir, 'unimodal_models')]
        },
        {
            'name': 'contribution',
            'script': os.path.join(src_dir, '02_modality_contribution_analysis.py'),
            'description': 'Step 2: 模态贡献度分析',
            'args': ['--project_dir', args.project_dir, '--subjects', args.subjects,
                    '--input_dir', os.path.join(run_dir, 'unimodal_models'),
                    '--output_dir', os.path.join(run_dir, 'modality_contribution')]
        },
        {
            'name': 'attention',
            'script': os.path.join(src_dir, '03_crossmodal_attention_analysis.py'),
            'description': 'Step 3: 跨模态注意力分析',
            'args': ['--project_dir', args.project_dir, '--subjects', args.subjects,
                    '--output_dir', os.path.join(run_dir, 'crossmodal_attention')]
        },
        {
            'name': 'network',
            'script': os.path.join(src_dir, '04_brain_network_analysis.py'),
            'description': 'Step 4: 功能网络分析',
            'args': ['--project_dir', args.project_dir, '--subjects', args.subjects,
                    '--input_dir', os.path.join(run_dir, 'unimodal_models'),
                    '--output_dir', os.path.join(run_dir, 'brain_networks')]
        },
        {
            'name': 'figures',
            'script': os.path.join(src_dir, 'generate_all_figures.py'),
            'description': 'Step 5: 生成论文图表',
            'args': ['--project_dir', args.project_dir,
                    '--input_dir', run_dir,
                    '--output_dir', os.path.join(run_dir, 'figures')]
        },
        {
            'name': 'dissociation_figure',
            'script': os.path.join(src_dir, 'generate_encoding_attention_dissociation_figure.py'),
            'description': 'Step 6: 生成编码-注意力分离图',
            'args': []  # Script uses hardcoded paths
        }
    ]
    
    results = {}
    total_start = time.time()
    
    for step in steps:
        if args.only and step['name'] != args.only:
            continue
        
        if step.get('skip', False):
            print(f"\n跳过: {step['description']}")
            results[step['name']] = 'skipped'
            continue
        
        success = run_script(step['script'], step.get('args', []), step['description'])
        results[step['name']] = 'success' if success else 'failed'
        
        if not success and step['name'] in ['train']:
            print("\n警告: 关键步骤失败，后续分析可能受影响")
    
    total_elapsed = time.time() - total_start
    
    # 更新配置
    config['end_time'] = datetime.now().isoformat()
    config['total_elapsed_seconds'] = total_elapsed
    config['results'] = results
    
    with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("分析流程完成")
    print("=" * 70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_elapsed/60:.1f}分钟")
    print("\n步骤结果:")
    for step in steps:
        if step['name'] in results:
            status = results[step['name']]
            icon = '✓' if status == 'success' else ('○' if status == 'skipped' else '✗')
            print(f"  {icon} {step['description']}: {status}")
    
    print(f"\n输出目录: {run_dir}")
    print("=" * 70)
    
    return run_dir


if __name__ == "__main__":
    main()
