"""
UniRig 一键推理脚本：输入模型 → 输出带骨骼蒙皮的模型

使用方法:
    python run_one.py --input model.obj --output result.glb
    python run_one.py --input model.glb --output result.fbx
    python run_one.py --input model.fbx --output output/model.glb
    
    # 高质量模式（更慢但质量更好）
    python run_one.py --input model.obj --output result.glb --num_beams 20
    
    # 快速模式（更快但可能质量下降）
    python run_one.py --input model.obj --output result.glb --num_beams 1

特性:
    - 默认保留原始纹理材质
    - 所有中间文件存放在 tmp/ 目录，自动清理
    - 支持格式: obj, fbx, glb, gltf, dae, vrm

关键质量参数:
    --num_beams       Beam search 数量，越大质量越高但越慢 (默认15，推荐1-20)
    --faces_target_count  目标面数，越大细节越多 (默认50000)
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, desc: str = ""):
    """运行命令并实时输出"""
    print(f"\n{'='*60}")
    print(f"[步骤] {desc}")
    print(f"[命令] {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[错误] {desc} 失败!")
        sys.exit(1)
    print(f"[完成] {desc}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="UniRig 一键推理：输入模型，输出带骨骼蒙皮的模型"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径 (支持 obj, fbx, glb, gltf, dae, vrm)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径 (支持 fbx, glb, gltf)")
    parser.add_argument("--tmp_dir", type=str, default="tmp",
                        help="临时文件目录，默认 tmp")
    parser.add_argument("--seed", type=int, default=12345,
                        help="随机种子")
    parser.add_argument("--keep_temp", action="store_true",
                        help="保留中间文件（用于调试）")
    
    # ===== 网格预处理参数 =====
    parser.add_argument("--faces_target_count", type=int, default=50000,
                        help="目标面数（用于网格简化，越大细节越多但越慢）")
    
    # ===== 骨骼生成质量参数（核心） =====
    parser.add_argument("--num_beams", type=int, default=15,
                        help="Beam search 数量，越大质量越高但越慢 (默认15，推荐1-20)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="最大生成 token 数，控制骨骼复杂度 (默认2048)")
    parser.add_argument("--repetition_penalty", type=float, default=3.0,
                        help="重复惩罚系数 (默认3.0)")
    
    # ===== 高级参数 =====
    parser.add_argument("--skeleton_task", type=str, 
                        default="configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml",
                        help="骨骼预测任务配置")
    parser.add_argument("--skin_task", type=str,
                        default="configs/task/quick_inference_unirig_skin.yaml",
                        help="蒙皮预测任务配置")
    
    args = parser.parse_args()
    
    # 确定工作目录
    workspace = Path(__file__).parent.absolute()
    os.chdir(workspace)
    
    # 解析输入输出路径
    input_path = Path(args.input).absolute()
    output_path = Path(args.output).absolute()
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"[错误] 输入文件不存在: {input_path}")
        sys.exit(1)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建临时目录（绝对路径，确保不污染输入输出目录）
    tmp_dir = Path(args.tmp_dir).absolute()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件信息
    input_name = input_path.stem
    input_suffix = input_path.suffix.lower()
    output_suffix = output_path.suffix.lower().lstrip('.')
    
    # 验证格式
    supported_input = ['.obj', '.fbx', '.glb', '.gltf', '.dae', '.vrm']
    supported_output = ['fbx', 'glb', 'gltf']
    
    if input_suffix not in supported_input:
        print(f"[错误] 不支持的输入格式: {input_suffix}")
        print(f"支持的格式: {', '.join(supported_input)}")
        sys.exit(1)
    
    if output_suffix not in supported_output:
        print(f"[错误] 不支持的输出格式: {output_suffix}")
        print(f"支持的格式: {', '.join(supported_output)}")
        sys.exit(1)
    
    # 创建本次任务的临时子目录
    task_tmp = tmp_dir / f"run_one_{input_name}"
    task_tmp.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("UniRig 一键推理")
    print("="*60)
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"格式: {output_suffix.upper()}")
    print(f"临时目录: {task_tmp}")
    print("="*60)
    
    # ========== 步骤1: 提取网格 ==========
    # 生成时间戳用于日志
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    extract_cmd = [
        sys.executable, "bpy_wrapper.py",
        "--module", "src.data.extract",
        "--",
        "--config=configs/data/quick_inference.yaml",
        f"--require_suffix=obj,fbx,FBX,dae,glb,gltf,vrm",
        "--force_override=true",
        "--num_runs=1",
        "--id=0",
        f"--faces_target_count={args.faces_target_count}",
        f"--time={time_str}",
        f"--input={input_path}",
        f"--output_dir={task_tmp}",
    ]
    run_command(extract_cmd, "步骤1/4: 提取网格")
    
    # ========== 步骤2: 预测骨骼 ==========
    skeleton_cmd = [
        sys.executable, "run.py",
        f"--task={args.skeleton_task}",
        f"--npz_dir={task_tmp}",
        f"--seed={args.seed}",
        f"--input={input_path}",
        f"--output_dir={task_tmp}",  # 输出到临时目录
        # 生成质量参数
        f"--num_beams={args.num_beams}",
        f"--max_new_tokens={args.max_new_tokens}",
        f"--repetition_penalty={args.repetition_penalty}",
    ]
    run_command(skeleton_cmd, "步骤2/4: 预测骨骼")
    
    # ========== 步骤3: 预测蒙皮 ==========
    skin_cmd = [
        sys.executable, "run.py",
        f"--task={args.skin_task}",
        f"--npz_dir={task_tmp}",
        f"--seed={args.seed}",
        "--data_name=predict_skeleton.npz",
        f"--input={input_path}",
        f"--output_dir={task_tmp}",  # 输出到临时目录
    ]
    run_command(skin_cmd, "步骤3/4: 预测蒙皮")
    
    # ========== 步骤4: 合并纹理（保留原始材质）==========
    # 查找预测结果文件（现在应该在临时目录下）
    predicted_fbx = task_tmp / input_name / "result_fbx.fbx"
    
    if not predicted_fbx.exists():
        # 备选路径
        alt_paths = [
            task_tmp / input_name / "predict.fbx",
            task_tmp / input_name / "skeleton.fbx",
        ]
        for p in alt_paths:
            if p.exists():
                predicted_fbx = p
                print(f"[信息] 找到预测结果: {p}")
                break
        else:
            print(f"[错误] 未找到预测结果 FBX 文件")
            print(f"[调试] 搜索路径:")
            print(f"  - {task_tmp / input_name / 'result_fbx.fbx'}")
            print(f"  - {task_tmp / input_name / 'predict.fbx'}")
            print(f"  - {task_tmp / input_name / 'skeleton.fbx'}")
            sys.exit(1)
    else:
        print(f"[信息] 找到预测结果: {predicted_fbx}")
    
    # 使用 transfer 合并：source=预测结果，target=原始模型
    merge_cmd = [
        sys.executable, "bpy_wrapper.py",
        "--module", "src.inference.merge",
        "--",
        "--require_suffix=fbx",  # 必需参数
        "--num_runs=1",          # 必需参数
        "--id=0",                # 必需参数
        f"--source={predicted_fbx}",   # 预测的骨骼蒙皮 FBX
        f"--target={input_path}",       # 原始模型（保留纹理）
        f"--output={output_path}",      # 最终输出
    ]
    run_command(merge_cmd, "步骤4/4: 合并纹理")
    
    # ========== 清理临时文件 ==========
    if not args.keep_temp:
        print("\n[清理] 删除临时文件...")
        shutil.rmtree(task_tmp, ignore_errors=True)
    
    print("\n" + "="*60)
    print("✅ 推理完成!")
    print(f"输出文件: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
