#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="把 list 格式的 JSON 标注文件转换成 JSONL，每行一个样本。"
    )
    parser.add_argument(
        "input",
        help="输入的 JSON 文件路径（里面是一个 list，每个元素是一个样本字典）",
    )
    parser.add_argument(
        "output",
        help="输出的 JSONL 文件路径（每行一个 json 对象）",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="如果没有 tqdm，每处理多少个样本打印一次进度（默认 10000）",
    )
    return parser.parse_args()


def convert_item(item):
    """
    把原始样本转换成目标格式：
    原始:
        {
          "height": 480,
          "width": 640,
          "ratio": 1.3333333333333333,
          "path_src": "0cec95678583_src.jpg",
          "prompt": "A flamingo is drinking from water.",
          "path": "0cec95678583_tgt.jpg"
        }

    新格式:
        {
          "img": "0cec95678583_tgt.jpg",
          "img_src": "0cec95678583_src.jpg",
          "prompt": "A flamingo is drinking from water."
        }
    """
    return {
        "img": item["path"],          # 目标图
        "img_src": item["path_src"],  # 源图
        "prompt": item["prompt"],     # 文本
    }


def main():
    args = parse_args()

    # 读取原始 JSON（list 格式）
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list。")

    total = len(data)
    print(f"总样本数: {total}")

    with open(args.output, "w", encoding="utf-8") as out_f:
        # 有 tqdm 就用进度条
        if tqdm is not None:
            for item in tqdm(
                data,
                total=total,
                desc="Converting",
                unit="sample",
            ):
                new_item = convert_item(item)
                json.dump(new_item, out_f, ensure_ascii=False)
                out_f.write("\n")
        else:
            # 没有 tqdm 就手动打印进度 + ETA
            print("未检测到 tqdm，将使用简单进度打印（可用 `pip install tqdm` 安装 tqdm）。")
            start = time.time()
            for i, item in enumerate(data, start=1):
                new_item = convert_item(item)
                json.dump(new_item, out_f, ensure_ascii=False)
                out_f.write("\n")

                if i % args.log_interval == 0 or i == total:
                    elapsed = time.time() - start
                    rate = i / elapsed if elapsed > 0 else float("inf")
                    remaining = total - i
                    eta = remaining / rate if rate > 0 else float("inf")

                    pct = i / total * 100 if total > 0 else 0.0
                    print(
                        f"[{i}/{total} - {pct:.2f}%] "
                        f"elapsed: {elapsed:.1f}s, "
                        f"ETA: {eta:.1f}s, "
                        f"{rate:.1f} samples/s"
                    )


if __name__ == "__main__":
    main()
