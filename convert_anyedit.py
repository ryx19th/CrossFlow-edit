#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from tqdm import tqdm


# 需要丢掉的字段
DROP_KEYS = {"height", "width", "ratio", "aspect_ratio"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="把 list 格式的 JSON 标注文件转换成 JSONL，每行一个样本。"
    )
    parser.add_argument(
        "input",
        help="输入的 JSON 文件路径（顶层是 list，每个元素是一个样本字典）",
    )
    parser.add_argument(
        "output",
        help="输出的 JSONL 文件路径（每行一个 json 对象）",
    )
    return parser.parse_args()


def convert_item(item: dict) -> dict:
    """
    按规则转换单个样本：
    - 删除 height / width / ratio / aspect_ratio
    - path      -> img
    - path_src  -> img_src
    - 其他 key 原样保留（包括有空格的 key，比如 "edited object"）
    - 值中的 None 会在 json.dump 时自动变成 null
    """
    new_item = {}

    for key, value in item.items():
        # 丢掉不需要的字段
        if key in DROP_KEYS:
            continue

        if key == "path":
            new_item["img"] = value
        elif key == "path_src":
            new_item["img_src"] = value
        else:
            # 其他字段原样保留
            new_item[key] = value

    return new_item


def main():
    args = parse_args()

    # 读取原始 JSON（必须是 list）
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list。")

    total = len(data)
    print(f"总样本数: {total}")

    # 写出为 jsonl，每行一个样本
    with open(args.output, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, total=total, desc="Converting", unit="sample"):
            new_item = convert_item(item)
            json.dump(new_item, out_f, ensure_ascii=False)
            out_f.write("\n")


if __name__ == "__main__":
    main()
