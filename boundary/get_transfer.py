import numpy as np
import torch
import argparse

def transfer(v, y,degree):
    # 计算投影的系数
    v = torch.load(v)
    v = v[0]
    y = torch.load(y)
    y = y[0]
    
    # 计算反射向量
    transfer = y - v * degree
    return transfer


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--degree", type=float, default=None, help="")
    parser.add_argument("--name", type=str, default=None, help="")
    parser.add_argument("--choose1", type=str, default=None, help="")
    parser.add_argument("--choose2", type=str, default=None, help="")

    args = parser.parse_args()
    result = transfer(f"./editings/celeba_directions/{args.name}_linear_{args.choose1}.pth",f"./editings/celeba_directions/{args.name}_linear_{args.choose2}.pth",args.degree)
    result = torch.nn.functional.normalize(result,dim = -1)
    torch.save(result,f"./editings/celeba_directions/{args.name}_linear_{args.choose2}_{args.choose1}_transfer_{args.degree}.pth")


