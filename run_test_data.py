import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer

from dataloader.mlm_loader_for_val import MLMDataLoader, MaskedMultiSentenceDataLoader,MultiSentencesDataLoader
from data_preprocess import OcnliDataset
from model.bertClassifier import BertClassifier


parser = argparse.ArgumentParser(description='Bert-OCNLI')
parser.add_argument('--model_name', type=str, default='./pretrained/bert-base-chinese')
parser.add_argument('--test_path', type=str, default='./data/test.json')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--exp-name', type=str, required=False)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.)
# parser.add_argument('--freeze_pooler', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=5)

args = parser.parse_args()

if args.exp_name is None:
    args.exp_name = \
        f'{os.path.basename(args.model_name)}' + \
        f'_b{args.batch_size}_e{args.epochs}' + \
        f'_len{args.max_length}_lr{args.lr}'


def test():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # result_path = 'ocnli_50k_predict.json'  # 直接使用文件名作为路径

    test_dataset = OcnliDataset(args.test_path, test_mode=True, s1_prompt='{0}', s2_prompt='{0}')

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    test_dataloader = MultiSentencesDataLoader(
        tokenizer=tokenizer,
        dataset=test_dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        shuffle=False,
        drop_last=False,
        test_mode=True,
    )
    model = BertClassifier(pretrained_name=args.model_name, num_classes=args.num_classes).to(args.device)

    state_dict = torch.load(os.path.join(args.checkpoint_dir, "best_model_256.pt"))
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        test_preds = []
        with tqdm(test_dataloader, desc="Test", unit="batch") as t:
            for input_ids, attention_mask, token_type_ids in t:
                logits = model(input_ids, attention_mask, token_type_ids)
                output = logits.argmax(dim=-1)
                test_preds.append(output)
        # for input_ids, attention_mask, token_type_ids in test_dataloader:
        #     outputs = model(input_ids, attention_mask, token_type_ids)
        #     output = outputs.argmax(dim=-1).item()
        #     test_preds.append(output)

    print("len(test_preds):", len(test_preds))
    # 将预测结果展平为一维列表
    test_preds = [pred.item() for preds in test_preds for pred in preds]

    test_df = pd.read_json(args.test_path, lines=True)

    test_df["label"] = test_preds
    test_df["label"] = test_df["label"].map({0: "entailment", 1: "neutral", 2: "contradiction"})
    test_df = test_df[["label", "id"]]
    test_df.to_json(os.path.join(args.checkpoint_dir, f"ocnli_50k_predict.json"), orient="records", lines=True,
                    force_ascii=False)

    os.system(f"zip -j {args.checkpoint_dir}/ocnli_50k_predict_3.zip {args.checkpoint_dir}/ocnli_50k_predict.json")


if __name__ == '__main__':
    test()