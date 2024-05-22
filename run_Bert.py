import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer

from dataloader.mlm_loader_for_val import MLMDataLoader, MaskedMultiSentenceDataLoader, MultiSentencesDataLoader
from data_preprocess import OcnliDataset
from model.bertClassifier import BertClassifier



parser = argparse.ArgumentParser(description='Bert-OCNLI')
parser.add_argument('--model_name', type=str, default='./pretrained/bert-base-chinese')
parser.add_argument('--train_path', type=str, default='./data/train.50k.json')
parser.add_argument('--dev_path', type=str, default='./data/dev.json')
parser.add_argument('--test_path', type=str, default='./data/test.json')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--exp-name', type=str, required=False)
parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.)
# parser.add_argument('--freeze_pooler', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_length', type=int, default=256)  #128
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--patience', type=int, default=5)

args = parser.parse_args()

if args.exp_name is None:
    args.exp_name = \
        f'{os.path.basename(args.model_name)}' + \
        f'_b{args.batch_size}_e{args.epochs}' + \
        f'_len{args.max_length}_lr{args.lr}'




def train():
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_dataset = OcnliDataset(args.train_path, s1_prompt='{0}', s2_prompt='{0}')
    val_dataset = OcnliDataset(args.dev_path, s1_prompt='{0}', s2_prompt='{0}')

    # 初始化一个 BERT 分词器,使用指定的预训练模型。有了这个 tokenizer 实例,就可以很方便地对输入文本进行各种预处理操作,
    # 为送入 BERT 模型做好准备
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    """
    tokenizer.tokenize(text): 将输入文本分词为单词列表。
    tokenizer.convert_tokens_to_ids(tokens): 将单词列表转换为对应的 token ID 列表。
    tokenizer.convert_ids_to_tokens(ids): 将 token ID 列表转换回单词列表。
    """

    # dataloader = MLMDataLoader(
    #     tokenizer=tokenizer,
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     max_length=args.max_length,
    #     shuffle=True,
    #     drop_last=True,
    #     device=device,
    #     tokenizer_name=args.model_name
    # )
    train_dataloader = MaskedMultiSentenceDataLoader(
        tokenizer=tokenizer,
        dataset=train_dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        shuffle=True,
    )

    val_dataloader = MultiSentencesDataLoader(
        tokenizer=tokenizer,
        dataset=val_dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        shuffle=True,
    )

    model = BertClassifier(pretrained_name=args.model_name, num_classes=args.num_classes).to(args.device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')  # 或者设置一个合理的初始值
    best_checkpoints = [{"val_acc": 0.0, "state_dict": None}] * 1

    for epoch in range(args.epochs):

        train_loss = 0
        train_accuracy = 0
        model.train()
        for input_ids, attention_mask, token_type_ids, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            outputs = outputs.argmax(dim=1)
            accuracy = (outputs == labels).float().mean()

            train_loss += loss.item()
            train_accuracy += accuracy.item()

        train_loss /= len(train_dataloader)
        train_accuracy /= len(train_dataloader)

        print(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}')

        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt')


        #### eval

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, labels in val_dataloader:
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * input_ids.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

        # Check if we need to save the model and if early stopping is needed
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model_256.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print('Early stopping triggered.')
                break



if __name__ == '__main__':

    train()