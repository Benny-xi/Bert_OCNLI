from torch import nn
import torch

from transformers import BertModel
# class BertClassifier(nn.Module):
#
#     # 初始化加载 bert-base-chinese 原型，即Bert中的Bert-Base模型
#     def __init__(self, pretrained_name, num_classes):
#         super(BertClassifier, self).__init__()
#
#         # 定义 Bert 模型
#         self.bert = BertModel.from_pretrained(pretrained_name)
#
#         # 外接全连接层
#         self.mlp = nn.Linear(768, num_classes)
#
#     def forward(self, tokens_X):
#         # 得到最后一层的 '<cls>' 信息， 其标志全部上下文信息
#         res = self.bert(**tokens_X)
#
#         # res[1]代表序列的上下文信息'<cls>'，外接全连接层，进行情感分析
#         return self.mlp(res[1])

class BertClassificationHead(nn.Module):
    def __init__(self, hidden_size=768, num_classes=3, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_classes)  # (输入维度，输出维度)

    def forward(self, features, **kwargs):
        x = features[-1][:, 0, :]  # features[-1]是一个三维张量，其维度为[批次大小, 序列长度, 隐藏大小]。
        x = self.dropout(x)  # 这是一种正则化技术，用于防止模型过拟合。在训练过程中，它通过随机将输入张量中的一部分元素设置为0，来增加模型的泛化能力。
        x = self.dense(x)  # 这是一个全连接层，它将输入特征映射到一个新的特征空间。这是通过学习一个权重矩阵和一个偏置向量，并使用它们对输入特征进行线性变换来实现的，方便后续可以引入非线性变换。
        x = torch.tanh(x)  # 这是一个激活函数，它将线性层的输出转换为非线性，使得模型可以学习并表示更复杂的模式。
        x = self.dropout(x)  # 增加模型的泛化能力。
        x = self.out_proj(x)  # 这是最后的全连接层，它将特征映射到最终的输出空间。在这个例子中，输出空间的维度等于分类任务的类别数量。
        return x
        # x_1 = features[-1][:, 0, :]
        # x_2 = self.dropout(x_1)
        # x_2 = self.dense(x_2)
        # x_2 = torch.tanh(x_2)
        # x_2 = x_2 + x_1
        # x = self.dropout(x_2)
        # # x = self.dense(x)
        # x = self.out_proj(x)
        # return x


class BertClassifier(nn.Module):
    def __init__(self, pretrained_name, num_classes, dropout_prob=0.1, freeze_pooler=True, freeze_layers=False):
        super().__init__()
        self.freeze_pooler = False
        self.freeze_layers = False
        self.bert = BertModel.from_pretrained(pretrained_name, output_hidden_states=True)
        self.classifier = BertClassificationHead(
            hidden_size=self.bert.config.hidden_size,
            num_classes=num_classes,
            dropout_prob=dropout_prob
        )
        # self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(128, 768),freeze=True)

    def forward(self, input_ids, attention_mask, token_type_ids):

        # input_ids = input_ids + self.pos_emb(input_ids)

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.freeze_pooler:
            for param in self.bert.pooler.parameters():
                param.requires_grad = False  # 冻结 Pooler 层的参数

        if self.freeze_layers:
            num_layers_to_freeze = 6  # 冻结前6层
            for layer in self.bert.encoder.layer[:num_layers_to_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False  # 冻结层的参数

        logits = self.classifier(outputs.hidden_states)
        return logits