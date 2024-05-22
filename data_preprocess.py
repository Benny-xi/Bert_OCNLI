from torch.utils.data import Dataset
import pandas as pd


class OcnliDataset(Dataset):
    labels = ["entailment", "neutral", "contradiction"]

    label_map = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }

    levels = ["na", "easy", "medium", "hard"]

    level_map = {
        "na": 0,
        "easy": 1,
        "medium": 2,
        "hard": 3
    }

    def __init__(
        self,
        data_path,
        test_mode=False,
        s1_prompt="{0}",
        s2_prompt="{0}"
    ):
        self.test_mode = test_mode
        self.data = pd.read_json(data_path, lines=True)
        # self.data.columns = ["level", "sentence1", "sentence2", "label"]
        self.data["sentence1"] = [s1_prompt.format(s) for s in self.data["sentence1"]]
        self.data["sentence2"] = [s2_prompt.format(s) for s in self.data["sentence2"]]
        if not self.test_mode:
            self.data = self.data[self.data["label"].isin(self.labels)]
            self.data["label"] = self.data["label"].map(self.label_map)
            self.data["level"] = self.data["level"].map(self.level_map)

        self.data = self.data.to_dict(orient="records")

    # 定义了一个  __len__方法（魔法方法）：当使用len()函数作用于一个对象时候，该对象会返回它的长度。
    def __len__(self):
        return len(self.data)

    # 定义了一个  __getitem__方法（魔法方法）：当使用索引访问一个对象的元素时,该对象应该如何返回该元素。
    def __getitem__(self, idx):
        if self.test_mode:
            return self.data[idx]["sentence1"], self.data[idx]["sentence2"]

        return (
            self.data[idx]["sentence1"],
            self.data[idx]["sentence2"],
            self.data[idx]["label"],
            self.data[idx]["level"]
        )