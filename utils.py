import joblib
import os
import numpy as np

class IncrementalLabelEncoder:
    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = {}
        self.next_id = 0

    def partial_fit(self, items):
        """增量学习：只添加没见过的元素"""
        for item in items:
            if item not in self.label_to_id:
                self.label_to_id[item] = self.next_id
                self.id_to_label[self.next_id] = item
                self.next_id += 1
        return self

    def transform(self, items):
        """将元素转换为 ID，遇到未知的报错（理论上不应发生，因为fit过了）"""
        return np.array([self.label_to_id.get(item, 0) for item in items])

    def inverse_transform(self, ids):
        """将 ID 转换回元素"""
        return np.array([self.id_to_label.get(i, "Unknown") for i in ids])

    def __len__(self):
        return self.next_id

    def save(self, path):
        joblib.dump(self.__dict__, path)

    def load(self, path):
        if os.path.exists(path):
            state = joblib.load(path)
            self.__dict__.update(state)
        return self