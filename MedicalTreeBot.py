import numpy as np

class MedicalTreeBot:
    def __init__(self, model, feature_names, class_names):
        self.tree = model.tree_
        self.feature_names = feature_names
        self.class_names = class_names
        self.current_node = 0

    def get_question(self):
        feature_idx = self.tree.feature[self.current_node]
        return self.feature_names[feature_idx] if feature_idx != -2 else None

    def submit_answer(self, answer):
        if answer == 1:
            self.current_node = self.tree.children_right[self.current_node]
        else:
            self.current_node = self.tree.children_left[self.current_node]

    def is_leaf(self):
        return self.tree.children_left[self.current_node] == -1

    def get_result(self):
        values = self.tree.value[self.current_node][0]
        total = np.sum(values)
        top_indices = np.argsort(values)[-3:][::-1]
        
        results = []
        for idx in top_indices:
            if values[idx] > 0:
                results.append({
                    "disease": self.class_names[idx],
                    "confidence": f"{(values[idx] / total) * 100:.2f}%"
                })
        return results