import numpy as np
from collections import Counter

class FaceRecognizer:
    def __init__(self,k=3):
        self.k=3
        self.embeddings=None
        self.labels=None

    def train(self, embeddings, names):
        self.embeddings = self._normalize(np.array(embeddings))
        self.labels = np.array(names)

    def predict(self, embedding):
            if self.embeddings is None or self.labels is None:
                raise ValueError("Model not trained.")

            embedding = self._normalize(embedding.reshape(1, -1))
            similarities = np.dot(self.embeddings, embedding.T).flatten()

            k_indices = similarities.argsort()[-self.k:][::-1]
            k_labels = self.labels[k_indices]

            print("Top K similarities:", similarities[k_indices])
            print("Top K labels:", k_labels)

            if len(k_labels) == 0:
                return "UNKNOWN"

            most_common = Counter(k_labels).most_common(1)[0][0]
            return most_common
        

    @staticmethod
    def _normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)




