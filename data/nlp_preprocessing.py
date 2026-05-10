import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:

        combined_list = positive + negative

        tokens = set()
        for sentence in combined_list:
            for word in sentence.split():
                tokens.add(word)

        tokens_list = sorted(tokens)

        dictionary = {}
        for i in range (len(tokens_list)):
            dictionary[tokens_list[i]] = i+1

        result = []
        for sentence in combined_list:
            a = []
            for word in sentence.split():
                a.append(dictionary[word])
            result.append(a)

        tensor_list = []
        for a in result:
            tensor_list.append(torch.tensor(a))


        return torch.nn.utils.rnn.pad_sequence(tensor_list, batch_first=True)
