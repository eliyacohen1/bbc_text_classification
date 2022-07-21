from transformers import BertModel

from .embedding_model_interface import EmbeddingModelInterface


class BertEmbedding(EmbeddingModelInterface):

    def __init__(self, pretrained_name: str):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)

    def forward(self, input_id, mask):
        _, pooled = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        return pooled
