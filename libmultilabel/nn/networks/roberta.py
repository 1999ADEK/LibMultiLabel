import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class RoBERTa(nn.Module):
    """RoBERTa.

    Args:
        num_classes (int): Total number of classes.
        encoder_hidden_dropout (float): The dropout rate of the feed forward sublayer in each BERT layer. Defaults to 0.1.
        encoder_attention_dropout (float): The dropout rate of the attention sublayer in each BERT layer. Defaults to 0.1.
        post_encoder_dropout (float): The dropout rate of the dropout layer after the BERT model. Defaults to 0.
        lm_weight (str): Pretrained model name or path. Defaults to 'roberta-base'.
        lm_window (int): Length of the subsequences to be split before feeding them to
            the language model. Defaults to 512.
    """

    def __init__(
        self,
        num_classes,
        encoder_hidden_dropout=0.1,
        encoder_attention_dropout=0.1,
        post_encoder_dropout=0,
        lm_weight="roberta-base",
        lm_window=512,
        **kwargs,
    ):
        super().__init__()
        self.lm_window = lm_window

        self.lm = RobertaModel.from_pretrained(
            lm_weight,
            hidden_dropout_prob=encoder_hidden_dropout,
            attention_probs_dropout_prob=encoder_attention_dropout,
        )

        self.classifier = nn.Linear(self.lm.config.hidden_size, num_classes)

        self.post_encoder_dropout = nn.Dropout(post_encoder_dropout) if post_encoder_dropout else nn.Identity()

    def forward(self, input):
        input_ids = input["text"]  # (batch_size, sequence_length)
 
        if input_ids.size(-1) > self.lm.config.max_position_embeddings:
            # Support for sequence length greater than 512 is not available yet
            raise ValueError(
                f"Got maximum sequence length {input_ids.size(-1)}, "
                f"please set max_seq_length to a value less than or equal to "
                f"{self.lm.config.max_position_embeddings}"
            )

        outputs = self.lm(input_ids, attention_mask=input["attention_mask"])
        sequence_output = self.post_encoder_dropout(outputs[0][:, 0, :])
        logits = self.classifier(sequence_output)

        return {"logits": logits}
