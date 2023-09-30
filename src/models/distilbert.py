# ===============[ IMPORTS ]===============
import torch
import torch.nn as nn
import transformers


# ===============[ DISTILBERT ]===============
class CustomDistilBert(nn.Module):
    def __init__(self):
        super().__init__()

        self.distilbert = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.pred_layer = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def get_distilbert(self):
        return self.distilbert

    def get_tokenizer(self):
        return self.tokenizer

    def get_pred_layer(self):
        return self.pred_layer

    def get_sigmoid(self):
        return self.sigmoid

    def get_criterion(self):
        return self.criterion

    def assign_optimizer(self, **kwargs):
        """
        This assigns the Adam optimizer to this model's parameters (self) and returns the
        optimizer.

        Parameters
        ----------
        **kwargs
            The arguments passed to the optimizer.

        Returns
        -------
        torch optimizer
            An Adam optimizer bound to the model
        """

        return torch.optim.Adam(self.parameters(), lr=kwargs["lr"])

    def slice_cls_hidden_state(self, x: transformers.modeling_outputs.BaseModelOutput) -> torch.Tensor:
        """
        Parameters
        ----------
        x: transformers BaseModelOutput
            The output of the distilbert model.
            The last hidden state has shape: [batch_size, sequence_length, hidden_size]

        Returns
        -------
        torch.Tensor of shape [batch_size, hidden_size]
            The last layer's hidden state representing the [CLS] token. Usually, CLS
            is the first token in the sequence.
        """
        return x.last_hidden_state[:, 0, :]

    def tokenize(
        self,
        premise: "list[str]",
        hypothesis: "list[str]",
        max_length: int = 128,
        truncation: bool = True,
        padding: bool = True,
    ):
        """
        This function will be applied to the premise and hypothesis (list of str)
        to obtain the inputs for the model.

        Parameters
        ----------
        premise: list of str
            The first text to be input in the model.
        hypothesis: list of str
            The second text to be input in the model.

        For the remaining params, see:
            https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__

        Returns
        -------
        transformers.BatchEncoding
            A dictionary-like object that can be directly given to the model with **
        """

        return self.tokenizer(
            premise, hypothesis, max_length=max_length, truncation=truncation, padding=padding, return_tensors="pt"
        )

    def forward(self, inputs: transformers.BatchEncoding):
        """
        Parameters
        ----------
        inputs: transformers.BatchEncoding
            The input ingested by our model. Output of tokenizer for a given batch

        Returns
        -------
        tensor of shape [batch_size]
            The output prediction for each element in the batch, with sigmoid
            activation. Make sure the shape is not [batch_size, 1]
        """

        x = self.distilbert(**inputs)
        x = self.slice_cls_hidden_state(x)
        x = self.pred_layer(x)
        x = self.sigmoid(x)
        x = torch.reshape(x, (-1,))

        return x
