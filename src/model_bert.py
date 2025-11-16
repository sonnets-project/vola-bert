import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel

class Vola_BERT(nn.Module):
    """
    TODO: support more model backbones, e.g. BERT, Gopher, etc.

    Notation:
        B: batch size
        N: number of time series
        E: number of dimensions of embedding
        L: length of input time series (lookback horizon)
        Y: length of prediction time series (future forecast horizon)
    """
    
    def __init__(
        self,
        num_series: int,
        input_len: int,
        pred_len: int,
        n_layer: int,
        revin: bool = True,
        head_drop_rate=0.2,
        semantic_tokens: dict = None
    ):
        """
        Arguments:
            num_series (int)       : number of features, N
            input_len (int)        : length of input time series, lookback horizon, L
            pred_len (int)         : length of prediction time series, future forecast horizon, Y
            n_layer (int)          : number of transformer layers used in BERT
            revin (bool)           : whether to use RevIN
            head_drop_rate (float) : dropout rate for the forecast head
        """
        super().__init__()
        
        self.revin = revin
        self.patch_num = num_series
        self.input_len = input_len
        self.pred_len = pred_len
        self.n_layer = n_layer


        # BERT backbone
        self.model_type = "bert"
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.encoder.layer = self.bert.encoder.layer[:self.n_layer]
        self.n_embd = self.bert.config.hidden_size

        # Stage 1: Input Encoder
        self.wte = nn.Linear(self.input_len, self.n_embd)
        self.wte.apply(self._init_weights)
        
        # Stage 2: Fine-tuning BERT
        # freeze the multihead attention layers and feedforward
        # layers by default.
        need_to_freeze = ["attention.self.query", "attention.self.key", "attention.self.value",
                          "attention.output.dense", "intermediate.dense", "output.dense"]
        for n, p in self.bert.named_parameters():
            if any(k in n for k in need_to_freeze):
                p.requires_grad = False

        # Stage 3: Linear Probing
        self.head = nn.Linear(self.n_embd * (len(semantic_tokens) + 1), self.pred_len, bias=True)
        self.head.apply(self._init_weights)
        self.head_drop = nn.Dropout(head_drop_rate)
        

        # Semantic Token Embeddings
        semantic_token_embeddings = {}
        for semantic_token_name, n_tokens in semantic_tokens.items():
            _embedding = nn.Embedding(n_tokens, self.n_embd)
            _embedding.apply(self._init_weights)
            semantic_token_embeddings[semantic_token_name] = _embedding
        self.semantic_token_embeddings = nn.ModuleDict(semantic_token_embeddings)



    def _init_weights(self, module):
        """
        Initialises weights for train-from-scratch components. Mean and Std adopted from BERT's training procedure
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)



    def encoder(self, x: torch.Tensor, input_tokens: dict):
        """
        Compute the output of the transformer encoder. This includes embeddings from numerical features
        as well as the semantic tokens
        """

        # input (B, N, L)

        # semantic token embeddings
        embedding_list = []
        for token_name in self.semantic_token_embeddings:
            token_vals = input_tokens[token_name]
            embedding_list.append(self.semantic_token_embeddings[token_name](token_vals).unsqueeze(1))  # (B, 1, E)
        
        # feature embeddings
        embedding_list.append(self.wte(x))  # B, N, E


        tok_emb = torch.cat(embedding_list, dim=1)  # (B, N+S, E)



        h = self.bert(inputs_embeds=tok_emb, attention_mask=None).last_hidden_state  # (B, N+S, E)
        
        return h  # B, N, E

    def forward(self, x_data: tuple):
        """
        Compute the output given the input data.

        Arguments:
            x_data (tuple): input data, shape (B, N, L) for the numereical time series, (B,) for tokens
        """

        x, input_tokens = x_data
        
        # norm
        if self.revin:
            x = x.permute(0, 2, 1)  # B, L, N
            
            # normalization
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            x = x.permute(0, 2, 1)  # B, N, L

        # encoder (Stage 1+2)
        h = self.encoder(x, input_tokens)  # B, N+S, E
        B, _, _ = h.shape


        # selects imporatnt tokens
        n_semantic_tokens = len(self.semantic_token_embeddings)
        token_indices = list(range(n_semantic_tokens)) + [-1]
        
        h = h[:, token_indices , :].reshape(B, (n_semantic_tokens + 1) * self.n_embd)  # (B, (S+1)*E)

        
        h = self.head_drop(h)
        
        dec_out = self.head(h)  # (B, pred_len)

        
        if self.revin:
            # denormalisation
            target_stdev = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)[:, :, -1]
            target_mean = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)[:, :, -1]
            dec_out = dec_out * target_stdev
            dec_out = dec_out + target_mean
        
        return dec_out.unsqueeze(1) # (B, 1, pred_len)

    @property
    def num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        n_params_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": n_params, "grad": n_params_grad}