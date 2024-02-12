from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
)
import logging
import torch.nn as nn
import torch

from models.fastformer.fastformer_wu import StandardFastformerEncoder


class AttentionPooling(nn.Module):
    """
    Implements an attention pooling layer based on a self-attention mechanism.

    This layer takes a 3D tensor of shape (batch_size, sequence_length, hidden_size) and
    computes a weighted sum along the sequence_length dimension, producing a 2D tensor of
    shape (batch_size, hidden_size). The weights for the summation are computed using a
    simple self-attention mechanism.

    Attributes:
        att_fc1 (nn.Linear): Linear layer used to compute attention scores.
        att_fc2 (nn.Linear): Linear layer used to compute attention scores.
    """

    def __init__(self, config):
        """
        Initializes the AttentionPooling layer with the given config.
        Args:
            config: Configuration object containing layer size and initialization settings.
        """
        self.config = config
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """
        Initializes the weights of the linear layers with a normal distribution,
        and biases with zeros.
        Args:
            module (nn.Module): The module whose weights are to be initialized.
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, attn_mask=None) -> torch.Tensor:
        """
        Forward pass through the attention pooling layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            attn_mask (torch.Tensor, optional): Optional mask tensor of shape
                                                (batch_size, sequence_length). Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size), obtained by
                            computing a weighted sum of the input tensor along the
                            sequence_length dimension, using self-attention.
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x


class FastSelfAttention(nn.Module):
    """Implements a fast self-attention layer in PyTorch.
    This layer utilizes a simplified self-attention mechanism to accelerate computation.
    It divides the input into multiple heads, computes the attention scores, and applies
    the attention to the input values. The results from all heads are then concatenated
    and linearly transformed into the final output.
    """

    def __init__(self, config):
        super(FastSelfAttention, self).__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim = config.hidden_size

        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transposes the input tensor for computing self-attention scores.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
        Returns:
            torch.Tensor: The transposed tensor of shape (batch_size, num_heads, sequence_length, head_dim).
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes the forward pass of the fast self-attention layer.
        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): The attention mask of shape (batch_size, 1, sequence_length).
        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, hidden_size),
                            obtained by applying self-attention to the input tensor.
        """

        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        query_for_score = (
            self.query_att(mixed_query_layer).transpose(1, 2)
            / self.attention_head_size**0.5
        )
        query_for_score += attention_mask
        query_weight = self.softmax(query_for_score).unsqueeze(2)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        pooled_query = (
            torch.matmul(query_weight, query_layer)
            .transpose(1, 2)
            .view(-1, 1, self.num_attention_heads * self.attention_head_size)
        )
        pooled_query_repeat = pooled_query.repeat(1, seq_len, 1)
        mixed_query_key_layer = mixed_key_layer * pooled_query_repeat

        query_key_score = (
            self.key_att(mixed_query_key_layer) / self.attention_head_size**0.5
        ).transpose(1, 2)

        query_key_score += attention_mask

        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        weighted_value = (pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2]
            + (self.num_attention_heads * self.attention_head_size,)
        )
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        return weighted_value


class FastAttention(nn.Module):
    """Implements a fast attention layer using FastSelfAttention and BertSelfOutput.
    This layer first applies a FastSelfAttention mechanism on the input tensor,
    then processes the self-attention output through a BertSelfOutput layer to produce
    the final attention output.
    """

    def __init__(self, config):
        super(FastAttention, self).__init__()
        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        """Computes the forward pass of the fast attention layer.

        It applies the FastSelfAttention mechanism on the input tensor using the provided
        attention mask, then processes the resulting self-attention output through the
        BertSelfOutput layer to produce the final attention output.
        Args:
            input_tensor (torch.Tensor): The input tensor of shape
                                            (batch_size, sequence_length, hidden_dimension).
            attention_mask (torch.Tensor): The attention mask of shape
                                            (batch_size, 1, sequence_length).
        Returns:
            torch.Tensor: The final attention output tensor.
        """
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class FastformerLayer(nn.Module):
    """Implements a Fastformer layer comprising FastAttention, BertIntermediate, and BertOutput layers.
    This layer first processes the input through a FastAttention mechanism, then passes the attention output
    through a BertIntermediate layer followed by a BertOutput layer to produce the final layer output.
    """

    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes the forward pass of the Fastformer layer.
        Processes the input through a FastAttention mechanism, followed by a BertIntermediate layer,
        and finally through a BertOutput layer to produce the final layer output.
        Args:
            hidden_states (torch.Tensor): The input tensor of shape
                                            (batch_size, sequence_length, hidden_dimension).
            attention_mask (torch.Tensor): The attention mask of shape
                                            (batch_size, 1, sequence_length).
        Returns:
            torch.Tensor: The final layer output tensor.
        """
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SequenceFastformerEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(SequenceFastformerEncoder, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList(
            [FastformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config.pooler_type == "weightpooler":
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(config))
        logging.info(f"This model has {len(self.poolers)} poolers.")
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_embs, attention_mask, pooler_index=0) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Note, we do er '.view()' in the implementation, meaning, that the shapes are taken care of.

        Parameters:
            input_embs (torch.Tensor): The input embeddings, with shape (batch_size, history_size, n_tokens, hidden_dimension).
            attention_mask (torch.Tensor): The attention mask, with shape (batch_size, n_tokens),
                                                        where values of 1 indicate positions to attend to and 0s indicate positions to mask.
            pooler_index (int, optional): Index of the pooler to use to aggregate the encoder's output. Default is 0.
        Returns:
            torch.Tensor: The output of the encoder, processed and pooled according to the specified pooler.
                            with shape (batch_size, config.hidden_size).
        Usage:
        >>> encoder_output = model.forward(input_embs, attention_mask, pooler_index=0)
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # JK IMP:
        batch_size, history_size, n_tokens, hidden_dim = input_embs.shape

        position_ids = torch.arange(
            history_size, dtype=torch.long, device=input_embs.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        # JK IMP:
        position_embeddings = position_embeddings.unsqueeze(2)

        embeddings = input_embs + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        # JK IMP:
        # Here, the '-1' is equivalent to (seq_length * n_tokens)
        embeddings = embeddings.view(batch_size, -1, hidden_dim)
        extended_attention_mask = extended_attention_mask.view(batch_size, 1, -1)
        attention_mask = attention_mask.view(batch_size, -1)

        all_hidden_states = [embeddings]

        for layer_module in self.encoders:
            # This loop iterates through each encoder layer in self.encoders.
            # In each iteration, it passes the output of the previous layer (all_hidden_states[-1])
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)

        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)
        return output


class Fastformer(nn.Module):
    """Implements the Fastformer model using a custom architecture.

    This model processes user click-history and candidate articles through a series of encoding,
    attention, and transformation layers to generate a relevance score indicating the likelihood
    that the user will click on the candidate articles.
    """

    def __init__(
        self,
        config,
        word_embedding: nn.Embedding = None,
    ):
        """Initializes the Fastformer model with the specified configuration and word embedding layer.
        Args:
            config: Configuration object containing model parameters.
            word_embedding (nn.Embedding): Embedding layer for token embeddings.
        """

        super(Fastformer, self).__init__()
        self.config = config

        if word_embedding is None:
            self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        else:
            self.word_embedding = word_embedding

        self.embedding_transform = nn.Linear(
            self.word_embedding.weight.shape[1], config.hidden_size
        )
        self.output_layer = nn.Linear(config.hidden_size * 2, 1)
        self.news_encoder = SequenceFastformerEncoder(config)
        self.user_attention_polling = AttentionPooling(config)
        # self.news_encoder_standard = StandardFastformerEncoder(config)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def user_encoder(self, history_input: torch.Tensor) -> torch.Tensor:
        """Encodes the user click-history to produce a user representation.
        Args:
            history_input (torch.Tensor): Tensor of shape (batch_size, history_size, n_tokens)
                                            representing the user click-history.
        Returns:
            torch.Tensor: User representation tensor of shape (batch_size, config.hidden_size).
        """
        attention_mask_history_input = history_input.bool().float()
        embds_history_input = self.word_embedding(history_input)
        embds_history_input = self.embedding_transform(embds_history_input)
        if self.news_encoder.__class__.__name__ == "SequenceFastformerEncoder":
            # This might be a bit slow, the idea is to apply the news_encoder for each
            # article in histroy. We then run a attention layer (re-use the news-encoder)
            # This would be the same as running the 'StandardFastformerEncoder'
            attention_mask_tokens = attention_mask_history_input[:, 0, :]
            attention_mask_history = attention_mask_history_input[:, :, 0]
            outputs = []
            for i in range(embds_history_input.size(1)):
                slice_input = embds_history_input[:, i, :, :].unsqueeze(1)
                self.news_encoder(slice_input, attention_mask_tokens)
                slice_output = self.news_encoder(
                    slice_input, attention_mask_tokens.unsqueeze(0)
                )
                outputs.append(slice_output.unsqueeze(1))
                # self.news_encoder_standard(slice_input, attention_mask_tokens)
            user_embeddings = torch.cat(outputs, dim=1)
            user_encodings = self.user_attention_polling(
                user_embeddings, attention_mask_history
            )
        else:
            # Alternative approach: here we apply the attention on a final concated
            user_encodings = self.news_encoder(
                embds_history_input, attention_mask_history_input
            )
        return user_encodings

    def forward(self, history_input, candidate_input) -> torch.Tensor:
        """Computes the forward pass of the Fastformer model.
        Processes the user click-history and candidate articles, and produces a relevance score.
        Args:
            history_input (torch.Tensor): Tensor of shape (batch_size, history_size, n_tokens)
                                            representing the user click-history.
            candidate_input (torch.Tensor): Tensor of shape (batch_size, 1, n_tokens)
                                            representing the candidate articles.
        Returns:
            torch.float: A tensor of shape (batch_size, 1) representing the relevance scores.
        """

        # ====
        # output: (batch_size, hidden_dimension)
        user_encoding = self.user_encoder(history_input)

        # ====
        attention_mask_candidate_input = candidate_input.bool().float()
        embds_candidate_input = self.word_embedding(candidate_input)
        embds_candidate_input = self.embedding_transform(embds_candidate_input)
        # output: (batch_size, hidden_dimension)
        candidate_encoding = self.news_encoder(
            embds_candidate_input, attention_mask_candidate_input
        )

        # ====
        concat_representation = torch.concat([user_encoding, candidate_encoding], dim=1)
        score = self.output_layer(concat_representation)
        return torch.sigmoid(score)
