from tqdm import tqdm
import numpy as np
import torch

from ebrec.utils._python import get_torch_device

try:
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("torch not available")
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("transformers not available")


def get_transformers_word_embeddings(model: AutoModel):
    return model.embeddings.word_embeddings.weight.data.to("cpu").numpy()


def generate_embeddings_with_transformers(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    text_list: list[str],
    batch_size: int = 8,
    device: str = None,
    disable_tqdm: bool = False,
) -> torch.Tensor:
    """
    Generates embeddings for a list of texts using a pre-trained transformer model.

    Args:
        model_name (str): The name of the pre-trained transformer model to use.
        text_list (list of str): A list of texts to generate embeddings for.
        batch_size (int): The batch size to use for generating embeddings. Defaults to 8.
        device (str): The device to use for generating embeddings (e.g., "cpu", "cuda").
            If None, defaults to the first available GPU or CPU.

    Returns:
        embeddings (torch.Tensor): A tensor containing the embeddings for the input texts.
            The shape of the tensor is (num_texts, embedding_dim), where num_texts is the number
            of input texts and embedding_dim is the dimensionality of the embeddings produced by
            the pre-trained model.

    Examples:
    >>> model_name = "bert-base-uncased"
    >>> text_list = ["hello world", "how are you"]
    >>> batch_size = 2
    >>> device = "cpu"
    >>> model = AutoModel.from_pretrained(model_name)
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name)
    >>> embeddings_tensor = generate_embeddings_with_transformers(model, tokenizer, text_list, batch_size, device)
    >>> print(embeddings_tensor)
        tensor([[-0.0243,  0.1144,  0.0830,  ..., -0.2666,  0.1662,  0.1519],
                [ 0.0827,  0.0877, -0.0688,  ..., -0.4381,  0.0462, -0.1446]])
    >>> print(embeddings_tensor.shape)
        torch.Size([2, 768])
    """
    device = get_torch_device(use_gpu=True) if device is None else device
    model = model.to(device)

    tokenized_text = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    feature_names = list(tokenized_text)

    dataset = TensorDataset(
        tokenized_text["input_ids"], tokenized_text["attention_mask"]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding", disable=disable_tqdm):
            inputs = {feat: t.to(device) for feat, t in zip(feature_names, batch)}
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze(dim=1))
    return torch.vstack(embeddings)


if __name__ == "__main__":
    #
    model_name = "xlm-roberta-base"
    batch_size = 8
    text_list = [
        "hej med dig. Jeg er en tekst.",
        "Jeg er en anden tekst, skal du spille smart?",
        "oh nej..",
    ]
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t = generate_embeddings_with_transformers(
        model, tokenizer, text_list, batch_size, "cpu"
    )
