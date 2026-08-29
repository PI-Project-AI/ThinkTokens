"""
Tie code token input embeddings to VQ codebook.

This ensures code semantics come from VQ quantization, not separate learned embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def tie_code_embeddings_to_codebook(
    model,
    vq_module,
    ir_token_ids: dict
):
    """
    Tie code token input embeddings to VQ codebook.

    After calling this, code tokens will use VQ codebook vectors as their embeddings.

    Args:
        model: Base language model with embeddings
        vq_module: ProjectionVQ module with codebook
        ir_token_ids: Dict with code token ranges
    """
    code_start = ir_token_ids['code_start']
    code_end = ir_token_ids['code_end']
    num_codes = code_end - code_start + 1

    # Get codebook
    codebook = vq_module.vq.codebook.weight  # (num_codes, code_dim)

    # Normalize if needed
    if vq_module.vq.use_unit_norm:
        codebook = F.normalize(codebook, dim=1)

    # Get embedding layer
    embed_layer = model.gpt_neox.embed_in

    # Check dimensions
    embed_dim = embed_layer.weight.shape[1]
    code_dim = codebook.shape[1]

    if embed_dim != code_dim:
        # Need projection: code_dim → embed_dim
        projection = nn.Linear(code_dim, embed_dim, bias=False).to(codebook.device)

        # Initialize projection to preserve norms
        with torch.no_grad():
            nn.init.xavier_uniform_(projection.weight)

        # Project codebook to embedding dimension
        code_embeddings = projection(codebook)  # (num_codes, embed_dim)
    else:
        # Direct copy
        code_embeddings = codebook

    # Copy to embedding layer
    with torch.no_grad():
        embed_layer.weight[code_start:code_end+1] = code_embeddings

    # Register hook to keep them tied during training
    def embedding_sync_hook(grad):
        """Keep code embeddings synced with codebook."""
        # During backward, update VQ codebook based on embedding gradients
        # This is optional - may want to only update via VQ loss
        return grad

    # Optionally register hook on code embeddings
    # embed_layer.weight[code_start:code_end+1].register_hook(embedding_sync_hook)

    print(f"Tied {num_codes} code embeddings to VQ codebook")
    print(f"Code token range: [{code_start}, {code_end}]")
    print(f"Codebook shape: {codebook.shape}")
    print(f"Embedding dimension: {embed_dim}")


class CodeEmbeddingTier(nn.Module):
    """
    Wrapper that maintains tied embeddings between code tokens and VQ codebook.

    Can be used to periodically re-sync embeddings during training.
    """

    def __init__(self, model, vq_module, ir_token_ids: dict):
        super().__init__()
        self.model = model
        self.vq_module = vq_module
        self.ir_token_ids = ir_token_ids

        self.code_start = ir_token_ids['code_start']
        self.code_end = ir_token_ids['code_end']

        # Projection if dimensions don't match
        embed_dim = model.gpt_neox.embed_in.weight.shape[1]
        code_dim = vq_module.vq.codebook.weight.shape[1]

        if embed_dim != code_dim:
            self.projection = nn.Linear(code_dim, embed_dim, bias=False)
            nn.init.xavier_uniform_(self.projection.weight)
        else:
            self.projection = None

    def sync_embeddings(self):
        """Sync code embeddings with current VQ codebook."""
        codebook = self.vq_module.vq.codebook.weight

        if self.vq_module.vq.use_unit_norm:
            codebook = F.normalize(codebook, dim=1)

        if self.projection is not None:
            code_embeddings = self.projection(codebook)
        else:
            code_embeddings = codebook

        with torch.no_grad():
            self.model.gpt_neox.embed_in.weight[self.code_start:self.code_end+1] = code_embeddings

    def forward(self):
        """Can be called periodically to re-sync embeddings."""
        self.sync_embeddings()


if __name__ == "__main__":
    print("Code embedding tying module loaded.")
    print("Use tie_code_embeddings_to_codebook() to sync embeddings with VQ codebook.")
