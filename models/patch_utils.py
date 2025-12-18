#
# SPDX-FileCopyrightText: 2025 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Utility to patch HuggingFace CausalLM models to support loss_reduction parameter.

This modification is required for contrastive perplexity training, which needs
per-token losses instead of the default mean-reduced loss.

Original HuggingFace implementation uses:
    loss_fct = CrossEntropyLoss()  # hardcoded reduction="mean"

Modified implementation:
    loss_fct = CrossEntropyLoss(reduction=loss_reduction)

This allows calling model(..., loss_reduction="none") to get per-token losses.
"""

from functools import wraps
import torch
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
import warnings


def patch_causal_lm_for_loss_reduction(model):
    """
    Patches a HuggingFace CausalLM model to support loss_reduction parameter.
    
    This function modifies the model's forward method to accept a loss_reduction
    parameter, enabling per-token loss computation required for contrastive
    perplexity training.
    
    Args:
        model: A HuggingFace CausalLM model instance (e.g., LlamaForCausalLM,
               MistralForCausalLM, etc.)
    
    Returns:
        The same model instance with patched forward method
    
    Example:
        >>> from transformers import MistralForCausalLM
        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> model = patch_causal_lm_for_loss_reduction(model)
        >>> # Now you can call:
        >>> output = model(..., loss_reduction="none")
    """
    
    # Store the original forward method
    original_forward = model.forward
    
    @wraps(original_forward)
    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        loss_reduction="mean",  # NEW PARAMETER
        **kwargs
    ):
        """
        Forward pass with support for configurable loss reduction.
        
        Additional Args:
            loss_reduction (str): Specifies the reduction to apply to the loss.
                                 Can be 'none', 'mean', or 'sum'.
                                 Default: 'mean' (standard HuggingFace behavior)
        """
        
        # Call the original forward to get all outputs
        outputs = original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # Don't compute loss in original forward
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # If labels are provided, compute custom loss
        loss = None
        if labels is not None:
            # Get logits from outputs
            logits = outputs.logits if return_dict else outputs[0]
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction=loss_reduction)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # Update outputs with the new loss
            if return_dict:
                outputs.loss = loss
            else:
                outputs = (loss,) + outputs
        
        return outputs
    
    # Replace the forward method
    # Use __get__ to bind the method to the instance
    model.forward = patched_forward.__get__(model, model.__class__)
    
    # Mark that this model has been patched
    model._loss_reduction_patched = True
    
    return model


def is_patched(model):
    """Check if a model has been patched for loss_reduction support."""
    return getattr(model, '_loss_reduction_patched', False)


def apply_patch_if_needed(model):
    """
    Apply patch only if not already patched.
    Useful for scripts that might load models multiple times.
    """
    if not is_patched(model):
        return patch_causal_lm_for_loss_reduction(model)
    else:
        warnings.warn("Model already patched for loss_reduction support.")
        return model
