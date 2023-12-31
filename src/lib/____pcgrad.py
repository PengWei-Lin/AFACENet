"""PCGrad PyTorch
WIP implementation of https://github.com/tianheyu927/PCGrad in PyTorch
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import random

from operator import mul
from functools import reduce

def pc_grad_update(gradient_list):
  '''
  PyTorch implementation of PCGrad.
  Gradient Surgery for Multi-Task Learning: https://arxiv.org/pdf/2001.06782.pdf
  Arguments:
    gradient_list (Iterable[Tensor] or Tensor): an iterable of Tensorsthat will 
    have gradients with respect to parameters for each task.
  Returns:
    List of gradients with PCGrad applied.
  '''

  assert type(gradient_list) is list
  assert len(gradient_list) != 0
  num_tasks = len(gradient_list)
  num_params = len(gradient_list[0])
  np.random.shuffle(gradient_list)
  # gradient_list = torch.stack(gradient_list)

  # grad_dims = []

  def flatten_and_store_dims(grad_task):
    output = []
    grad_dim = []
    for param_grad in grad_task: # TODO(speedup): convert to map since they are faster
      grad_dim.append(tuple(param_grad.shape))
      output.append(torch.flatten(param_grad))

    # grad_dims.append(grad_dim)

    return torch.cat(output), grad_dim

  # gradient_list = list(map(flatten_and_store_dims, gradient_list))

  def restore_dims(grad_task, chunk_dims):
    ## chunk_dims is a list of tensor shapes
    chunk_sizes = [reduce(mul, dims, 1) for dims in chunk_dims]
    
    grad_chunk = torch.split(grad_task, split_size_or_sections=chunk_sizes)
    resized_chunks = []
    for index, grad in enumerate(grad_chunk): # TODO(speedup): convert to map since they are faster
      grad = torch.reshape(grad, chunk_dims[index])
      resized_chunks.append(grad)

    return resized_chunks

  def project_gradients(grad_task):
    """
    Subtracts projected gradient components for each grad in gradient_list
    if it conflicts with input gradient.
    Argument:
      grad_task (Tensor): A tensor for a gradient
    Returns:
      Component subtracted gradient
    """
    grad_task, grad_dim = flatten_and_store_dims(grad_task)

    for k in range(num_tasks): # TODO(speedup): convert to map since they are faster
      conflict_gradient_candidate = gradient_list[k]
      # no need to store dims of candidate since we are not changing it in the array
      conflict_gradient_candidate, _ = flatten_and_store_dims(conflict_gradient_candidate)
      
      inner_product = torch.dot(torch.flatten(grad_task), torch.flatten(conflict_gradient_candidate))
      # TODO(speedup): put conflict check condition here so that we aren't calculating norms for non-conflicting gradients
      proj_direction = inner_product / torch.norm(conflict_gradient_candidate)**2
      
      ## sanity check to see if there's any conflicting gradients
      # if proj_direction < 0.:
      #   print('conflict')
      # TODO(speedup): This is a cumulative subtraction, move to threaded in-memory map-reduce
      grad_task = grad_task - min(proj_direction, 0.) * conflict_gradient_candidate
    
    # get back grad_task
    grad_task = restore_dims(grad_task, grad_dim)
    return grad_task

  flattened_grad_task = list(map(project_gradients, gradient_list))

  yield flattened_grad_task

