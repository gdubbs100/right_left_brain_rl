import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extend_tensor(tensor, max_len):
    """
    Extends tensors on dimension zero to a given max_len
    Only extends on 0 dimension
    Use as input for torch RNN with batch_first = False
    """
    if tensor.size(0) >= max_len:
        base = tensor
    elif tensor.size(0) == 0:
        base = torch.zeros(*[max_len,*tensor.size()[1:]])
    else:
        base = torch.zeros(*[max_len,*tensor.size()[1:]])
        base[-tensor.size(0):] = tensor.clone()
        
    return base.to(device)
    
def build_sequence_from_storage(storage_data, next_value, max_len):
    """
    combines data from storage with latest observation
    Should return tensor with most recent observation last
    any missing values will be padded with prior zeros
    """

    extended = extend_tensor(storage_data, max_len)
    return torch.cat((extended, next_value), dim = 0)