import torch


def get_loss(adj_est, gnd, beta):
    p = torch.ones_like(gnd)
    p_beta = beta*p
    p = torch.where(gnd == 0, p, p_beta)
    loss = torch.norm(torch.mul((adj_est - gnd), p), p='fro')**2

    return loss