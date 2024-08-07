import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as Init
from torch.nn.parameter import Parameter


class SAMG_LP_module(nn.Module):
    def __init__(self, FT_dims, TAt_dims, SAt_dims, GCN_SAt_dims, AtF_dims, LP_dims, num_heads, dropout_rate, device):
        super(SAMG_LP_module, self).__init__()
        self.device = device
        self.FT_0 = Feat_transfer_layer(FT_dims[0], FT_dims[1], dropout_rate)
        self.FT_1 = Feat_transfer_layer(FT_dims[0], FT_dims[1], dropout_rate)
        self.FT_2 = Feat_transfer_layer(FT_dims[0], FT_dims[1], dropout_rate)

        self.TAt_0 = Temporal_Attention_layer(num_heads, TAt_dims[0], TAt_dims[1], dropout_rate)
        self.TAt_1 = Temporal_Attention_layer(num_heads, TAt_dims[0], TAt_dims[1], dropout_rate)
        self.TAt_2 = Temporal_Attention_layer(num_heads, TAt_dims[0], TAt_dims[1], dropout_rate)

        self.SAt_0 = Spatial_Attention_layer(SAt_dims[0], SAt_dims[1], SAt_dims[2], SAt_dims[3])
        self.SAt_1 = Spatial_Attention_layer(SAt_dims[0], SAt_dims[1], SAt_dims[2], SAt_dims[3])
        self.SAt_2 = Spatial_Attention_layer(SAt_dims[0], SAt_dims[1], SAt_dims[2], SAt_dims[3])

        self.GCN_SAt_0 = GCN_SAt_layer(GCN_SAt_dims[0], GCN_SAt_dims[1], dropout_rate)
        self.GCN_SAt_1 = GCN_SAt_layer(GCN_SAt_dims[0], GCN_SAt_dims[1], dropout_rate)
        self.GCN_SAt_2 = GCN_SAt_layer(GCN_SAt_dims[0], GCN_SAt_dims[1], dropout_rate)

        self.AtF = Attention_fusion_layer(AtF_dims[0], AtF_dims[1], dropout_rate)
        self.LP = Link_pre_layer(LP_dims[0], LP_dims[1], LP_dims[2], dropout_rate)


    def forward(self, adj_tnr_list_0, adj_tnr_list_1, adj_tnr_list_2):
        # ==========
        ## FT
        adj_tnr_feat_0 = self.FT_0(adj_tnr_list_0)
        adj_tnr_feat_1 = self.FT_1(adj_tnr_list_1)
        adj_tnr_feat_2 = self.FT_2(adj_tnr_list_2)
        # ==========
        ## TAt
        temporal_At_0 = self.TAt_0(adj_tnr_feat_0)
        temporal_At_1 = self.TAt_1(adj_tnr_feat_1)
        temporal_At_2 = self.TAt_2(adj_tnr_feat_2)
        # ==========
        ## SAt
        spatial_At_0 = self.SAt_0(temporal_At_0, adj_tnr_list_0)
        spatial_At_1 = self.SAt_1(temporal_At_1, adj_tnr_list_1)
        spatial_At_2 = self.SAt_2(temporal_At_2, adj_tnr_list_2)
        # ==========
        ## GCN
        spatial_gcn_0 = self.GCN_SAt_0(temporal_At_0, spatial_At_0, self.device)
        spatial_gcn_1 = self.GCN_SAt_1(temporal_At_1, spatial_At_1, self.device)
        spatial_gcn_2 = self.GCN_SAt_2(temporal_At_2, spatial_At_2, self.device)
        # ==========
        ## Multi-granularity feature fusion
        feat_At_fusion = self.AtF(spatial_gcn_0, spatial_gcn_1, spatial_gcn_2)
        # ==========
        ## output
        adj_est = self.LP(feat_At_fusion)

        return adj_est


class Feat_transfer_layer(nn.Module):
    def __init__(self, Ft_in, Ft_out, dropout_rate):
        super(Feat_transfer_layer, self).__init__()
        self.ft_linear = nn.Linear(Ft_in, Ft_out)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, adj_tnr_list):
        output_list = []
        for t in range(len(adj_tnr_list)):
            adj = adj_tnr_list[t]
            adj_tnr_feat = self.ft_linear(adj)
            temp_struc_output = self.dropout(adj_tnr_feat)
            struc_output = torch.relu(temp_struc_output)
            output_list.append(struc_output)
        output = torch.stack(output_list).permute(1, 0, 2)  # (T,B,Ft_out)->(B,T,Ft_out)

        return output


class Temporal_Attention_layer(nn.Module):
    def __init__(self, num_heads, d_input, d_k, dropout_rate):
        super(Temporal_Attention_layer, self).__init__()
        self.h = num_heads
        self.d_k = d_k
        self.d_output = num_heads * d_k

        self.q_linear = nn.Linear(d_input, self.d_output)
        self.k_linear = nn.Linear(d_input, self.d_output)
        self.v_linear = nn.Linear(d_input, self.d_output)
        self.out_linear = nn.Linear(self.d_output, self.d_output)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, adj_tnr_feat):
        x = adj_tnr_feat  # (B,T,Ft_out)
        B, T, _ = x.size()
        # ==========
        ## 将输入线性转换为查询(q)、键(k)和值(v)
        q = self.q_linear(x).view(B, T, self.h, self.d_k).transpose(1, 2)  # (B,T,h,d_k)->(B,h,T,d_k)
        k = self.k_linear(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        # ==========
        ## 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B,h,T,T)
        # ==========
        ## 使用softmax来获得注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (B,h,T,T)
        # ==========
        ## 使用注意力权重对值向量进行加权求和
        context = torch.matmul(attn_weights, v).transpose(1, 2). \
            contiguous().view(B, T, self.d_output)  # (B,T,d_output)
        # ==========
        ## 通过输出线性层进一步转换融合后的上下文向量
        temp_output = self.out_linear(context)  # (B,T,d_output)
        output = self.dropout(temp_output)[:, -1, :]  # (B,d_output)

        return output


class Spatial_Attention_layer(nn.Module):
    def __init__(self, num_nodes, F_in, F_m, F_out):
        super(Spatial_Attention_layer, self).__init__()
        self.W1 = Init.xavier_uniform_(Parameter(torch.FloatTensor(F_in, F_m)))
        self.W2 = Init.xavier_uniform_(Parameter(torch.FloatTensor(F_m, F_out)))
        self.W3 = Init.xavier_uniform_(Parameter(torch.FloatTensor(F_in, F_out)))
        self.bs = Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.Vs = Init.xavier_uniform_(Parameter(torch.FloatTensor(num_nodes, num_nodes)))

    def forward(self, temporal_At, adj_tnr_list):
        lhs = torch.matmul(torch.matmul(temporal_At, self.W1), self.W2)  # (B,F_in)(F_in,F_m)->(B,F_m)(F_m,F_out)->(B,F_out)
        rhs = torch.matmul(temporal_At, self.W3).transpose(0, 1)  # (B,F_in)(F_in,F_out)->(B,F_out)->(F_out,B)
        product = torch.matmul(lhs, rhs)  # (B,F_out)(F_out,B)->(B,B)
        scores = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (B,B)(B,B)->(B,B)
        # ==========
        attn_mask = torch.stack(adj_tnr_list).sum(dim=0)
        # attn_mask = torch.where(sum_result > 1, torch.tensor(1.0), sum_result)
        attn_scores = scores.masked_fill(attn_mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=0)  # (B,B)

        return attn_weights


class GCN_SAt_layer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout_rate):
        super(GCN_SAt_layer, self).__init__()
        self.W = Init.xavier_uniform_(Parameter(torch.FloatTensor(in_feat, out_feat)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat, A, device):
        A_hat = A + torch.eye(A.shape[0], device=device)
        D = torch.diag(torch.sum(A_hat, dim=1))
        D_sqrt_inv = torch.inverse(torch.sqrt(D))
        A_hat_norm = torch.matmul(torch.matmul(D_sqrt_inv, A_hat), D_sqrt_inv)

        feat_agg = torch.spmm(A_hat_norm, feat)  # (B,in_feat)
        agg_output = torch.relu(torch.matmul(feat_agg, self.W))  # (B,in_feat)(in_feat,out_feat)->(B,out_feat)
        temp_output = F.normalize(agg_output, dim=1, p=2)  # l2-normalization
        output = self.dropout(temp_output)  # (B,out_feat)

        return output


class Attention_fusion_layer(nn.Module):
    def __init__(self, d_in, d_model, dropout_rate):
        super(Attention_fusion_layer, self).__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_in, d_model)

        self.k0_linear = nn.Linear(d_in, d_model)
        self.v0_linear = nn.Linear(d_in, d_model)
        self.k1_linear = nn.Linear(d_in, d_model)
        self.v1_linear = nn.Linear(d_in, d_model)
        self.k2_linear = nn.Linear(d_in, d_model)
        self.v2_linear = nn.Linear(d_in, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat_0, feat_1, feat_2):
        # ==========
        ## 将输入线性转换为查询(q)、键(k)和值(v)
        q = self.q_linear(feat_0)  # (B,d_model)

        k0 = self.k0_linear(feat_0)  # (B,d_model)
        v0 = self.v0_linear(feat_0)  # (B,d_model)

        k1 = self.k1_linear(feat_1)  # (B,d_model)
        v1 = self.v1_linear(feat_1)  # (B,d_model)

        k2 = self.k2_linear(feat_2)  # (B,d_model)
        v2 = self.v2_linear(feat_2)  # (B,d_model)
        # ==========
        ## 计算注意力分数
        s0 = torch.matmul(q, k0.transpose(-2, -1)) / (self.d_model ** 0.5)  # (B,B)
        s1 = torch.matmul(q, k1.transpose(-2, -1)) / (self.d_model ** 0.5)  # (B,B)
        s2 = torch.matmul(q, k2.transpose(-2, -1)) / (self.d_model ** 0.5)  # (B,B)
        ## 使用softmax来获得注意力权重
        attn_w0 = F.softmax(s0, dim=-1)  # (B,B)
        attn_w1 = F.softmax(s1, dim=-1)  # (B,B)
        attn_w2 = F.softmax(s2, dim=-1)  # (B,B)
        # ==========
        ## 使用注意力权重对值向量进行加权求和
        context_0 = torch.matmul(attn_w0, v0)  # (B,B)(B,d_model)->(B,d_model)
        context_1 = torch.matmul(attn_w1, v1)  # (B,B)(B,d_model)->(B,d_model)
        context_2 = torch.matmul(attn_w2, v2)  # (B,B)(B,d_model)->(B,d_model)
        # ==========
        ## 特征相加
        temp_context = context_0 + context_1 + context_2  # (B,d_model)
        # ==========
        ## 通过输出非线性层进一步转化上下文向量
        context = self.out_linear(temp_context)  # (B,d_model)
        temp_output = torch.relu(context)
        output = self.dropout(temp_output)  # (B,d_model)

        return output


class Link_pre_layer(nn.Module):
    def __init__(self, Lp_in, Lp_m, Lp_out, dropout_rate):
        super(Link_pre_layer, self).__init__()
        self.Lp1_linear = nn.Linear(Lp_in, Lp_m)
        self.Lp2_linear = nn.Linear(Lp_m, Lp_out)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, feat_At_fusion):
        dec_output = self.Lp1_linear(feat_At_fusion)
        dec_output = self.dropout(dec_output)
        dec_output = torch.relu(dec_output)

        dec_output = self.Lp2_linear(dec_output)
        dec_output = torch.sigmoid(dec_output)
        adj_est = dec_output

        return adj_est