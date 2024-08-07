import random
from lib.utils import *
from lib.metrics import *
import torch.optim as optim
from model.SAMG_LP import SAMG_LP_module
from model.loss import get_loss


# ==============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# ==============
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(28)

# ==============
## 读取数据
data_name = 'Haggle'
adj_seq = np.load('data/%s.npy' % data_name, allow_pickle=True)  # T_total*N*N

num_snaps = adj_seq.shape[0]  # 快照数量
num_nodes = adj_seq.shape[1]  # 节点数量

# ==============
## 常规配置
moving_step = 4  # 移动步长
decay_theta = 0.1  # 时间衰减函数参数
beta = 2  # 损失函数惩罚系数
win_size = 10  # 输入的快照数量
epsilon = 0.01  # 置零的阈值
num_epochs = 90  # 训练代数
num_samples = num_snaps - win_size  # 样本数
num_test_snaps = int(num_samples * 0.2)  # 测试样本数
num_val_snaps = int(num_samples * 0.1)  # 验证样本数
num_train_snaps = num_samples-num_test_snaps-num_val_snaps  # 训练样本数

# ==============
## 模型参数配置
num_heads = 2  # 注意力头数
dropout_rate = 0
FT_dims = [num_nodes, 32]
TAt_dims = [FT_dims[-1], 32]  # d_input, d_k
SAt_dims = [num_nodes, num_heads*TAt_dims[-1], 64, 32]  # num_nodes, F_in, F_m, F_out
GCN_SAt_dims = [num_heads*TAt_dims[-1], 32]  # in_feat, out_feat
AtF_dims = [GCN_SAt_dims[-1], 32]  # d_in, d_model
LP_dims = [AtF_dims[-1], 64, num_nodes]  # st_in, st_out

# ==============
## 加权移动平均法 + 时间衰减函数
### 原始矩阵
adj_tnr_seq_0 = []
for t in range(adj_seq.shape[0]):
    adj = adj_seq[t, :, :]  # N*N
    adj_tnr = torch.FloatTensor(adj).to(device)
    adj_tnr_seq_0.append(adj_tnr)

### 首次移动矩阵
adj_tnr_seq_1 = WGMA(adj_tnr_seq_0, moving_step, decay_theta, device)

### 二次移动矩阵
adj_tnr_seq_2 = WGMA(adj_tnr_seq_1, moving_step, decay_theta, device)

# ====================
## 定义模型
model = SAMG_LP_module(FT_dims, TAt_dims, SAt_dims, GCN_SAt_dims, AtF_dims, LP_dims, num_heads, dropout_rate, device).to(device)

# ==========
## 定义优化器
opt = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

# ====================
## 训练过程
for epoch in range(num_epochs):
    model.train()
    loss_list = []
    for k in range(num_train_snaps):
        adj_tnr_list_0 = adj_tnr_seq_0[k: k + win_size]
        adj_tnr_list_1 = adj_tnr_seq_1[k: k + (win_size - moving_step + 1)]
        adj_tnr_list_2 = adj_tnr_seq_1[k: k + ((win_size - moving_step + 1) - moving_step + 1)]
        ### ==========
        gnd = adj_seq[k + win_size, :, :]  # N*N
        gnd_tnr = torch.FloatTensor(gnd).to(device)
        ### ==========
        adj_est = model(adj_tnr_list_0, adj_tnr_list_1, adj_tnr_list_2)
        loss = get_loss(adj_est, gnd_tnr, beta)
        ### ==========
        opt.zero_grad()
        loss.backward()
        opt.step()
        ### ==========
        loss_list.append(loss.item())

    ### ==========
    loss_mean = np.mean(loss_list)
    print('Epoch %d Train Loss %f' % (epoch, loss_mean))

    # ====================
    # 验证过程
    model.eval()
    val_loss_list = []
    for k in range(num_train_snaps, num_train_snaps + num_val_snaps):
        adj_tnr_list_0 = adj_tnr_seq_0[k: k + win_size]
        adj_tnr_list_1 = adj_tnr_seq_1[k: k + (win_size - moving_step + 1)]
        adj_tnr_list_2 = adj_tnr_seq_1[k: k + ((win_size - moving_step + 1) - moving_step + 1)]
        ### ==========
        gnd = adj_seq[k + win_size, :, :]  # N*N
        gnd_tnr = torch.FloatTensor(gnd).to(device)
        ### ==========
        adj_est = model(adj_tnr_list_0, adj_tnr_list_1, adj_tnr_list_2)
        ### ==========
        #### 细化预测结果
        adj_est = (adj_est + adj_est.t()) / 2
        torch.diagonal(adj_est).fill_(0)
        adj_est[adj_est <= epsilon] = 0
        ### ==========
        val_loss = get_loss(adj_est, gnd_tnr, beta)
        val_loss_list.append(val_loss.item())
    ### ==========
    val_loss_mean = np.mean(val_loss_list)
    print('Epoch %d Val Loss %f' % (epoch, val_loss_mean))

    # ====================
    # 测试过程
    AUC_list = []
    MR_list = []
    Recall_list = []
    model.eval()
    for k in range(num_train_snaps + num_val_snaps, num_samples):
        adj_tnr_list_0 = adj_tnr_seq_0[k: k + win_size]
        adj_tnr_list_1 = adj_tnr_seq_1[k: k + (win_size - moving_step + 1)]
        adj_tnr_list_2 = adj_tnr_seq_1[k: k + ((win_size - moving_step + 1) - moving_step + 1)]
        ### ==========
        gnd = adj_seq[k + win_size, :, :]  # N*N
        ### ==========
        adj_est = model(adj_tnr_list_0, adj_tnr_list_1, adj_tnr_list_2)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        ### ==========
        #### 细化预测结果
        adj_est = (adj_est + adj_est.T) / 2
        np.fill_diagonal(adj_est, 0)
        adj_est[adj_est <= epsilon] = 0
        ### ==========
        AUC = get_AUC(adj_est, gnd, num_nodes)
        MR = get_MR(adj_est, gnd, num_nodes)
        Recall = get_Recall(adj_est, gnd, num_nodes)
        ### ==========
        AUC_list.append(AUC)
        MR_list.append(MR)
        Recall_list.append(Recall)
    ### ==========
    AUC_mean = np.mean(AUC_list)
    MR_mean = np.mean(MR_list)
    Recall_mean = np.mean(Recall_list)
    print('Epoch %d Test AUC %f MR %f Recall %f'
          % (epoch, AUC_mean, MR_mean, Recall_mean))
    print()
