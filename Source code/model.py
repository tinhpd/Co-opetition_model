import time
import numpy as np
import torch
import pandas as pd
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class inside_competitive_dynamics_model():
    def __init__(
        self,
        N, # Số nút trong mạng
        W,
        W_att, # Trọng số liên kết của các thuộc tính
        num_agent,
        alpha, # Hệ số lan truyền ảnh hưởng
        gamma, # Hệ số cạnh tranh 
        lamda, # Hệ số hợp tác
        decay, # Hệ số suy giảm ảnh hưởng 
        id_node_to_gene, # Chuyển từ id node trong mạng tới tên gene
        gene_to_node_id, # Chuyển từ tên gene trong mạng tới id node
                ):
        
        self.N = N
        self.W = torch.tensor(W, dtype=torch.float64, device=device).T
        self.W_att = W_att
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda
        self.decay = torch.tensor(decay, dtype=torch.float64, device=device)
        
        # Từ điển ánh xạ node id đến tên hoặc id gene 
        self.id_node_to_gene = id_node_to_gene
        self.gene_to_node_id = gene_to_node_id

        self.num_agent = num_agent
        # Khởi tạo ảnh hưởng của các tác nhân
        self.influence_all = torch.ones((num_agent, N), dtype=torch.float64, device=device) * 1e-10

        self.time_start = time.time()
        self.rank_table = pd.DataFrame()
        
    def train(self, callback, epsilon = 1e-14):   
        one = torch.tensor(1, dtype=torch.float64, device=device)  
        delta = torch.tensor(1, dtype=torch.float64, device=device)  
        epsilon = torch.tensor(epsilon, dtype=torch.float64, device=device)
        while delta > epsilon:
            # print(delta)
        
            # Tổng ảnh hưởng
            ifl_sum = self.influence_all.sum(dim=0) 
            
            # Chỉ số ảnh hưởng của các thuộc tính với nhau
            diff = self.W_att @ self.influence_all
            term = self.influence_all * (1 - self.gamma.unsqueeze(1) * torch.minimum(diff, one)) + self.lamda.unsqueeze(1) * torch.abs(1 - diff)

            # Chỉ số ảnh hươngr mới
            influence_all_new = self.influence_all * (1 - self.decay) + self.alpha.unsqueeze(1) * term @ self.W                                  # shape: [N, D]

            # Mẫu số chuẩn hóa
            S = influence_all_new.sum(dim=1)            
            avg_S = S.sum() / self.N

            # Tính delta
            sum_I_new = influence_all_new.sum(dim=0)    
            delta     = (ifl_sum - sum_I_new / avg_S).abs().max()
            callback(delta)
            # Cập nhật
            self.influence_all = influence_all_new / avg_S              # broadcasting over rows
        self.influence = self.influence_all.sum(dim=0)
        self.time_end = time.time()
        print(self.time_end - self.time_start)
        
    def result(self):
        self.influence = self.influence / self.influence.sum().abs()
        scores = self.influence.detach().cpu().numpy()
        genes = [self.id_node_to_gene[i] for i in range(self.N)]
        self.top_df = pd.DataFrame({
            "Gene": genes,
            "Score": scores,
        }).sort_values(by="Score", ascending=False)
        ranking = []
        pred_score = -10
        rank = 0
        for score in np.sort(scores)[::-1]:
            if abs(score - pred_score) < 1e-10:
                ranking.append(rank)
            else:
                rank += 1
                ranking.append(rank)
            pred_score = score
        self.top_df['Ranking'] = ranking
        return self.top_df

    
