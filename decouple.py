import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class SharedEncoder(nn.Module):
    def __init__(self, input_dim=768, shared_dim=768, dropout=0.3):
        super(SharedEncoder, self).__init__()

        # 直接编码交集
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        """
        输入两个特征，输出它们的公有特征
        保持输入输出形状一致：[B, L, D]
        """
        # 计算公有特征作为A和B的交集
        # 可以使用各种交集计算方法：

        # 方法1: 拼接编码（最直接）
        concatenated = torch.cat([x1, x2], dim=-1)
        shared_feat = self.encoder(concatenated)

        # 方法2: 门控交集（可选）
        # gate = torch.sigmoid(self.gate_layer(torch.cat([x1, x2], dim=-1)))
        # shared_feat = gate * x1 + (1 - gate) * x2

        return shared_feat  # [B, L, D]

# 私有编码器（新增 Dropout 和 ReLU）
class PrivateEncoder(nn.Module):
    def __init__(self, input_dim=768, private_dim=384, dropout=0.3):
        super(PrivateEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, private_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)


class DecoupleModule(nn.Module):
    def __init__(self, input_dim=768, shared_dim=768, private_dim=384):
        super(DecoupleModule, self).__init__()
        # 共享编码器
        self.shared_encoder = SharedEncoder(input_dim, 768).to(device)
        # 私有编码器
        self.private_encoder_A = PrivateEncoder(input_dim, private_dim).to(device)
        self.private_encoder_B = PrivateEncoder(input_dim, private_dim).to(device)
        self.proj_dif1 = nn.Linear(384, 768)



    def forward(self, img_feat_A, img_feat_B, CPP):

        shared_feat = self.shared_encoder(img_feat_A, img_feat_B)  # (batch, seq_len, shared_dim)
        # 提取私有特征
        private_feat_A = self.private_encoder_A(img_feat_A)  # (batch, seq_len, private_dim)
        private_feat_B = self.private_encoder_B(img_feat_B)  # (batch, seq_len, private_dim)

        # 计算损失
        loss = self._compute_loss(shared_feat, private_feat_A, private_feat_B, img_feat_A, img_feat_B,
                                   CPP)
        dif = torch.cat([private_feat_B, private_feat_A], dim=-1)

        return dif, loss

    def _compute_loss(self, shared_feat, private_feat_A, private_feat_B, original_feat_A,
                      original_feat_B, changeflag):
        """
        计算总损失函数，使用余弦相似度的绝对值
        Args:
            changeflag: 0表示变化，1表示未变化
        """

        # 共享特征损失 - 使用绝对余弦相似度
        # 我们希望共享特征尽量相似，所以最大化|cos_sim|
        # loss_shared = 1.0 - torch.abs(F.cosine_similarity(shared_feat_A, shared_feat_B)).mean()

        # 重构损失 - 共享特征应该能重构原始特征
        loss_recon_A = 1.0 - torch.abs(F.cosine_similarity(shared_feat, original_feat_A)).mean()
        loss_recon_B = 1.0 - torch.abs(F.cosine_similarity(shared_feat, original_feat_B)).mean()

        private_sim = torch.abs(F.cosine_similarity(
            private_feat_A.flatten(1),
            private_feat_B.flatten(1),
            dim=1
        )).mean()  # 值在[0, 1]之间

        loss_private_0 = private_sim * (changeflag == 0).float()
        loss_private_1 = (1.0 - private_sim) * (changeflag == 1).float()

        loss_private = (loss_private_0 + loss_private_1).mean()

        proj_private_A = self.proj_dif1(private_feat_A)
        proj_private_B = self.proj_dif1(private_feat_B)

        orth_loss = torch.abs(F.cosine_similarity(shared_feat, proj_private_A)).mean() + \
                    torch.abs(F.cosine_similarity(shared_feat, proj_private_B)).mean()

        # 总损失
        total_loss = loss_recon_A + loss_recon_B + loss_private + 0.3 * orth_loss

        # 可选：返回各项损失用于监控
        loss_dict = {
            'total': total_loss.item(),

            'recon_A': loss_recon_A.item(),
            'recon_B': loss_recon_B.item(),
            'private': loss_private.item(),
            'orth': orth_loss.item()
        }

        return total_loss

    def _spectral_mse(self, x, y):
        """光谱敏感型MSE损失"""
        # 光谱维度压缩 (B,L,D) -> (B,L,16)
        x_band = x[..., :16]  # 取前16维作为关键波段表征
        y_band = y[..., :16]
        return 0.7 * F.mse_loss(x_band, y_band) + 0.3 * F.mse_loss(x.mean(dim=-1), y.mean(dim=-1))

    def _band_aware_sim(self, x, y):
        bands = [0, 3, 5, 7]  # Sentinel-2关键波段
        return F.cosine_similarity(
            x[..., bands].mean(dim=-1),  # 波段平均
            y[..., bands].mean(dim=-1),
            dim=-1
        ).mean()  # 最终返回标量

        # ---------------------------- fusion -------------------------------------


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入维度384，输出维度768
        self.input_dim = 384
        self.output_dim = 768

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.input_dim * 2,  # cat_feat的维度 768
            num_heads=8,  # 8个头
            batch_first=True  # 输入格式为 (batch, seq, feature)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(self.input_dim * 2)
        self.norm2 = nn.LayerNorm(self.input_dim * 2)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.input_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.input_dim * 4, self.input_dim * 2),
            nn.Dropout(0.1)
        )

        # 输出投影到目标维度
        self.output_proj = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, dif1, dif, img_a, img_b):
        # 计算dif2
        dif2 = img_a - img_b  # (1, 49, 384)

        # 构建cat_feat和dif2的查询、键、值
        cat_feat = torch.cat([dif1, dif], dim=-1)  # (1, 49, 768)

        # 方案1: dif2作为Query, cat_feat作为Key和Value
        # 这会让dif2从cat_feat中检索相关信息
        attended_dif2, attention_weights = self.cross_attention(
            query=dif2,  # Query: dif2 (1, 49, 384)
            key=cat_feat,  # Key: cat_feat (1, 49, 768)
            value=cat_feat,  # Value: cat_feat (1, 49, 768)
            need_weights=False
        )

        # 残差连接 + 层归一化
        attended_dif2 = self.norm1(attended_dif2 + dif2)

        # 方案2: 也可以反过来，cat_feat作为Query, dif2作为Key和Value
        attended_cat_feat, _ = self.cross_attention(
            query=cat_feat,  # Query: cat_feat
            key=attended_dif2,  # Key: 增强后的dif2
            value=attended_dif2,  # Value: 增强后的dif2
            need_weights=False
        )

        # 残差连接 + 层归一化
        attended_cat_feat = self.norm2(attended_cat_feat + cat_feat)

        # 前馈网络
        output = self.ffn(attended_cat_feat)
        output = self.ffn(attended_dif2)

        # 最终投影到目标维度
        # output = self.output_proj(output)

        return output



class TEXT_Cross(nn.Module):
    def __init__(self, txt_dim=512, img_dim=768, output_len=5):
        super().__init__()
        # 文本特征投影层
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, img_dim),
            nn.LayerNorm(img_dim)
        )
        # 双向注意力交互
        self.cross_attn = nn.MultiheadAttention(img_dim, num_heads=8, batch_first=True)
        # 输出形状调整
        self.output_proj = nn.Linear(img_dim * 2, img_dim)
        self.pos_embed = nn.Parameter(torch.randn(output_len, img_dim))

    def forward(self, img_feat, txt_feat):
        """
        Inputs:
            img_feat: [batch, 98, 768]
            txt_feat: [batch, 512]
        Outputs:
            fused_feat: [batch, 5, 768]
        """
        # 文本特征形状调整 [batch,512] -> [batch,98,768]
        txt_proj = self.txt_proj(txt_feat).unsqueeze(1)  # [batch,1,768]
        txt_proj = txt_proj.expand(-1, 98, -1)  # [batch,98,768]

        # 双向注意力交互
        img_enhanced, _ = self.cross_attn(
            query=img_feat,
            key=txt_proj,
            value=txt_proj
        )
        txt_enhanced, _ = self.cross_attn(
            query=txt_proj,
            key=img_feat,
            value=img_feat
        )

        # 特征融合与输出调整
        fused = torch.cat([img_enhanced, txt_enhanced], dim=-1)  # [batch,98,1536]
        fused = self.output_proj(fused)  # [batch,98,768]

        # 自适应生成5个token
        output = fused.mean(dim=1, keepdim=True) + self.pos_embed.unsqueeze(0)  # [batch,5,768]
        return output


class Text_self(nn.Module):
    def __init__(self, txt_dim=512, output_len=5, hidden_dim=768):
        super().__init__()
        # 文本编码器
        self.encoder = nn.Sequential(
            nn.Linear(txt_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        # 动态token生成器
        self.token_gen = nn.Linear(hidden_dim * 2, output_len * hidden_dim)
        # 输出精炼层
        self.refiner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=2
        )

    def forward(self, txt_feat):
        """
        Input:
            txt_feat: [batch, 512]
        Output:
            output: [batch, 5, 768]
        """
        # 特征压缩与扩展
        hidden = self.encoder(txt_feat)  # [batch, 1536]
        tokens = self.token_gen(hidden).view(-1, 5, 768)  # [batch,5,768]

        # 时序特征精炼
        output = self.refiner(tokens)  # [batch,5,768]
        return output