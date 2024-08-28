# SparseMemoryEfficientAdam

`SparseMemoryEfficientAdam` 是一個基於 Adam 優化器的記憶體友好且稀疏的優化器。它結合了自動預熱試探法和長期學習速率緩衝，並且在每次迭代中只更新一部分參數，以減少計算量和記憶體占用，從而加快訓練速度。

## 特點

- **記憶體友好**：優化器狀態（如 `exp_avg` 和 `exp_avg_sq`）以 `bf16` 格式儲存，以減少記憶體占用。
- **稀疏更新**：在每次迭代中，只更新一部分參數，而不是所有參數，以減少計算量和記憶體占用。
- **自動預熱試探法**：在訓練初期，學習率從一個較小的值逐步增加到初始學習率。
- **長期學習速率緩衝**：在訓練後期，學習率逐步降低，以提高訓練的穩定性和效果。
- **參數區塊分配**：參數分成區塊，每個區塊有自己的優化器狀態和學習率。

## 安裝

目前，`SparseMemoryEfficientAdam` 可以直接從 GitHub 克隆並使用。未來我們計劃將其發布到 PyPI。

```bash
git clone https://github.com/yourusername/SparseMemoryEfficientAdam.git
cd SparseMemoryEfficientAdam
使用方法
以下是如何在你的 PyTorch 項目中使用 SparseMemoryEfficientAdam 的示例：


import torch
import torch.nn as nn
import torch.optim as optim
from sparse_memory_efficient_adam import SparseMemoryEfficientAdam

# 定義一個簡單的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# 初始化模型、損失函數和優化器
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = SparseMemoryEfficientAdam(model.parameters(), lr=1e-3, sparsity_ratio=0.1)

# 訓練循環
for epoch in range(10):
    # 假設我們有一些訓練數據
    inputs = torch.randn(32, 10)
    targets = torch.randint(0, 2, (32,))

    # 前向傳播
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向傳播和優化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


參數
lr (float): 學習率，默認為 1e-3。
betas (tuple[float, float]): 用於計算梯度和梯度平方的指數衰減率，默認為 (0.9, 0.999)。
eps (float): 為了數值穩定性而添加到分母的項，默認為 1e-8。
weight_decay (float): 權重衰減，默認為 1e-2。
block_size (int): 參數區塊的大小，默認為 1024。
sparsity_ratio (float): 稀疏更新的比例，默認為 0.1。
warmup_steps (int): 預熱步數，默認為 1000。
warmup_factor (float): 預熱因子，默認為 0.1。
lr_decay_steps (int): 學習率衰減步數，默認為 10000。
lr_decay_factor (float): 學習率衰減因子，默認為 0.1。
貢獻
歡迎對 SparseMemoryEfficientAdam 進行貢獻！如果你有任何問題或建議，請提交 Issue 或 Pull Request。

許可證
本項目採用 Apache-2.0 許可證。詳見 LICENSE 文件。

聯繫
如果你有任何問題或建議，請聯繫 win10ogod。
