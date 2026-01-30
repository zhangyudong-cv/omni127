import os
from typing import Optional
import torch

from utils.build import *
from adaptation.pl_generator import PseudoLabelGenerator
from adaptation.trainer import Trainer
from adaptation.strategies import WithMemStrategy, NoMemStrategy
from utils.callbacks import CheckpointCB, LRCB, LoggerCB
from utils.io import ensure_dir, clear_dir
from utils.lr_scheduler import build_poly_warmup_scheduler

class Engine:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_builder = None
        self.trainer = None

        self.class_names = None
        self.num_classes = None
        self.is_warmup = None

        self.target_pl_loader = None
        self.source_loader = None
        self.val_loader = None
        self.target_loader = None

        self.model_t = None
        self.model_s = None
        self.save_dir = None

        self.global_step = 0
        self.best_mIoU = 0.0
        self.best_ckpt_path: Optional[str] = None

    def setup(self):
        self.data_builder = DatasetBuilder(self.args)
        self.target_pl_loader, self.class_names, self.num_classes, frame_cnt = self.data_builder.build_tgt_for_pl()

        self.model_t = build_model(self.args, self.num_classes, self.args.num_maskmem, self.device)
        self.model_s = build_model(self.args, self.num_classes, self.args.num_maskmem, self.device)

        load_checkpoint(self.model_t, self.args.tgt_ckpt)
        load_checkpoint(self.model_s, self.args.src_ckpt)

        criteria = self.data_builder.build_criteria(self.device, self.args.dataset)
        optimizer = torch.optim.AdamW(self.model_t.parameters(), lr=self.args.lr, weight_decay=1e-4)

        scheduler = build_poly_warmup_scheduler(optimizer, power=1.0, warmup_iters=1800)

        mem_strategy = WithMemStrategy() if self.args.dataset != 'Stanford2D3D' else NoMemStrategy()
# “如果是跑全景图（WildPASS/DensePASS），请开启‘显存记忆’功能，让模型把切片连起来看；如果是跑 Stanford2D3D，就把每张图当成独立的，别用记忆功能。”
        
        callbacks = [
            CheckpointCB(save_dir=self._save_dir(), backbone=self.args.backbone),
            LRCB(scheduler),
            LoggerCB()
        ]
# DatasetBuilder:
# 这套代码的逻辑非常“独特”：它把 dataset='Cityscapes' 当作了一个 “套餐名字”。只要你指定了 'Cityscapes'，
# 代码内部已经写死了你要用的全套数据配置：源域是 Cityscapes，目标域是 WildPASS，验证集是 DensePASS。
# 不用改任何参数，它原本就是为你现在的需求设计的。
#         套餐一：dataset='Cityscapes' (你正在用的)
# 这是典型的 “真实到真实 (Real-to-Real)” 街景适应。
# 源域 (Source): Cityscapes (德国街景，有标签)
# 代码: build_src -> 读取 data_root/cityscapes
# 目标域训练 (Target Train): WildPASS (全球全景街景，无标签)
# 代码: build_tgt_for_pl -> 读取 data_root/WildPASS
# 目标域验证 (Target Val): DensePASS (全景街景，有标签)
# 代码: build_tgt_from_pd -> 读取 data_root/DensePASS
# 适用场景: 用标准的 Cityscapes 模型，去适应 WildPASS 这种全景相机的畸变。
# 📦 套餐二：dataset='SynPASS'
# 这是典型的 “合成到真实 (Sim-to-Real)” 街景适应。
# 源域 (Source): SynPASS (虚拟合成的街景，有标签)
# 代码: build_src -> 读取 data_root/SynPASS
# 目标域训练 (Target Train): WildPASS (同上)
# 代码: build_tgt_for_pl -> 复用逻辑，读取 data_root/WildPASS
# 目标域验证 (Target Val): DensePASS (同上)
# 代码: build_tgt_from_pd -> 复用逻辑，读取 data_root/DensePASS
# 适用场景: 只有虚拟数据训练的模型，如何迁移到真实世界的全景图上。注意：它的目标域和套餐一是一模一样的。
# 📦 套餐三：dataset='Stanford2D3D'
# 这是 “室内场景 (Indoor)” 的 “透视到全景 (Pin-to-Pan)” 适应。
# 源域 (Source): Stanford2D3D (Pin) (普通视角的切片图)
# 代码: build_src -> StanfordPin8DataSet
# 目标域训练 (Target Train): Stanford2D3D (Pan) (360度全景图，无标签)
# 代码: build_tgt_for_pl -> StanfordPan8forPL
# 目标域验证 (Target Val): Stanford2D3D (Pan) (360度全景图，有标签)
# 代码: build_tgt_from_pd -> StanfordPan8forVal
#dataset  是三者套餐之一的名字  根据套餐名字自动区分了文件夹
        if self.args.dataset == 'Stanford2D3D':
            temp_path_file = os.path.join(self.args.pd_root, 'Stanford2D3D', self.args.backbone, 'unused_path.txt')
            if os.path.exists(temp_path_file):
                os.remove(temp_path_file)
        else:
            temp_path_file = os.path.join(self.args.pd_root, 'DensePASS', self.args.backbone, self.args.dataset, 'unused_path.txt')
            if os.path.exists(temp_path_file):
                os.remove(temp_path_file)
        
        self.trainer = Trainer(args=self.args,
                               model_tgt=self.model_t,
                               model_src=self.model_s,
                               optimizer=optimizer,
                               criteria=criteria, #数据构建
                               adaptation_kwargs=dict(num_classes=self.num_classes,
                                                    feature_dim=256),
                               mem_strategy=mem_strategy,
                               callbacks=callbacks,
                               device=self.device)

    def need_pseudo_labels(self, epoch: int) -> bool:
        return (epoch % self.args.pseudo_every) == 0

    def generate_pseudo_labels(self, epoch: int):
        self.save_dir = self._pl_epoch_dir(epoch)  #产生伪标签的保存地方
        parent_save_dir = os.path.dirname(self.save_dir)
        clear_dir(self.save_dir)
        max_num = self.args.warmup_pl if epoch == 0 else self.args.iters_pl
        self.is_warmup = True if epoch == 0 else False

        plg = PseudoLabelGenerator(model=self.model_t,
                                   device=self.device,
                                   class_names=self.class_names,
                                   num_classes=self.num_classes)
        used_paths = plg.run(self.target_pl_loader, self.save_dir, max_num, self.args.dataset, self.args.uc_threshold)

        self.data_builder.write_splits_after_pl(parent_save_dir, used_paths)

    def train(self, epoch: int):
        if self.save_dir is None:
            self.save_dir = self._pl_epoch_dir(epoch) 
        self.target_loader, self.val_loader = self.data_builder.build_tgt_from_pd(self.save_dir)
        # 调用数据构建器 (data_builder)，传入刚才确认的伪标签路径 (save_dir)，重新创建目标域的训练加载器 (target_loader) 和验证加载器 (val_loader)
        # self.target_loader (训练用的目标域数据) 身份：它是 WildPASS 数据集 + 伪标签。
        # self.val_loader (验证用的目标域数据) 身份：它是 DensePASS 数据集 + 真实标签
        #训练基每一轮都在变  验证集一直没变
        if self.source_loader is None:
            self.source_loader, _ = self.data_builder.build_src()
        #如果源域加载器 (source_loader) 还没加载过（是 None），那就去加载一下；如果已经有了，就跳过
        self.global_step = self.trainer.train_one_epoch(epoch, self.global_step, self.source_loader, self.target_loader, self.is_warmup)

    def validate(self, epoch: int):
        mIoU = self.trainer.validate(epoch, self.val_loader, self.class_names)
        if mIoU > self.best_mIoU:
            self.best_mIoU = mIoU
            self.best_ckpt_path = os.path.join(self._save_dir(), f"best_{self.args.backbone}_iou{mIoU:.2f}.pth")
            torch.save(self.model_t.state_dict(), self.best_ckpt_path)
        print(f"Epoch {epoch+1}: val mIoU={mIoU:.2f}, best={self.best_mIoU:.2f}")

    def step_epoch_end(self, epoch: int):
        if (epoch + 1) % 5 == 0:
            self.trainer.reset_optimizer(lr=self.args.lr)

    def _save_dir(self):
        # 1. 如果你在 TrainConfig 里定义了 save_ckpt_dir 并且在命令行指定了它
        if hasattr(self.args, 'save_ckpt_dir') and self.args.save_ckpt_dir:
            d = self.args.save_ckpt_dir
        else:
            # 2. 只有没指定时，才用这种默认的拼接逻辑
            d = os.path.join(self.args.pd_root, 'checkpoints', self.args.dataset, self.args.backbone)
            #模型权重保存路径 (Checkpoints)
        ensure_dir(d)
        return d
    
# 验证时的路径（永远固定的）
# 验证（Validation）是为了测试模型的真实水平，我们必须用 真实的标签（Ground Truth），绝对不能用老师生成的“伪标签”来验证（那是自欺欺人）。
#     路径来源：来自于你的 args.data_root（原始数据目录）。
# 每一轮发生了什么：
# 无论第几轮，验证程序都会去读取 Cityscapes/val 或者 OmniSAM/val 里的原始人工标注图片。
# 结论：验证路径是 “永久”且“静态” 的，跟 _pl_epoch_dir 没有任何关系。
# 3. 模型保存路径（Checkpoints）
# 你可能还关心“我的 .pth 模型文件保存在哪里？”。这也跟 _pl_epoch_dir 不一样。
# 函数：通常是由 _save_dir（我们在上一个问题里讨论的那个）控制。
# 路径：通常是 checkpoints/Cityscapes/sam2_s/。
# 结论：模型权重都堆在这里，不会按 Epoch 分文件夹存（通常是覆盖保存 last.pth 或保存 best.pth）。


    def _pl_epoch_dir(self, epoch: int):
        if self.args.dataset == 'Stanford2D3D':
            return os.path.join(self.args.pd_root, 'Stanford2D3D', self.args.backbone, f'epoch{epoch + 1}')
        else:
            return os.path.join(self.args.pd_root, 'DensePASS', self.args.backbone, self.args.dataset,
                                f'epoch{epoch + 1}')
#   训练时的路径（动态变化的）
# 在 UDA 训练中，因为老师（Teacher）每一轮都在进步，它生成的伪标签每一轮都不一样。所以，训练数据的读取路径必须每一轮都变。
# 路径来源：正是你贴出来的这个 _pl_epoch_dir 函数。
# 每一轮发生了什么：
# Epoch 1: 老师把生成的伪标签存在 .../epoch1。学生去 .../epoch1 读取数据来训练。
# Epoch 2: 老师重新生成更好的标签，存在 .../epoch2。学生去 .../epoch2 读取数据来训练。
# ...# 结论：训练路径是 “一次性”且“动态” 的。      
#这里  老师和学生都是自己