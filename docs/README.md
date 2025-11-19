# CLAM模型运行记录

环境准备

```shell
#参考INSTALLATION.md 很详细
```


### ✅NULL

```python
#WSI 分段和修补
#相较于其他wsi文件，这个需要更高的像素，所以在代码中注释掉了面积检查部分
#用tcga来分割的话，肉眼看没啥问题，跑出来感觉也还行
(clam_env) cuiping@amax:~/CLAM$ python check_tcga_fnac_full.py
随机 500 张 patch 里空白约占 6.8% #还行啊还行

#用fnac来分割，速度较慢
python /home/cuiping/CLAM/create_patches_fp.py \
  --source /data/cuiping/NULL/00_raw_wsi \
  --save_dir /data/cuiping/NULL/patches_out \
  --patch_size 256 \
  --seg --patch --stitch \
  --preset fnac.csv
#结果是真的还行啊
(clam_env) cuiping@amax:~/CLAM$ python check_tcga_fnac_full.py
随机 500 张 patch 里空白约占 0.6%  #降低了很多

#特征提取
CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py \
    --data_h5_dir /data/cuiping/NULL/patches_out \
    --data_slide_dir /data/cuiping/NULL/00_raw_wsi \
    --csv_path /data/cuiping/NULL/patches_out/process_list_autogen.csv \
    --feat_dir /data/cuiping/NULL/features \
    --batch_size 128 \
    --slide_ext .svs

#Training Splits  训练拆分（缺失标签，后续无法运行）
python create_splits_seq.py \
  --task task_2_tumor_subtyping \
  --csv_path /data/cuiping/NULL/labels_digital.csv \
  --output_dir ~/CLAM/splits \
  --k 10 \
  --seed 42 \
  --val_frac 0.15 \
  --test_frac 0.15
         
```



### ✅RCC（成功）

```shell
cd /home/cuiping/CLAM
#WSI 分段和修补
python /home/cuiping/CLAM/create_patches_fp.py \   
    --source /data/cuiping/RCC/00_raw_wsi \        # 原始 .svs/.ndpi 文件夹
    --save_dir /data/cuiping/RCC/patches_out \     # 输出根目录
    --patch_size 256 \                             # 补丁尺寸（patch_size） & 步长（step）
    --seg --patch --stitch \                       # 三连：分割→切补丁→拼概览图
    --preset tcga.csv                              # 超参数表（一行 CSV）
1-组织前景分割（只用 --seg）  mask
目的：把玻璃空白、刀痕、笔迹去掉，得到“组织轮廓”并保存一张 mask 预览图。
① /project/output/masks/TCGA-XXXX.jpg
‑ 一张长宽≈千像素级别的 JPEG，黑底白框就是算法认为“有组织”的区域。
② /project/output/process_list_autogen.csv
‑ 记录这次用的阈值、层数、面积过滤值，下次可复用或微调。
2-切 patch（--seg --patch 一起开）
目的：只在“有组织”的轮廓内部切 256×256 小图，并保存坐标。
③ /project/output/patches/TCGA-XXXX.h5
‑ 里面数据集 /coords 保存所有 patch 左上角 (x,y) 在 0 级参考系下的坐标；
‑ /contours 保存对应轮廓，方便后续“拼回去”。
‑ 如果 use_padding=True，边缘不足 256 会镜像补边。
④ 控制台会打印切了多少张，例如 Wrote 18,764 patches。
3-（可选）把 patch 结果拼成热图（--stitch）
目的：肉眼确认切得匀不匀、有没有漏切。
⑤ /project/output/stitches/TCGA-XXXX.jpg
‑ 一张缩小 64 倍的鸟瞰图，每个小格就是一张 patch，绿色=保留，红色=被过滤，黑区=背景。
‑ 若发现某些区域没格子，可回头调小面积过滤或改轮廓函数再切一次。

# 特征提取
CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py \
    --data_h5_dir /data/cuiping/RCC/patches_out \      # 已裁剪patch的h5文件目录
    --data_slide_dir /data/cuiping/RCC/00_raw_wsi \    # 原始WSI目录
    --csv_path /data/cuiping/RCC/patches_out/process_list_autogen.csv \  # 处理列表
    --feat_dir features \                                # 输出特征目录
    --batch_size 128 \                                   # 推理batch size
    --slide_ext .svs                                     # 切片文件扩展名，，用于过滤非 svs 文件


(clam_env) cuiping@amax:/data/cuiping/RCC/features$ tree -L 2 
.
├── h5_files          # 每个 WSI 对应一个 .h5 文件，里面是 patch 特征（features） + 坐标（coords）
├── pt_files          # 每个 WSI 对应一个 .pt 文件，.pt 可能是 dict 或 Tensor 里面是 tensor 格式的特征（PyTorch 可用）


#Training Splits  训练拆分
  python create_splits_seq.py \
  --task task_2_tumor_subtyping \
  --csv_path /data/cuiping/RCC/labels_digital.csv \
  --output_dir ~/CLAM/splits \
  --k 10 \
  --seed 42 \
  --val_frac 0.15 \  #一定要加这个，md搞了两天，划分数据集就是不对
  --test_frac 0.15


#新版本scipy（1.11.0+）移除了对非数值数据的 stats.mode 支持,用 np.unique 来替代实现相同的功能
#将
label = stats.mode(label)[0]
# 替换旧的 stats.mode 调用
unique, counts = np.unique(label, return_counts=True)
label = unique[np.argmax(counts)]

整个划分文件里，所有样本都被标成了“验证集”
这意味着什么？
训练集 = 0 条 → train 列全是 False（甚至不存在）
测试集 = 0 条 → test 列也全是 False
于是 weighted_sample 在计算 训练集 类权重时，某一类可用样本 = 0 → 除零崩溃

# 训练三分类CLAM模型（放在服务器后台---脚本在CLAM/tools）
CUDA_VISIBLE_DEVICES=0 python ~/CLAM/main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --split_dir ~/CLAM/splits/task_2_tumor_subtyping_100 \
  --exp_code RCC_subtyping_clam_sb \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_2_tumor_subtyping \
  --model_type clam_sb \
  --log_data \
  --data_root_dir /data/cuiping/RCC/features \
  --embed_dim 1024 \
  --subtyping


# 看缺失
comm -13 \
  <(ls /data/cuiping/RCC/features/pt_files/ | sed 's/\.pt$//' | sort) \
  <(cut -d, -f2-4 splits_0.csv | tr ',' '\n' | sort)
  


# ==========================================================
# 实验全局参数
# ==========================================================
exp_arguments:
  n_classes: 3                    # 模型输出的类别数（与训练时保持一致）
  save_exp_code: HEATMAP_RCC_OUTPUT   # 本次推理结果的主文件夹名，所有产出都会放在它下面
  raw_save_dir: /data/cuiping/RCC/heatmap_raw
                                  # 原始热图数据（numpy、csv、patch坐标等）落盘路径
  production_save_dir: /data/cuiping/RCC/heatmap_production_results
                                  # 最终可展示的彩色热图/JPEG保存路径
  batch_size: 128                 # 前向推理时，一次性喂给encoder的patch数

# ==========================================================
# 数据/Slide 相关
# ==========================================================
data_arguments:
  data_dir: /data/cuiping/RCC/00_raw_wsi/ # 切片根目录；也可写成dict做分库映射
  #data_dir_key: source            # 当data_dir是dict时，CSV中对应哪一列作为key
  process_list: /data/cuiping/RCC/patches_out/process_list_autogen.csv
                                  # 必需CSV，至少含slide_id；可额外包含label、seg/patch参数
  preset: presets/tcga.csv  # 预设的segment/patch参数表（组织前景分割、白片过滤等）
  slide_ext: .svs                 # 切片文件扩展名
  label_dict:                     # 字符串标签→整数映射，用于可视化时显示类别名
    KICH: 0
    KIRC: 1
    KIRP: 2

    # ==========================================================
# 切patch参数
# ==========================================================
patching_arguments:
  patch_size: 256                 # 在原图0级下采样的patch像素大小
  overlap: 0.5                    # 相邻patch重叠比例（0.5=50%）
  patch_level: 0                  # 从哪一层下采样开始切patch；0=最高分辨率
  custom_downsample: 1            # 在patch_level基础上再手动下采样几倍（1=不额外下采）

# ==========================================================
# 编码器（特征提取器）
# ==========================================================
encoder_arguments:
  model_name: resnet50_trunc      # 可选resnet50_trunc / uni_v1 / conch_v1
  target_img_size: 224            # 把patch resize成多大再送入encoder

# ==========================================================
# 下游模型（CLAM）加载
# ==========================================================
model_arguments:
  ckpt_path: /home/cuiping/CLAM/results/RCC_subtyping_clam_sb_s1/s_3_checkpoint.pt
                                  # 训练得到的最佳checkpoint
  model_type: clam_sb             # clam_sb / clam_mb / mil / transmil 等
  initiate_fn: initiate_model     # 对应utils/eval_utils.py中的初始化函数名
  model_size: small               # CLAM内部fc大小（small/large）
  drop_out: 0.                    # 推理时一般设0
  embed_dim: 1024                 # encoder输出特征维度（与训练时一致）

# ==========================================================
# 热图可视化专属参数
# ==========================================================
heatmap_arguments:
  vis_level: 1                    # 在slide的哪一层下采样上绘制热图；-1≈32×下采
  alpha: 0.4                      # 热图与原图融合透明度（0=只看原图，1=只看热图）
  blank_canvas: false             # true=纯白背景绘制热图；false=原图H&E当背景
  save_orig: true                 # 是否额外保存一份原始H&E图（与热图同分辨率）
  save_ext: jpg                   # 热图/原图保存格式
  use_ref_scores: true            # 是否用“非重叠patch”分布做percentile归一化
  blur: false                     # 是否对热图再做高斯平滑
  use_center_shift: true          # 判断patch是否在前景轮廓内时，是否把角点往中心移
  use_roi: false                  # 是否只计算指定ROI（x1,x2,y1,y2）内的热图
  calc_heatmap: true              # 是否真正计算重叠细粒度热图（false则只输出粗热图）
  binarize: false                 # 是否把attention得分二值化
  binary_thresh: -1               # 二值化阈值；<0 时自动用Otsu
  custom_downsample: 1            # 最终保存前再下采样几倍（1=不额外降分辨率）
  cmap: jet                       # 热图配色盘（matplotlib colormap）

# ==========================================================
# 采样/保存高注意力patch示例
# ==========================================================
sample_arguments:
  samples:
    - name: "topk_high_attention" # 采样策略名称，随意起
      sample: true                # 是否启用该策略
      seed: 1                     # 随机种子（top-k本身无随机，但后续可能加打乱）
      k: 15                       # 取attention最高的k张patch存图
      mode: topk                  # 固定写topk；后续可扩展random/bottomk等

/data/cuiping/RCC/heatmap_production_results/
HEATMAP_RCC_OUTPUT/
├── sampled_patches/        ← 抽样出的高注意力 ROI patch
│   ├── label_Unspecified_pred_0/
│   │   └── topk_high_attention/   ← attention最高的patch在这里！
│   ├── label_Unspecified_pred_1/
│   └── label_Unspecified_pred_2/
│
└── Unspecified/            ← 每个 slide 对应的完整热图、原图、mask等（可视化）

/data/cuiping/RCC/heatmap_raw/
├── HEATMAP_RCC_OUTPUT/          ← 本次热图生成实验的主输出目录（--save_exp_code）
│   ├── config.yaml              ← 本次热图生成的配置文件
│   └── Unspecified/             ← 存放每张 slide 的热图文件夹
│       ├── TCGA-2Z-A9J2-01Z-00-DX1.AC19245F-B3B9-4A3A-89A3-A8B2E4BD988A/
│       ├── TCGA-BP-5006-01A-01-BS1.1d7bd0e2-6853-42db-b42e-ab4a413d0430/
│       ├── TCGA-KM-8440-01Z-00-DX1.528E053C-E8A4-464F-BE47-412836B1C31B/
│       ├── TCGA-KM-8442-01Z-00-DX1.46835CE2-819D-4887-9633-422BC9F5E366/
│       └── ... (更多 slide)

单个 slide 文件夹结构
/data/cuiping/RCC/heatmap_raw/HEATMAP_RCC_OUTPUT/Unspecified/TCGA-2Z-A9J2-01Z-00-DX1.AC19245F-B3B9-4A3A-89A3-A8B2E4BD988A/
├── TCGA-2Z-A9J2-01Z-00-DX1..._mask.jpg              ← 组织分割图（mask 可视化）
├── TCGA-2Z-A9J2-01Z-00-DX1..._mask.pkl              ← 分割掩膜数据文件（numpy）
├── TCGA-2Z-A9J2-01Z-00-DX1..._blockmap.png          ← 注意力热图（可视化）
├── TCGA-2Z-A9J2-01Z-00-DX1..._blockmap.h5           ← 热图原始数值（attention_scores + coords）
├── TCGA-2Z-A9J2-01Z-00-DX1..._0.5_roi_False.h5      ← 中间版本热图数据（不同参数）
├── TCGA-2Z-A9J2-01Z-00-DX1...pt                     ← slide 的特征向量（torch tensor）
└── TCGA-2Z-A9J2-01Z-00-DX1...h5                     ← 特征 + 坐标数据（HDF5）
```

```
H&E 染色
| 染料                  | 染色对象               | 显示颜色      | 生物学意义     |
| ------------------- | ------------------ | --------- | --------- |
| **Hematoxylin苏木精 (H)** | 细胞核、核仁（富含 DNA/RNA） | 深蓝色 / 紫蓝色 | 细胞活动、增殖   |
| **Eosin伊红 (E)**       | 细胞质、胶原、基质、血浆       | 粉红色 / 橙红色 | 细胞结构、间质环境 |

颜色	组织类型	说明
🔵 深蓝紫	细胞核密集区域（肿瘤细胞团、淋巴细胞浸润）	往往是肿瘤病灶或活跃区域
💗 淡粉红	细胞质或间质、结缔组织	正常组织或支撑结构
⚪ 白色	背景、空泡、血管腔	非组织区域
🟣 紫红混合	细胞团块、核密集但带细胞质的区域	常为肿瘤实质区
🩸 暗红 / 橙红	红细胞聚集、出血区	可见坏死或血管相关病灶

mask 暗红 + heatmap 红黄重叠 → 模型聚焦肿瘤组织；
mask 暗红 + heatmap 蓝 → 模型识别为正常结构；
mask 白 + heatmap 红 → 模型出错（看了背景伪影）。

癌细胞（Malignant cells）在 H&E 下的典型表现
1️⃣ 核异型性（nuclear atypia）
| 现象               | 含义            |
| ---------------- | ------------- |
| 核大、染色深（深蓝或深紫）    | DNA 含量增多、分裂活跃 |
| 核形不规则（圆→椭圆→扭曲）   | 染色质异常聚集       |
| 核仁明显、偏位          | RNA 活跃合成      |
| 核质比高（N/C ratio↑） | 核体积显著大于细胞质    |
👉 在 H&E 图像中表现为：
蓝紫色区域密集、颗粒粗、核轮廓不平滑。
2️⃣ 组织结构紊乱（architectural disorganization）
| 现象        | H&E 下表现          |
| --------- | ---------------- |
| 腺体或导管排列紊乱 | 正常结构消失，形成“乱堆”细胞团 |
| 极性消失      | 细胞方向混乱、排列不整齐     |
| 边界浸润性生长   | 细胞侵入周围间质、血管      |
👉 在热图上通常对应模型的 高注意力红区。
3️⃣ 细胞异质性（cellular pleomorphism）
| 现象            | H&E 表现    |
| ------------- | --------- |
| 同一区域细胞大小差异大   | 某些核大、某些核小 |
| 核形多样（圆、椭圆、肾形） | 说明克隆异质性   |
| 染色质分布不均       | 蓝紫深浅不一    |
👉 模型往往捕捉到这种纹理不均的区域作为诊断线索。
4️⃣ 增生与有丝分裂（mitosis）
| 现象                     | H&E 下表现      |
| ---------------------- | ------------ |
| 出现分裂象（Mitotic figures） | 核呈梅花状、Y 形或棒状 |
| 多核巨细胞                  | 同一细胞有多个蓝核    |
👉 深紫色小点状核多 → 分裂活跃 → 高度恶性区域。
5️⃣ 坏死与出血（necrosis / hemorrhage）
| 现象        | H&E 颜色表现          |
| --------- | ----------------- |
| 坏死        | 粉白或灰区，无清晰核，细胞界限模糊 |
| 出血        | 暗红色、背景中红细胞堆积      |
| 核碎裂 / 核溶解 | 蓝色残核碎片散在          |
👉 模型常在坏死边缘区高注意力，因为该区域代表肿瘤生长活跃边界。
```



### ✅NSCLC（跑到一半放弃）

```shell
#WSI 分段和修补
python /home/cuiping/CLAM/create_patches_fp.py \
    --source /data/cuiping/NSCLC/00_raw_wsi \
    --save_dir /data/cuiping/NSCLC/patches_out \
    --patch_size 256 \
    --seg --patch --stitch \
    --preset tcga.csv
    
#可能存在部分未seg，需要重新切割，后面跑特征提取的时候就知道了
patches_out/
    ├── patches/
    │   ├── slide_1.h5
    │   ├── slide_2.h5
    │   └── ...
    ├── process_list_autogen.csv
    ├── stitches/   (可选：拼接后的图像)
    └── masks/      (可选：分割的掩膜图像)

#特征提取（GPU 示例）
#手动从huggingface下载bin文件和config文件
CLAM/
    ├── resnet50_tv_in1k/
    │   ├── pytorch_model.bin
    │   ├── config.json

#特征提取   
CUDA_VISIBLE_DEVICES=2 python extract_features_fp.py \
    --data_h5_dir /data/cuiping/NSCLC/patches_out \      # 已裁剪patch的h5文件目录
    --data_slide_dir /data/cuiping/NSCLC/00_raw_wsi \    # 原始WSI目录
    --csv_path /data/cuiping/NSCLC/patches_out/process_list_autogen.csv \  # 处理列表
    --feat_dir features \                                # 输出特征目录
    --batch_size 128 \                                   # 推理batch size
    --slide_ext .svs                                     # 切片文件扩展名，，用于过滤非 svs 文件
python /home/cuiping/CLAM/create_patches_fp.py \
    --source /data/cuiping/NSCLC/00_raw_wsi \
    --save_dir /data/cuiping/NSCLC/patches_out \
    --patch_size 256 \
    --seg --patch --stitch \
    --preset tcga.csv
 features/
    └── NSCLC_dataset/
        ├── h5_files/*.h5
        └── pt_files/*.pt

#手动从hugging face下载模型后设置环境变量使得训练时使用下载模型
CONCH: https://huggingface.co/MahmoodLab/CONCH
海螺：https://huggingface.co/MahmoodLab/CONCH
checkpoints
├── conch
│   ├── meta.yaml
│   └── pytorch_model.bin
└── uni
    ├── config.json
    └── pytorch_model.bin
export CONCH_CKPT_PATH=checkpoints/conch/pytorch_model.bin
export UNI_CKPT_PATH=checkpoints/uni/pytorch_model.bin
 
#将文字标签转化为数字标签，即LUAD->0,LUSC->1
import pandas as pd
# 读取原始标签文件
df = pd.read_csv('labels.csv')
# NSCLC标签映射
label_mapping = {'LUAD': 0, 'LUSC': 1}
# 方法一：添加数字标签列（推荐）
df['label_digital'] = df['label'].map(label_mapping)
# 保存
df.to_csv('labels_digital.csv', index=False)
print("NSCLC标签映射：", label_mapping)
print("数据预览：")
print(df.head())

#Training Splits  训练拆分
cd ~/CLAM
python create_splits_seq.py \
  --task task_2_tumor_subtyping \
  --csv_path /data/cuiping/NSCLC/labels_digital.csv \
  --output_dir ~/CLAM/splits \
  --k 10 \
  --seed 42 \
  --val_frac 0.15 \
  --test_frac 0.15
  
CLAM任务类型的区别：
task_1_tumor_vs_normal
用途：肿瘤组织 vs 正常组织的二分类

示例：癌组织 vs 癌旁正常组织

特点：类别含义明确（肿瘤/正常），数据分布通常不平衡

task_2_tumor_subtyping
用途：肿瘤亚型之间的分类

示例：LUAD vs LUSC、KIRC vs KIRP vs KICH

特点：都是肿瘤组织，区分不同亚型
  
#训练 CLAM 模型

#新版本的NumPy（2.0+）中 np.Inf 被移除了，需要使用 np.inf
# 查找所有np.Inf的使用
grep -r "np.Inf" /home/cuiping/CLAM/
# 批量替换
find /home/cuiping/CLAM -name "*.py" -exec sed -i 's/np\.Inf/np.inf/g' {} \;

CUDA_VISIBLE_DEVICES=0 python ~/CLAM/main_for_NSCLC.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 10 \
  --split_dir ~/CLAM/splits/task_2_tumor_subtyping_100_for_NSCLC \
  --exp_code NSCLC_subtyping_clam_sb \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_2_tumor_subtyping \
  --model_type clam_sb \
  --log_data \
  --data_root_dir /data/cuiping/NSCLC/features \
  --embed_dim 1024 \
  --subtyping

#模型评估
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --k 10 \
    --models_exp_code NSCLC_CLAM_s1 \
    --save_exp_code NSCLC_CLAM_s1_cv \
    --task task_1_tumor_vs_normal \
    --model_type clam_sb \
    --results_dir results \
    --data_root_dir features \
    --csv_path labels.csv \
    --embed_dim 1024
    
#可视化
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config config_template.yaml
```

### ✅CAMEYON