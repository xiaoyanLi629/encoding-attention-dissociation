"""
Schaefer 1000分区脑区映射指南
Brain Region Mapping Guide for Schaefer 2018 Atlas

本项目使用Schaefer 2018图谱，包含1000个脑区分区，分为7个功能网络。
分区编号0-499为左半球，500-999为右半球。

主要参考文献:
Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex 
from intrinsic functional connectivity MRI. Cerebral cortex, 28(9), 3095-3114.
"""

import numpy as np

# ============================================================================
# Schaefer 1000分区的7个功能网络定义
# ============================================================================

SCHAEFER_7_NETWORKS = {
    'Visual': {
        'left_hemisphere': list(range(0, 60)),      # 0-59
        'right_hemisphere': list(range(500, 560)),   # 500-559
        'total_regions': 120,
        'description': '初级和次级视觉皮层 (V1, V2, V3, V4, MT)',
        'anatomical_regions': ['枕叶', '纹状皮层', '纹外皮层']
    },
    'Somatomotor': {
        'left_hemisphere': list(range(60, 130)),     # 60-129
        'right_hemisphere': list(range(560, 630)),   # 560-629
        'total_regions': 140,
        'description': '躯体运动和前运动皮层',
        'anatomical_regions': ['中央前回', '中央后回', '辅助运动区']
    },
    'DorsalAttention': {
        'left_hemisphere': list(range(130, 175)),    # 130-174
        'right_hemisphere': list(range(630, 685)),   # 630-684
        'total_regions': 100,
        'description': '背侧注意力网络',
        'anatomical_regions': ['顶内沟', '额叶眼动区', '顶上小叶']
    },
    'VentralAttention': {
        'left_hemisphere': list(range(175, 220)),    # 175-219
        'right_hemisphere': list(range(685, 740)),   # 685-739
        'total_regions': 100,
        'description': '腹侧注意力/突显网络',
        'anatomical_regions': ['颞顶联合区', '前岛叶', '前扣带回']
    },
    'Limbic': {
        'left_hemisphere': list(range(220, 250)),    # 220-249
        'right_hemisphere': list(range(740, 780)),   # 740-779
        'total_regions': 70,
        'description': '边缘系统',
        'anatomical_regions': ['眶额皮层', '颞极', '海马旁回']
    },
    'Frontoparietal': {
        'left_hemisphere': list(range(250, 330)),    # 250-329
        'right_hemisphere': list(range(780, 870)),   # 780-869
        'total_regions': 170,
        'description': '额顶控制网络',
        'anatomical_regions': ['背外侧前额叶', '顶下小叶', '前扣带回背侧']
    },
    'Default': {
        'left_hemisphere': list(range(330, 500)),    # 330-499
        'right_hemisphere': list(range(870, 1000)),  # 870-999
        'total_regions': 300,
        'description': '默认模式网络',
        'anatomical_regions': ['内侧前额叶', '后扣带回', '角回', '颞中回']
    }
}

# ============================================================================
# 特定功能区域的精细映射
# 基于Schaefer分区与解剖区域的对应关系
# ============================================================================

FUNCTIONAL_REGIONS = {
    # ========== 视觉处理 ==========
    'PrimaryVisualCortex': {
        'description': '初级视觉皮层 (V1/BA17)',
        'network': 'Visual',
        'indices': list(range(0, 20)) + list(range(500, 520)),  # 估计范围
        'expected_modality': 'visual',
        'hemisphere': 'bilateral'
    },
    'SecondaryVisualCortex': {
        'description': '次级视觉皮层 (V2-V4)',
        'network': 'Visual',
        'indices': list(range(20, 50)) + list(range(520, 550)),
        'expected_modality': 'visual',
        'hemisphere': 'bilateral'
    },
    'MotionArea_MT': {
        'description': '运动敏感区 (MT/V5)',
        'network': 'Visual',
        'indices': list(range(50, 60)) + list(range(550, 560)),
        'expected_modality': 'visual',
        'hemisphere': 'bilateral'
    },
    
    # ========== 听觉处理 ==========
    'PrimaryAuditoryCortex': {
        'description': '初级听觉皮层 (A1/颞横回)',
        'network': 'Somatomotor',  # 在Schaefer中，听觉区域部分归入躯体运动网络
        'indices': list(range(100, 115)) + list(range(600, 615)),  # 估计范围
        'expected_modality': 'audio',
        'hemisphere': 'bilateral',
        'note': '初级听觉皮层在Schaefer图谱中位于Somatomotor网络边缘'
    },
    'SuperiorTemporalGyrus': {
        'description': '颞上回 (STG) - 听觉联合皮层',
        'network': 'VentralAttention',  # STG后部
        'indices': list(range(190, 210)) + list(range(700, 720)),
        'expected_modality': 'audio',
        'hemisphere': 'bilateral',
        'note': '颞上回后部参与语音和语言处理'
    },
    'SuperiorTemporalSulcus': {
        'description': '颞上沟 (STS) - 多感官整合区',
        'network': 'VentralAttention',
        'indices': list(range(200, 220)) + list(range(710, 740)),
        'expected_modality': 'multimodal',
        'hemisphere': 'bilateral',
        'note': '关键的多感官整合区域'
    },
    
    # ========== 语言处理 ==========
    'BrocaArea': {
        'description': "Broca区 (左侧额下回)",
        'network': 'Frontoparietal',
        'indices': list(range(280, 310)),  # 仅左半球
        'expected_modality': 'language',
        'hemisphere': 'left',
        'note': '语言产生核心区域'
    },
    'WernickeArea': {
        'description': "Wernicke区 (左侧颞上回后部)",
        'network': 'Default',  # 部分在Default网络中
        'indices': list(range(420, 450)),  # 仅左半球
        'expected_modality': 'language',
        'hemisphere': 'left',
        'note': '语言理解核心区域'
    },
    'AngularGyrus': {
        'description': '角回 (语义整合)',
        'network': 'Default',
        'indices': list(range(380, 410)) + list(range(900, 930)),
        'expected_modality': 'language',
        'hemisphere': 'bilateral',
        'note': '语义加工和阅读理解'
    },
    'LeftTemporalCortex': {
        'description': '左侧颞叶语言区',
        'network': 'Default',
        'indices': list(range(350, 420)),  # 仅左半球
        'expected_modality': 'language',
        'hemisphere': 'left',
        'note': '包括颞中回和颞下回的语言区域'
    },
    
    # ========== 多感官整合 ==========
    'TemporoparietalJunction': {
        'description': '颞顶联合区 (TPJ)',
        'network': 'VentralAttention',
        'indices': list(range(175, 195)) + list(range(685, 705)),
        'expected_modality': 'multimodal',
        'hemisphere': 'bilateral',
        'note': '注意力重新定向和社会认知'
    },
    'PosteriorSuperiorTemporalSulcus': {
        'description': '颞上沟后部 (pSTS)',
        'network': 'VentralAttention',
        'indices': list(range(205, 220)) + list(range(715, 740)),
        'expected_modality': 'multimodal',
        'hemisphere': 'bilateral',
        'note': '生物运动感知和社会信号整合'
    },
    'IntraperetalSulcus': {
        'description': '顶内沟 (IPS)',
        'network': 'DorsalAttention',
        'indices': list(range(140, 165)) + list(range(645, 675)),
        'expected_modality': 'multimodal',
        'hemisphere': 'bilateral',
        'note': '空间注意力和数量处理'
    },
    
    # ========== 默认模式网络 ==========
    'MedialPrefrontalCortex': {
        'description': '内侧前额叶皮层 (mPFC)',
        'network': 'Default',
        'indices': list(range(330, 360)) + list(range(870, 895)),
        'expected_modality': 'language',  # 社会认知和自我参照
        'hemisphere': 'bilateral',
        'note': '自我参照加工和社会认知'
    },
    'PosteriorCingulateCortex': {
        'description': '后扣带回皮层 (PCC)',
        'network': 'Default',
        'indices': list(range(360, 380)) + list(range(895, 915)),
        'expected_modality': 'multimodal',
        'hemisphere': 'bilateral',
        'note': '默认模式网络核心节点'
    },
    'Precuneus': {
        'description': '楔前叶',
        'network': 'Default',
        'indices': list(range(460, 500)) + list(range(960, 1000)),
        'expected_modality': 'multimodal',
        'hemisphere': 'bilateral',
        'note': '情景记忆和自我意识'
    }
}

# ============================================================================
# 用于研究分析的区域分组
# ============================================================================

RESEARCH_REGION_GROUPS = {
    'visual_cortex': {
        'description': '视觉皮层（预期对Visual特征高敏感）',
        'regions': ['PrimaryVisualCortex', 'SecondaryVisualCortex', 'MotionArea_MT'],
        'network_indices': SCHAEFER_7_NETWORKS['Visual']['left_hemisphere'] + 
                          SCHAEFER_7_NETWORKS['Visual']['right_hemisphere']
    },
    'auditory_cortex': {
        'description': '听觉皮层（预期对Audio特征高敏感）',
        'regions': ['PrimaryAuditoryCortex', 'SuperiorTemporalGyrus'],
        'approximate_indices': list(range(100, 130)) + list(range(190, 220)) +
                              list(range(600, 630)) + list(range(700, 740))
    },
    'language_network': {
        'description': '语言网络（预期对Language特征高敏感，左半球优势）',
        'regions': ['BrocaArea', 'WernickeArea', 'AngularGyrus', 'LeftTemporalCortex'],
        'left_dominant_indices': list(range(280, 310)) + list(range(350, 450))
    },
    'multisensory_integration': {
        'description': '多感官整合区（预期对多模态均敏感）',
        'regions': ['SuperiorTemporalSulcus', 'TemporoparietalJunction', 
                   'PosteriorSuperiorTemporalSulcus', 'IntraperetalSulcus'],
        'approximate_indices': list(range(175, 220)) + list(range(140, 175)) +
                              list(range(685, 740)) + list(range(630, 685))
    },
    'default_mode_network': {
        'description': '默认模式网络（预期高度多模态整合）',
        'regions': ['MedialPrefrontalCortex', 'PosteriorCingulateCortex', 
                   'Precuneus', 'AngularGyrus'],
        'network_indices': SCHAEFER_7_NETWORKS['Default']['left_hemisphere'] + 
                          SCHAEFER_7_NETWORKS['Default']['right_hemisphere']
    }
}


def get_region_indices(region_name):
    """获取特定区域的分区索引"""
    if region_name in FUNCTIONAL_REGIONS:
        return FUNCTIONAL_REGIONS[region_name]['indices']
    elif region_name in SCHAEFER_7_NETWORKS:
        return (SCHAEFER_7_NETWORKS[region_name]['left_hemisphere'] + 
                SCHAEFER_7_NETWORKS[region_name]['right_hemisphere'])
    else:
        print(f"未找到区域: {region_name}")
        return []


def get_network_for_parcel(parcel_index):
    """根据分区索引返回所属网络"""
    for network_name, network_info in SCHAEFER_7_NETWORKS.items():
        all_indices = network_info['left_hemisphere'] + network_info['right_hemisphere']
        if parcel_index in all_indices:
            hemisphere = 'left' if parcel_index < 500 else 'right'
            return network_name, hemisphere
    return None, None


def print_brain_region_summary():
    """打印脑区映射摘要"""
    print("=" * 70)
    print("Schaefer 1000分区脑区映射摘要")
    print("=" * 70)
    
    print("\n【7大功能网络】")
    print("-" * 50)
    for network_name, network_info in SCHAEFER_7_NETWORKS.items():
        left_range = f"{network_info['left_hemisphere'][0]}-{network_info['left_hemisphere'][-1]}"
        right_range = f"{network_info['right_hemisphere'][0]}-{network_info['right_hemisphere'][-1]}"
        print(f"\n{network_name}:")
        print(f"  左半球: 分区 {left_range}")
        print(f"  右半球: 分区 {right_range}")
        print(f"  总数: {network_info['total_regions']} 个分区")
        print(f"  描述: {network_info['description']}")
        print(f"  解剖区域: {', '.join(network_info['anatomical_regions'])}")
    
    print("\n\n【研究重点区域】")
    print("-" * 50)
    for group_name, group_info in RESEARCH_REGION_GROUPS.items():
        print(f"\n{group_name}:")
        print(f"  描述: {group_info['description']}")
        print(f"  包含区域: {', '.join(group_info['regions'])}")
    
    print("\n" + "=" * 70)


def create_region_mask(region_name, num_parcels=1000):
    """创建特定区域的二值掩码"""
    mask = np.zeros(num_parcels, dtype=bool)
    indices = get_region_indices(region_name)
    for idx in indices:
        if 0 <= idx < num_parcels:
            mask[idx] = True
    return mask


# ============================================================================
# 预期的模态敏感性
# ============================================================================

EXPECTED_MODALITY_SENSITIVITY = """
【预期的脑区-模态敏感性对应关系】

1. 视觉皮层 (Visual Network, 分区0-59, 500-559)
   → 预期: Visual特征敏感性最高
   → 原因: 初级和次级视觉皮层直接处理视觉输入

2. 听觉皮层 / 颞上回 (STG, 部分Somatomotor和VentralAttention)
   → 预期: Audio特征敏感性最高
   → 原因: 初级听觉皮层和听觉联合区处理声音信息
   → 注意: Schaefer图谱中听觉区域分布在多个网络

3. 语言网络 (左半球为主)
   → 预期: Language特征敏感性最高
   → 关键区域: Broca区(左额下回), Wernicke区(左颞上回后部), 角回
   → 注意: 主要位于Default和Frontoparietal网络中

4. 多感官整合区
   → 预期: 对多模态均有中等敏感性，多模态增益显著
   → 关键区域: 颞上沟(STS), 颞顶联合区(TPJ), 顶内沟(IPS)
   → 位置: 主要在VentralAttention和DorsalAttention网络

5. 默认模式网络 (Default Network, 分区330-499, 870-999)
   → 预期: 高度多模态整合，Language特征可能略占优势
   → 原因: 涉及叙事理解、社会认知等高级功能
   → 关键节点: mPFC, PCC, 角回, 楔前叶
"""


if __name__ == "__main__":
    print_brain_region_summary()
    print(EXPECTED_MODALITY_SENSITIVITY)

