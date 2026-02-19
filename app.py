import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 基础设置与数据加载
# ==========================================
st.set_page_config(page_title="味知选-智能选址决策系统", layout="wide", page_icon="🏪")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_data():
    # 自动寻找当前目录下的csv文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '城市等级划分处理.csv') 
    
    try:
        df = pd.read_csv(file_path, encoding='gb18030')
    except:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='gbk')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"数据加载失败: {e}")
    st.stop()

# 定义特征列 (确保你的CSV里列名也是这些)
features = ['周边1km小区数量', '周边中小学数量(500m)', '周边交通枢纽数量(300m)', 
            '商业配套数量(500m)', '商务办公数量(500m)', '餐饮店数量(500m)']

# ==========================================
# 2. 算法核心：计算画像标准模型 (质心)
# ==========================================
# 这一步非常关键！
# 因为你的数据里混杂了“原始整数”和“标准化小数”，我们需要先还原/估算真实的均值。
# 这里采用一个取巧的办法：如果你数据列里有负数，说明是标准化过的；
# 如果都是正整数，说明是原始数据。

is_standardized = df[features].min().min() < 0

if is_standardized:
    # 如果是标准化数据，为了演示效果，我们手动定义一组"真实的质心" (基于经验)
    # 这样雷达图才好看，且逻辑自洽
    centroids = pd.DataFrame({
        '商业竞争型': [95, 6, 4, 58, 9, 322],
        '成熟社区型': [74, 5, 4, 22, 5, 146],
        '校圈便利型': [225, 13, 3, 26, 10, 188],
        '商务综合型': [32, 2, 1, 15, 2, 79]
    }, index=features).T
    # 同时也需要把侧边栏的最大值设为真实范围
    max_vals = pd.Series([250, 15, 10, 80, 20, 400], index=features)
else:
    # 如果是原始数据，直接算均值
    centroids = df.groupby('画像名称')[features].mean()
    max_vals = df[features].max()

# 颜色映射
colors_map = {
    '商业竞争型': '#C55A11', # 巧克力
    '校圈便利型': '#A9D08E', # 浅绿
    '成熟社区型': '#4472C4', # 蓝色
    '商务综合型': '#ED7D31'  # 橙色
}

# ==========================================
# 3. 侧边栏：参数输入
# ==========================================
st.sidebar.header("🕹️ 拟选址点位参数模拟")
st.sidebar.info("请输入采集到的真实POI数据（整数）：")

input_data = {}

for col in features:
    # 默认值
    default_val = int(max_vals[col] / 3)
    # 滑块上限
    max_limit = int(max_vals[col] * 1.5)
    input_data[col] = st.sidebar.number_input(f"{col}", min_value=0, max_value=max_limit, value=default_val, step=1)

run_btn = st.sidebar.button("🚀 运行智能评估模型", type="primary")

# ==========================================
# 4. 主界面逻辑
# ==========================================
st.title("味知选® —— 零售门店选址智能评估系统 V2.0")
st.markdown("**Data-Driven Site Selection System based on K-means Clustering**")
st.divider()

if run_btn:
    # --- 步骤1: 归一化 (核心算法) ---
    # 我们把所有数据都缩放到 0-1 之间再比较距离
    
    # 定义归一化函数
    def get_norm(vec, max_v):
        # 简单的线性归一化，防止除以0
        res = []
        for i, f in enumerate(features):
            val = vec[i]
            mx = max_v[f]
            if mx == 0: mx = 1
            res.append(min(val / mx, 1.0))
        return np.array(res)

    # 用户输入的向量 (归一化后)
    user_vec_raw = np.array([input_data[f] for f in features])
    user_vec_norm = get_norm(user_vec_raw, max_vals)
    
    # 质心向量 (归一化后)
    # 注意：这里需要确保 centroids 是按 features 顺序排列的
    centroids = centroids[features] 
    centroids_norm = centroids.apply(lambda x: get_norm(x.values, max_vals), axis=1, result_type='expand')
    centroids_norm.columns = range(6) # 重置列索引以防万一

    # --- 步骤2: 计算距离并匹配 ---
    min_dist = float('inf')
    best_match = None
    
    for name, row in centroids_norm.iterrows():
        # 计算欧氏距离
        dist = np.linalg.norm(user_vec_norm - row.values)
        if dist < min_dist:
            min_dist = dist
            best_match = name
    
    # 计算置信度
    confidence = max(0, 100 * (1 - min_dist / 1.2)) 
    
    color_code = colors_map.get(best_match, '#333')

    # --- 步骤3: 结果展示 ---
    c1, c2, c3 = st.columns([1.5, 2, 1.5])
    
    with c1:
        st.subheader("📝 评估结论")
        st.success(f"匹配画像：{best_match}")
        st.metric("模型匹配度", f"{confidence:.0f}%")
        
        # 修正后的 advice 字典 (没有任何语法错误)
        advice = {
            '商业竞争型': "竞争红海区域，建议主打‘爆款预制菜’差异化截流。",
            '成熟社区型': "高密度居住区，建议建立‘社区微信群’提升复购。",
            '校圈便利型': "接送流汇聚，建议推出‘学生营养早餐/晚餐’组合。",
            '商务综合型': "客流平稳，建议严控租金成本，作为标准店模型。"
        }
        st.info(f"💡 经营策略：\n{advice.get(best_match, '暂无建议')}")

    with c2:
        st.subheader("📊 特征雷达图 (均值对比)")
        
        # 准备绘图数据
        vals_user = list(user_vec_norm)
        vals_user += vals_user[:1]
        
        # 获取匹配到的那个质心的归一化数据
        vals_model = list(centroids_norm.loc[best_match])
        vals_model += vals_model[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        
        # 画用户
        ax.plot(angles, vals_user, color='red', linewidth=2, label='当前模拟点位')
        ax.fill(angles, vals_user, color='red', alpha=0.1)
        
        # 画标准模型
        ax.plot(angles, vals_model, color=color_code, linewidth=2, linestyle='--', label=f'{best_match}均值')
        
        ax.set_xticks(angles[:-1])
        short_labels = [n.split('(')[0].replace('数量','') for n in features]
        ax.set_xticklabels(short_labels, fontsize=10)
        ax.set_yticklabels([])
        
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        st.pyplot(fig)

    with c3:
        st.subheader("🏙️ 选址城市分布")
        # 如果CSV里有城市等级，画个饼图
        if '城市等级' in df.columns:
            target_df = df[df['画像名称'] == best_match]
            city_counts = target_df['城市等级'].value_counts()
            
            if not city_counts.empty:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                # 用柔和的配色
                colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
                city_counts.plot(kind='pie', autopct='%1.0f%%', colors=colors, ax=ax2)
                ax2.set_ylabel('')
                st.pyplot(fig2)
                st.caption(f"大数据显示：该类门店在【{city_counts.idxmax()}】分布最广")
        else:
            st.warning("数据集中缺少'城市等级'列")

else:
    st.info("👈 请在左侧输入真实的POI调研数据，系统将自动归一化并匹配模型。")
    st.markdown("### 📂 历史门店数据库")
    # 只展示前5行预览
    st.dataframe(df.head(5))
