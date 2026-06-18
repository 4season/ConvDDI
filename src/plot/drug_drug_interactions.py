import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_drug_csv(file_name, top_n=10):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "..", "data", file_name)

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        print(f"❌ 파일을 읽는 중 오류가 발생했어요: {e}")
        return

    target_cols = ['material_1', 'material_2']

    df = df.dropna(subset=target_cols)
    df[target_cols[0]] = df[target_cols[0]].astype(str).str.strip().str.lower()
    df[target_cols[1]] = df[target_cols[1]].astype(str).str.strip().str.lower()
    df[target_cols[0]] = df[target_cols[0]].apply(lambda x: x.split('('))
    df[target_cols[1]] = df[target_cols[1]].apply(lambda x: x.split('('))

    top_main_drugs = df[target_cols[0]].value_counts().head(top_n).index.tolist()

    filtered_df = df[df[target_cols[0]].isin(top_main_drugs)].copy()
    filtered_df = filtered_df.explode(target_cols[1])
    filtered_df[target_cols[1]] = filtered_df[target_cols[1]].astype(str).str.strip()
    connected_sub_drugs = sorted(filtered_df[target_cols[1]].unique())

    pos = {}

    left_y_coords = np.linspace(1, 0, len(top_main_drugs))
    for i, drug in enumerate(top_main_drugs):
        pos[f"L_{drug}"] = (0, left_y_coords[i])

    right_y_coords = np.linspace(1, 0, len(connected_sub_drugs))
    for i, drug in enumerate(connected_sub_drugs):
        pos[f"R_{drug}"] = (1, right_y_coords[i])

    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, max(8, len(connected_sub_drugs) * 0.35)))

    for _, row in filtered_df.iterrows():
        m1_key = f"L_{row['material_1']}"
        m2_key = f"R_{row['material_2']}"

        x_pts = [pos[m1_key][0], pos[m2_key][0]]
        y_pts = [pos[m1_key][1], pos[m2_key][1]]

        ax.plot(x_pts, y_pts, color='#748FFC', alpha=0.5, linewidth=1.2, zorder=1)

    for drug, (x, y) in pos.items():
        if drug.startswith("L_"):
            clean_name = drug[2:]
            ax.scatter(x, y, color='#4D77FF', s=120, zorder=2, edgecolor='navy')
            ax.text(x - 0.03, y, clean_name, fontsize=11, fontweight='bold',
                    ha='right', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

        elif drug.startswith("R_"):
            clean_name = drug[2:]
            ax.scatter(x, y, color='#20CC85', s=60, zorder=2, edgecolor='#127C50')
            ax.text(x + 0.03, y, clean_name, fontsize=10,
                    ha='left', va='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))

    ax.set_title(f"MLP 구조형 병용금기 약물 매핑 (상위 {top_n}개 기준 성분)", fontsize=16, pad=20, weight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['기준 성분 1 (Main Nodes)', '병용금기 성분 2 (Edge Nodes)'], fontsize=12, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#D1D1D1')
    ax.get_yaxis().set_visible(False)

    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_drug_csv('contraindicated_drugs.xlsx', 10)