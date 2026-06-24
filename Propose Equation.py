import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===========================
# 0. Font and style
# ===========================
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 18
rcParams['axes.grid'] = True
rcParams['grid.linestyle'] = '--'
rcParams['grid.alpha'] = 0.5

# ===========================
# 1. Load raw Excel
# ===========================
file_path = 'Cleaned.xlsx'
sheet_name = 'CleanedDataSet'

df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
df.columns = [
    'crr', 'blr', 'e', 'P', 'bd', 'bw', 'ch', 'cw',
    'fy', 'fc', 'Vexp', 'Vgb', 'Vrf', 'Vst'
]

df = df.apply(pd.to_numeric, errors='coerce').dropna()
print(f"Loaded {len(df)} specimens")

# ===========================
# 2. Convert experimental units MN -> kN
# ===========================
df['Vexp'] *= 1000.0
df['Vgb']  *= 1000.0
df['Vrf']  *= 1000.0
df['Vst']  *= 1000.0

# ===========================
# 3. Derived parameters
# ===========================
df['Aj'] = df['cw'] * df['ch']              # mm²
df['sqrt_fc'] = np.sqrt(df['fc'])           # MPa^0.5
df['rho_bl'] = df['blr'] / 100.0
df['norm_P'] = df['P'] / (df['fc'] * df['Aj'])
df['base'] = df['sqrt_fc'] * df['Aj']

# ===========================
# 4. Calibrate proposed model
# ===========================
X = np.column_stack([df['base'], df['base']*df['rho_bl'], df['base']*df['norm_P']])
y = df['Vexp'] * 1000.0  # N

model = LinearRegression(fit_intercept=False)
model.fit(X, y)
C, alpha_term, beta_term = model.coef_
alpha = alpha_term / C
beta  = beta_term / C
print(f"Calibrated Coefficients: C={C:.4f}, alpha={alpha:.3f}, beta={beta:.3f}")

df['V_proposed'] = (C * df['sqrt_fc'] * df['Aj'] * (1 + alpha*df['rho_bl'] + beta*df['norm_P'])) / 1000.0

# ===========================
# 5. Balanced ACI
# ===========================
V_ACI_nom = df['sqrt_fc'] * df['Aj'] / 1000.0
gamma_opt = (df['Vexp'] / V_ACI_nom).mean()
print(f"Balanced ACI factor gamma_opt = {gamma_opt:.3f}")
df['V_ACI_balanced'] = gamma_opt * V_ACI_nom

# ===========================
# 6. Performance metrics function
# ===========================
def performance_metrics(y_true, y_pred):
    ratio = y_pred / y_true
    mean_ratio = ratio.mean()
    cov = ratio.std() / mean_ratio * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mean_ratio, cov, rmse, r2

# ===========================
# 7. Create Results folder
# ===========================
figures_dir = "Results/Subplots"
os.makedirs(figures_dir, exist_ok=True)

# ===========================
# 8. Comparisons
# ===========================
comparisons = [
    ('Vexp','Vgb',r'Gradient Boosting vs $V_{exp}$'),
    ('Vexp','Vrf',r'Random Forest vs $V_{exp}$'),
    ('Vexp','Vst',r'Stacked Ensemble vs $V_{exp}$'),
    ('Vexp','V_proposed',r'Proposed Model vs $V_{exp}$'),
    ('V_ACI_balanced','V_proposed',r'ACI Balanced vs $V_{proposed}$'),
    ('V_ACI_balanced','Vexp',r'ACI Balanced vs $V_{exp}$')
]

error_bands = [0.2,0.3]

# ===========================
# 9. Create 2x3 subplots
# ===========================
fig, axes = plt.subplots(3,2,figsize=(16,18))
axes = axes.flatten()
summary_records = []

for i, (y_true_col, y_pred_col, title) in enumerate(comparisons):
    ax = axes[i]
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]

    mean_ratio, cov, rmse, r2 = performance_metrics(y_true, y_pred)
    summary_records.append([title, mean_ratio, cov, rmse, r2])

    # Scatter
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', label='Predicted vs Experimental', s=40)

    # Ideal line
    max_val = max(y_true.max(), y_pred.max())*1.05
    ax.plot([0,max_val],[0,max_val],'k--',linewidth=1.5,label='Ideal line (Y=X)')

    # Error bands ±20%, ±30%
    for j, band in enumerate(error_bands):
        ax.plot([0,max_val],[0,max_val*(1+band)],'r--',linewidth=1,label=f'+{int(band*100)}%' if j==0 else "")
        ax.plot([0,max_val],[0,max_val*(1-band)],'r--',linewidth=1,label=f'-{int(band*100)}%' if j==0 else "")

    ax.set_title(title, fontsize=22)
    ax.set_xlabel(r'$V_{exp}$ (kN)' if 'Vexp' in y_true_col else r'$V_{pred}$ (kN)', fontsize=18)
    ax.set_ylabel(f'{y_pred_col} (kN)', fontsize=18)
    ax.legend(fontsize=18)

    # Save metrics text
    text_path = os.path.join(figures_dir,f"{title.replace(' ','_')}_metrics.txt")
    with open(text_path,"w") as f:
        f.write(f"{title}\n{'='*40}\n")
        f.write(f"Mean Ratio (Pred/Exp): {mean_ratio:.3f}\n")
        f.write(f"COV (%):               {cov:.2f}\n")
        f.write(f"RMSE (kN):             {rmse:.2f}\n")
        f.write(f"R²:                    {r2:.3f}\n")
    print(f"Saved metrics: {text_path}")

# Hide unused axes if any
for j in range(len(comparisons), len(axes)):
    axes[j].axis('off')

plt.tight_layout(h_pad=3.0, w_pad=3.0)  # add space between plots

# ===========================
# 10. Save figure PNG and PDF
# ===========================
subplot_png_path = os.path.join(figures_dir,"All_Model_2x3_Comparisons.png")
subplot_pdf_path = os.path.join(figures_dir,"All_Model_2x3_Comparisons.pdf")
plt.savefig(subplot_png_path,dpi=300)
plt.savefig(subplot_pdf_path)
plt.show()
plt.close()
print(f"Saved subplot figure PNG: {subplot_png_path}")
print(f"Saved subplot figure PDF: {subplot_pdf_path}")

# ===========================
# 11. Save full Excel with predictions
# ===========================
output_file = 'Cleaned_with_Final_Unit_Corrected_Model.xlsx'
df.to_excel(output_file,index=False)
print(f"Saved full results to: {output_file}")
