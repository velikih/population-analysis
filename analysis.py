import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists('figures'):
    os.makedirs('figures')

file_name = 'countries_population (2).csv' 
df = pd.read_csv(file_name)

agg_isos = {'ARB','CSS','EAS','TEA','EAP','ECS','TEC','ECA','EMU','EUU','FCS',
            'HIC','HPC','IBD','IBT','IDB','IDX','IDA','LCN','LAC','TLA','LTE',
            'LDC','LMY','LIC','LMC','MIC','MEA','TMN','MNA','NAC','OED','OSS',
            'PRE','PST','SAS','TSA','SSF','TSS','SSA','SST','PSS','WLD'}

countries = df[~df['ISO3'].isin(agg_isos)].copy()


print("\n--- 1. Общие задания ---")
# 1.1 Посчитать количество NAN в каждом столбце, вывести столбец с самым большим количеством NAN
nan_counts = df.isna().sum()
print("Количество NaN в столбцах:\n", nan_counts)
max_nan_col = nan_counts.idxmax()
print(f"Столбец с наибольшим количеством NaN: {max_nan_col} ({nan_counts.max()} шт.)")

# 1.2 Заполнить все NAN средним значением по стране
df['Population Growth'] = df.groupby('Country')['Population Growth'].transform(lambda s: s.fillna(s.mean()))
df['Growth Rate (%)'] = df.groupby('Country')['Growth Rate (%)'].transform(lambda s: s.fillna(s.mean()))
countries['Population Growth'] = df[df['ISO3'].isin(countries['ISO3'].unique())].groupby('Country')['Population Growth'].transform(lambda s: s.fillna(s.mean()))
countries['Growth Rate (%)'] = df[df['ISO3'].isin(countries['ISO3'].unique())].groupby('Country')['Growth Rate (%)'].transform(lambda s: s.fillna(s.mean()))
print("Пропуски после заполнения:\n", df.isna().sum())

# 1.3 Найти страну с наибольшим % приростом (абсолютное значение за один год)
max_growth_idx = countries['Growth Rate (%)'].idxmax()
max_growth_row = countries.loc[max_growth_idx]
print(f"Страна с макс. % приростом: {max_growth_row['Country']} ({max_growth_row['Year']} г., {max_growth_row['Growth Rate (%)']:.2f}%)")

sub = countries[countries['Country'] == max_growth_row['Country']].sort_values('Year')
plt.figure(figsize=(10,4))
plt.plot(sub['Year'], sub['Growth Rate (%)'], 'g--', label=f"{max_growth_row['Country']} growth rate")
plt.xlabel('Год'); plt.ylabel('Growth Rate (%)')
plt.title(f"Темп прироста населения: {max_growth_row['Country']}")
plt.legend(); plt.grid(True)
plt.savefig('figures/1.3_growth_rate.png'); plt.close()

# 1.4 Среди стран на букву "А" найти с наибольшим и наименьшим населением за 2000 год
a_countries = countries[countries['Country'].str.startswith('A')]
pop2000 = a_countries[a_countries['Year'] == 2000][['Country','Population']]
max_c = pop2000.loc[pop2000['Population'].idxmax(), 'Country']
min_c = pop2000.loc[pop2000['Population'].idxmin(), 'Country']
print(f"Страны на 'А' в 2000: наибольшая - {max_c}, наименьшая - {min_c}")

fig, axes = plt.subplots(1, 2, figsize=(14,4))
for ax, c, color in zip(axes, [max_c, min_c], ['tab:blue','tab:orange']):
    s = countries[(countries['Country']==c) & (countries['Year'].between(1995,2005))].sort_values('Year')
    ax.plot(s['Year'], s['Population'], color=color, marker='o', label=c)
    ax.set_xlabel('Год'); ax.set_ylabel('Население')
    ax.set_title(f'Динамика населения: {c} (1995–2005)')
    ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig('figures/1.4_A_countries.png'); plt.close()

# 1.5 Доля населения пяти самых населенных стран в 2007 году
year_07 = 2007
pop_year = countries[countries['Year']==year_07]
top5 = pop_year.nlargest(5, 'Population')
world_pop = df[(df['ISO3']=='WLD') & (df['Year']==year_07)]['Population'].values[0]
shares = top5.set_index('Country')['Population'] / world_pop * 100

plt.figure(figsize=(8,8))
plt.pie(shares, labels=[f'{c}\n{s:.2f}%' for c,s in shares.items()], autopct='%1.2f%%', startangle=90)
plt.title(f'Доля 5 крупнейших стран в населении мира, {year_07}')
plt.savefig('figures/1.5_top5_pie.png'); plt.close()

# 1.6 Найти год, в который у наибольшего количества стран была максимальная % убыль
decline = countries[countries['Growth Rate (%)'] < 0]
year_counts = decline.groupby('Year').size()
worst_year = year_counts.idxmax()
print(f"Год с макс. числом стран с убылью населения: {worst_year}")

win = decline[decline['Year'].between(worst_year-10, worst_year+10)]
plt.figure(figsize=(14,6))
for c, g in win.groupby('Country'):
    g = g.sort_values('Year')
    plt.plot(g['Year'], g['Growth Rate (%)'], marker='.', alpha=0.6, label=c)
plt.axvline(worst_year, color='red', linestyle='--', label=f'Пик: {worst_year}')
plt.xlabel('Год'); plt.ylabel('Growth Rate (%)')
plt.title(f'Страны с убылью населения ({worst_year}±10 лет)')
plt.legend(fontsize=6, ncol=2); plt.grid(True)
plt.savefig('figures/1.6_declines.png'); plt.close()

# 1.7 За весь период найти страну и окно в 3 года с максимальным абсолютным ростом
def max_3yr_growth(g):
    g = g.sort_values('Year').reset_index(drop=True)
    if len(g) < 3: return pd.Series({'start': np.nan, 'gain': np.nan})
    gains = g['Population'].shift(-2) - g['Population']
    i = gains.idxmax()
    return pd.Series({'start': g.loc[i,'Year'], 'end': g.loc[i+2,'Year'], 'gain': gains.loc[i]})

res = countries.groupby('Country').apply(max_3yr_growth).dropna()
best = res.loc[res['gain'].idxmax()]
print(f"Максимальный рост за 3 года: {best.name} ({best['start']}-{best['end']}), прирост: {best['gain']:.0f} чел.")


# 2.1 Нестабильные темпы роста за последние 20 лет
last_year = countries['Year'].max()
last20 = countries[countries['Year'] >= last_year-20]
vol = last20.groupby('Country')['Growth Rate (%)'].std().dropna()
top_unstable = vol.sort_values(ascending=False).head(5)
print("Топ-5 стран с наиболее нестабильными темпами роста (за 20 лет):\n", top_unstable)

# 2.2 Годы с наибольшей разницей прироста между Китаем и Индией
cn = countries[countries['Country']=='China'].set_index('Year')['Growth Rate (%)']
ind = countries[countries['Country']=='India'].set_index('Year')['Growth Rate (%)']
diff = (cn - ind).dropna().abs()
print("Топ-3 года с макс. разницей темпов Китай-Индия:\n", diff.sort_values(ascending=False).head(3))


sel = ['China', 'India', 'Nigeria', 'United States', 'Brazil', 'Russian Federation']
plt.figure(figsize=(14,6))
for c in sel:
    g = countries[countries['Country']==c].sort_values('Year')
    if not g.empty:
        plt.plot(g['Year'], g['Population']/1e6, label=c)
plt.xlabel('Год'); plt.ylabel('Население, млн')
plt.title('Сравнительная динамика численности населения (1960–2018)')
plt.legend(); plt.grid(True)
plt.savefig('figures/3_comparison.png'); plt.close()