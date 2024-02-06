import pandas as pd
from imblearn.over_sampling import SMOTENC
import csv

# Load data
data = pd.read_excel('Studentmeal_data.xls')
X = data[['Economic level', 'Month', 'Classification_Code', "Salmonella", "Listeria monocytogenes", "Packing_Code", 'Aerobic plate counts_Normalization', 'Escherichia coli_Normalization', "Bacillus cereus_Normalization", "Year"]]
Y = data['State']

# Write data
f = open('SMOTENC_data.csv', 'w+', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(
    ('Economic level', 'Month', 'Classification', "Salmonella", "Listeria monocytogenes", "Packing", 'Aerobic plate counts', 'Escherichia coli', "Bacillus cereus", "Year", 'State'))

# Generating new data using SMOTENC algorithm
smotenc = SMOTENC(random_state=5, categorical_features=[2, 4, 5, 3], k_neighbors=2)
X_resampled, Y_resampled = smotenc.fit_resample(X, Y)

# According to the standards formulated by Guangdong Province, necessary corrections are made to the generated data to ensure their accuracy and compliance.
# Among them, the numerator represents the qualified critical value.
Yt = []
APC_qualified = 10 ** 5 / max(data['Aerobic plate counts'])
Ecoli_qualified = 10 ** 2 / max(data['Escherichia coli'])
Bcereu_qualified = 10 ** 5 / max(data['Bacillus cereus'])

for i in range(len(X_resampled['Economic level'])):
    if X_resampled['Aerobic plate counts_Normalization'][i] < APC_qualified and X_resampled['Escherichia coli_Normalization'][i] < Ecoli_qualified and \
            X_resampled['Salmonella'][i] < 1 and X_resampled['Listeria monocytogenes'][
        i] < 1 and X_resampled['Bacillus cereus_Normalization'][i] < Bcereu_qualified:
        Yt.append(0)
    else:
        Yt.append(1)

for i in range(len(X_resampled['Economic level'])):
    writer.writerow([X_resampled['Economic level'][i], X_resampled['Month'][i],
                     X_resampled['Classification_Code'][i], X_resampled['Salmonella'][i],
                     X_resampled['Listeria monocytogenes'][i], X_resampled['Packing_Code'][i],
                     X_resampled['Aerobic plate counts_Normalization'][i],
                     X_resampled['Escherichia coli_Normalization'][i],
                     X_resampled['Bacillus cereus_Normalization'][i],
                     X_resampled['Year'][i],
                     Yt[i]])

