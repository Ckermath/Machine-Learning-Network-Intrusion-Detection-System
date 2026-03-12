import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

df = pd.read_csv("IoT_Intrusion.csv")

#code that helps to calculate the imbalance ratio
def calculate_rho(data):
    print("Maximum value")
    print(max(df['label'].value_counts()))
    print("Minimum value")
    print(min(df['label'].value_counts()))
    print("Rho(p) value")
    print(max(df['label'].value_counts())/min(df['label'].value_counts()))

#code to regroup labels within the CSV file
for i in df.index:
    if 'DDoS' in df['label'][i]:
        df.at[i, 'label'] = "DDoS"
    if 'Flood' in df['label'][i]:
        df.at[i, 'label'] = "DoS"
    if 'Recon' in df['label'][i] or 'VulnerabilityScan' in df['label'][i]:
        df.at[i, 'label'] = "Recon"
    if 'Mirai' in df['label'][i]:
        df.at[i, 'label'] = "Mirai"
    if 'Uploading_Attack' in df['label'][i] or 'XSS' in df['label'][i] or 'BrowserHijacking' in df['label'][i] or 'Backdoor_Malware' in df['label'][i] or 'SqlInjection' in df['label'][i] or 'CommandInjection' in df['label'][i]:
        df.at[i, 'label'] = "Web-Based"
    if 'DNS_Spoofing' in df['label'][i]  or 'MITM-ArpSpoofing' in df['label'][i]:
        df.at[i, 'label'] = "Spoofing"
    if 'DictionaryBruteForce' in df['label'][i]:
        df.at[i, 'label'] = "Brute Force"
df_features = df.drop('label', axis=1)
df_label = df['label']
calculate_rho(df)

#scale data values
transformer = MinMaxScaler().fit(df_features)
df_features=pd.DataFrame(transformer.transform(df_features),columns=df_features.columns,index=df.index)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_features, df_label,
test_size=0.2, random_state=42)

#code that uses SMOTE to address data imbalance
sm = SMOTE()
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
print('Resampled (SMOTE) dataset shape %s' % Counter(y_train_smote))

#code that uses Random Over Sampling to address data imbalance
ros = RandomOverSampler(random_state=42)
X_res_ros, y_res_ros = ros.fit_resample(X_train, y_train)
#code that uses Random Under Sampling to address data imbalance
rus = RandomUnderSampler(random_state=42)
X_res_rus, y_res_rus = rus.fit_resample(X_train,y_train)

#save file to pickled files
with open('X_train.sav','wb') as f:
    pickle.dump(X_train,f)
with open('X_test.sav','wb') as f:
    pickle.dump(X_test,f)
with open('y_train.sav','wb') as f:
    pickle.dump(y_train,f)
with open('y_test.sav','wb') as f:
    pickle.dump(y_test,f)
with open('X_train_smote.sav', 'wb') as f:
    pickle.dump(X_train_smote,f)
with open('y_train_smote.sav','wb') as f:
    pickle.dump(y_train_smote, f)

print(df['label'].value_counts())
print(df_features.info())
print(df_features.describe().T)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print('Resampled (ROS) dataset shape %s' % Counter(y_res_ros))
print('Resampled (RUS) dataset shape %s' % Counter(y_res_rus))