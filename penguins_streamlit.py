import streamlit as st
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

st.title('企鹅分类')
st.write("应用程序使用6个输入来预测企鹅的种类，使用建立在帕尔默企鹅数据集上的模型。使用下面的表单开始吧！")
penguin_file = st.file_uploader(label="Upload your own penguin data")

if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle','rb')
    map_pickle = open('output_penguin.pickle','rb')

    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
    penguin_df = pd.read_csv('penguins.csv')
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[[
        'island',
        'bill_length_mm',
        'bill_depth_mm',
        'flipper_length_mm',
        'body_mass_g',
        'sex'
    ]]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train,x_test, y_train, y_test = train_test_split(features,output, test_size=0.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_test.values)
    score = round(accuracy_score(y_pred, y_test), 2)
    st.write(f'''我们基于这些数据训练了一个随机森林模型，其准确率达到了 {score}%！请使用以下输入来试用该模型。''')

with st.form('user_inputs'):
    island = st.selectbox('岛屿名字', options=['比科斯岛','梦幻岛', '托格森岛'])
    sex = st.selectbox('性别',options=['雌性','雄性'])
    bill_length = st.number_input('嘴长(mm)', min_value=0)
    bill_depth = st.number_input('喙深(mm)', min_value=0)
    flipper_length = st.number_input('鳍长(mm)', min_value=0)
    body_mass = st.number_input('体重(g)', min_value=0)
    user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]
    st.form_submit_button(label='提交')

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == '比科斯岛':
    island_biscoe = 1
elif island == '梦幻岛':
    island_dream = 1
elif island == '托格森岛':
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == '雌性':
    sex_female = 1
elif sex == '雄性':
    sex_male = 1

new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male
        ]
    ]
)
st.subheader("预测你的企鹅种类：")
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f'预测的企鹅属于 {prediction_species} 种类')
st.write("""我们使用了一种机器学习（随机森林）模型来进行物种预测，此次预测所使用的特征按其相对重要性进行了排序如下。""")
st.image('feature_importance.png')

st.write('以下是针对每种企鹅种类的连续变量所绘制的直方图。竖线代表输入的数值。')

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_length_mm"], hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Bill Length by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"], hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["flipper_length_mm"], hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)
