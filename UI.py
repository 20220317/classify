import pickle

import tensorflow as tf
from transformers import TFRobertaModel, RobertaTokenizer, BertTokenizer, TFBertModel
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# 配置页面
st.set_page_config(
    page_title="情感分析聊天机器人",
    page_icon="💬",
    layout="wide"
)

# 模型路径配置
MODEL_PATHS = {
    "RoBERTa": r"Roberta\best_roberta_model.h5",
    "BERT": r"BERT\best_bert_model.h5",
    "CNN": r"CNN\final_cnn_model.h5",
    "RNN": r"CNN\final_cnn_model.h5",
    "LSTM": r"LSTM\best_lstm_model.h5",
    "SVM": r"SVM\SVM_model.pkl",
    "随机森林": r"SVM\SVM_model.pkl",
    "朴素贝叶斯": r"朴素贝叶斯\beiysesi_model.pkl",
}


# 检查模型文件是否存在
def check_model_files():
    missing_models = []
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            missing_models.append(f"{name}模型 ({path})")
    return missing_models


# 缓存模型加载
@st.cache_resource
def load_model(model_name):
    if model_name in ["RoBERTa", "BERT"]:
        # 加载Transformer模型
        if model_name == "RoBERTa":
            tokenizer = RobertaTokenizer.from_pretrained(r'RoBerta\roberta-base')
            base_model = TFRobertaModel.from_pretrained(r'RoBerta\roberta-base')
        else:  # BERT
            tokenizer = BertTokenizer.from_pretrained(r'BERT\bert-base-uncased')
            base_model = TFBertModel.from_pretrained(r'BERT\bert-base-uncased')

        max_length = 128

        # 构建模型
        input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

        outputs = base_model(input_ids, attention_mask=attention_mask)
        pooled = outputs[1]

        x = tf.keras.layers.Dropout(0.3)(pooled)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(2e-5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.load_weights(MODEL_PATHS[model_name])

        return {
            'model': model,
            'tokenizer': tokenizer,
            'max_length': max_length,
            'type': 'transformer'
        }

    elif model_name in ["CNN", "LSTM", "RNN"]:
        # 加载深度学习模型
        max_length = 100

        if model_name == "CNN":
            model_path = r'CNN\final_cnn_model.h5'
            tokenizer_path = r'CNN\tokenizer.pkl'
        elif model_name == "LSTM":
            model_path = r'LSTM\best_lstm_model.h5'
            tokenizer_path = r'LSTM\lstmtokenizer.pkl'
        else:
            model_path=r"CNN\final_cnn_model.h5"
            tokenizer_path=r"CNN\tokenizer.pkl"
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)

        return {
            'model': model,
            'tokenizer': tokenizer,
            'max_length': max_length,
            'type': 'deep_learning'
        }

    else:  # SVM, 随机森林, 朴素贝叶斯
        # 加载机器学习模型
        model = joblib.load(MODEL_PATHS[model_name])
        vectorizer = joblib.load("vectorizer.pkl")

        return {
            'model': model,
            'vectorizer': vectorizer,
            'type': 'machine_learning'
        }


# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = "RoBERTa"
if 'model_loaded' not in st.session_state:
    missing_models = check_model_files()
    if missing_models:
        st.warning(f"以下模型文件不存在: {', '.join(missing_models)}")
        st.warning("这些模型将无法使用，请确保模型文件路径正确。")
    try:
        st.session_state.model_info = load_model(st.session_state.current_model)
        st.session_state.model_loaded = True
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()


# 定义预测函数
def predict_sentiment(text):
    model_info = st.session_state.model_info

    if model_info['type'] == 'transformer':
        # Transformer模型 (RoBERTa, BERT)
        encoded = model_info['tokenizer'](
            text,
            max_length=model_info['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        pred = model_info['model'].predict([encoded['input_ids'], encoded['attention_mask']])
        score = float(pred[0][0])

    elif model_info['type'] == 'deep_learning':
        # 深度学习模型 (CNN, RNN, LSTM)
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # 文本向量化
        sequence = model_info['tokenizer'].texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=model_info['max_length'])

        pred = model_info['model'].predict(padded_sequence)
        score = float(pred[0][0])


    else:  # 机器学习模型

        # 直接使用原始文本进行预测

        if hasattr(model_info['model'], 'predict_proba'):

            pred_proba = model_info['model'].predict_proba([text])

            score = float(pred_proba[0][1])  # 假设索引1是积极类

        else:

            pred = model_info['model'].predict([text])

            score = float(pred[0])

    sentiment = "积极" if score > 0.5 else "消极"
    return sentiment, score


# 聊天界面
st.title("情感分析聊天机器人")
st.markdown(f"💬 使用 **{st.session_state.current_model}** 模型分析文本情感倾向（积极/消极）")

# 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['role'] == 'assistant':
            # 根据情感结果显示不同颜色
            if msg['sentiment'] == '积极':
                st.success(f"情感: {msg['sentiment']} (概率: {msg['score']:.4f})")
                st.progress(msg['score'])
            else:
                st.error(f"情感: {msg['sentiment']} (概率: {1 - msg['score']:.4f})")
                st.progress(1 - msg['score'])
            st.caption(f"使用 {msg['model']} 模型分析")

# 用户输入框
user_input = st.chat_input("请输入要分析的文本...")

if user_input:
    # 添加用户消息到历史
    st.session_state.messages.append({
        'role': 'user',
        'content': user_input
    })

    # 显示用户消息
    with st.chat_message('user'):
        st.markdown(user_input)

    # 生成AI回复
    with st.chat_message('assistant'):
        with st.spinner(f"正在使用 {st.session_state.current_model} 模型分析情感..."):
            sentiment, score = predict_sentiment(user_input)
            st.markdown(user_input)
            # 根据情感结果显示不同颜色
            if sentiment == '积极':
                st.success(f"情感: {sentiment} (概率: {score:.4f})")
                st.progress(score)
            else:
                st.error(f"情感: {sentiment} (概率: {1 - score:.4f})")
                st.progress(1 - score)
            st.caption(f"使用 {st.session_state.current_model} 模型分析")

    # 添加AI回复到历史
    st.session_state.messages.append({
        'role': 'assistant',
        'content': user_input,
        'sentiment': sentiment,
        'score': score,
        'model': st.session_state.current_model
    })

# 侧边栏 - 模型选择和清空聊天功能
with st.sidebar:
    st.header("模型选择")
    selected_model = st.selectbox(
        "选择情感分析模型",
        list(MODEL_PATHS.keys()),
        index=list(MODEL_PATHS.keys()).index(st.session_state.current_model)
    )

    if selected_model != st.session_state.current_model:
        with st.spinner(f"正在加载 {selected_model} 模型..."):
            try:
                st.session_state.model_info = load_model(selected_model)
                st.session_state.current_model = selected_model
                st.success(f"{selected_model} 模型加载成功！")
                st.experimental_rerun()  # 刷新页面应用新模型
            except Exception as e:
                st.error(f"加载 {selected_model} 模型失败: {str(e)}")

    st.header("功能选项")
    if st.button("清空聊天记录"):
        st.session_state.messages = []
        st.experimental_rerun()

# 页脚
st.markdown("""
---
*支持多种情感分析模型，包括深度学习和机器学习方法*
""")