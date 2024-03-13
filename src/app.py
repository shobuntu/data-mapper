import streamlit as st
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-MiniLM-L6-v2')



def sentence_creation(dt):
    col_names = dt.iloc[0].tolist()
    indexes = dt.columns.tolist()

    sentence_list = []
    for columns in dt.columns:
        sentences = ' '.join(dt[columns].astype(str))
        sentence_list.append(sentences)
    return {
        "sentences": sentence_list,
        "columns_names": col_names,
        "indices": indexes
    }


def generate_missing_values(col):
    missing_count = 1
    for i in range(len(col)):
        if pd.isnull(col[i]):
            col[i] = 'missing' + str(missing_count)
            missing_count += 1
    return col


def display_file_upload(input_file_name):
    df = pd.DataFrame(columns=[])

    uploaded_file = input_file_name
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        try:
            if file_extension == "txt":
                df = pd.read_csv(uploaded_file, delimiter="\t", header=None, skip_blank_lines=True)
            elif file_extension == "csv":
                df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines='skip', header=None,
                                 skip_blank_lines=True)
            elif file_extension == "xlsx":
                df = pd.read_excel(uploaded_file, header=None)
            else:
                st.error(f"Unsupported file format, please upload TSV, CSV, or XLSX file")
                return
        except Exception as e:
            st.error(f"An error occurred while reading the file: {str(e)}")
        return df


def read_csv(input_file_name):
    df1 = ''
    print(input_file_name)
    spt = ',' if input_file_name.name.lower().endswith("csv") else "\t" if input_file_name.name.lower().endswith("txt") \
        else "|"
    try:
        df1 = pd.read_csv(input_file_name, sep=spt, header=None, skip_blank_lines=True)
    except ValueError as err:
        if "utf-8" in str(err):
            df1 = pd.read_csv(input_file_name, encoding='windows-1252', sep=spt, header=None, skip_blank_lines=True)
    return df1


def heat_map(matric):
    pivot_table = pd.pivot_table(matric, values='Score', index='Query_columns', columns='Corpus_columns')
    f, ax = plt.subplots(figsize=(15, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.2f', linewidths=.5, ax=ax)
    plt.xlabel('Matrix 1 columns')
    plt.ylabel('Matrix 2 columns')
    return f


def processing(input_data1, input_data2):
    source_f = input_data1
    target_f = input_data2

    src_sent = sentence_creation(source_f)
    trg_sent = sentence_creation(target_f)

    corpus = src_sent["sentences"]

    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    print("corpus_encoding_done")

    queries = trg_sent["sentences"]
    query_embeddings = embedder.encode(queries, convert_to_tensor=True)
    print("query_encoding_done")

    top_k = min(1, len(corpus))

    results = []

    for query_idx, query_embedding in enumerate(query_embeddings):
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        for score, idx in zip(top_results[0], top_results[1]):
            result_row = {
                'Query': queries[query_idx],
                'Query_columns': trg_sent['columns_names'][query_idx],
                'Corpus_columns': src_sent['columns_names'][idx.item()],
                'Query Index': query_idx,
                'Corpus Index': idx.item(),
                'Sentence': corpus[idx],
                'Score': score.item(),
            }
            results.append(result_row)
    results = pd.concat([pd.DataFrame(results)])

    results['Query'] = results['Query'].str[:50]
    results['Sentence'] = results['Sentence'].str[:50]

    results = results.apply(generate_missing_values, axis=0)
    rs = results[results["Score"] > 0.7]
    MiniLM = rs
    return MiniLM


def main():
    st.title('DATA MAPPER')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            file_1 = st.file_uploader("source data", type=["csv", "xlsx"])
        with col2:
            file_2 = st.file_uploader("target data", type=["csv", "xlsx"])

    preview_button = st.button("Process", key="preview")
    st.markdown(
        """<style>
        .preview-button{
            background-color: #8B0000;
            color: #8B0000;
            padding: 10px 20px;
            font-size: 16px;
            border-radius:5px;
            border: none;
        }
        </style>""",
        unsafe_allow_html=True
    )

    if preview_button and file_1 and file_2:
        df1 = display_file_upload(file_1)
        df2 = display_file_upload(file_2)

        with st.container():
            col3, _, col4 = st.columns([1, 0.1, 1])

            with col3:
                st.markdown("<h3 class='header-text'> Dataset 1 </h3>", unsafe_allow_html=True)
                st.dataframe(df1)
            with col4:
                st.markdown("<h3 class='header-text'> Dataset 2 </h3>", unsafe_allow_html=True)
                st.dataframe(df2)
        result_table1 = processing(input_data1=df1, input_data2=df2)
        st.markdown("***")
        st.markdown("***")
        st.markdown("Mapped Columns")
        st.dataframe(result_table1)

        st.markdown("***")
        st.markdown("***")
        st.text("X and Y axis depicts columns of Input Matrix 1 & 2 and similarity scores")
        st.pyplot(heat_map(result_table1))


if __name__ == "__main__":
    main()
