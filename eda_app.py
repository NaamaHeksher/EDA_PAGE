import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='EDA App', layout='wide')

@st.cache_data
def load_sample_data():
    return pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': np.random.rand(100),
        'D': np.random.rand(100),
    })


def run_kmeans(data, n_clusters=3, max_iter=100):
    values = data.to_numpy().astype(float)
    rng = np.random.default_rng(42)
    centroids = values[rng.choice(values.shape[0], size=n_clusters, replace=False)]

    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([values[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                                  for i in range(n_clusters)])
        if np.allclose(new_centroids, centroids, atol=1e-5):
            break
        centroids = new_centroids

    return labels, centroids


def get_dataset():
    with st.sidebar:
        st.header('Dataset controls')
        uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])
        use_sample = st.checkbox('Use sample data', value=True)
        show_data = st.checkbox('Show raw data', value=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif use_sample:
        df = load_sample_data()
    else:
        st.warning('Please upload a CSV file or enable sample data.')
        return None, False

    return df, show_data


def render_overview(df, show_data):
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    rows, columns = df.shape
    missing_percentage = df.isnull().mean().mean() * 100

    st.title('Exploratory Data Analysis')
    st.write('Upload a CSV file or use the sample dataset to explore metrics, statistics, and visualizations.')

    st.subheader('Dataset overview')
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric('Rows', rows)
    metric_col2.metric('Columns', columns)
    metric_col3.metric('Missing %', f'{missing_percentage:.2f}%')
    metric_col4.metric('Numeric columns', len(numeric_columns))

    if show_data:
        st.write('### Raw data preview')
        st.dataframe(df, use_container_width=True)

    st.write('### Summary statistics')
    with st.expander('Show descriptive statistics'):
        st.dataframe(df.describe(include='all').T)

    st.write('### Column details')
    with st.expander('Show column data types and missing values'):
        summary = pd.DataFrame({
            'dtype': df.dtypes.astype(str),
            'missing': df.isnull().sum(),
            'unique': df.nunique(dropna=False),
        })
        st.dataframe(summary)

    if numeric_columns:
        st.write('### Visualizations')

        hist_col = st.selectbox('Select numeric column for histogram', numeric_columns, index=0)
        fig, ax = plt.subplots()
        ax.hist(df[hist_col].dropna(), bins=30, color='cornflowerblue', edgecolor='black')
        ax.set_title(f'Histogram of {hist_col}')
        ax.set_xlabel(hist_col)
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close(fig)

        if len(numeric_columns) >= 2:
            x_col = st.selectbox('Select X-axis for scatter plot', numeric_columns, index=0)
            y_col = st.selectbox('Select Y-axis for scatter plot', numeric_columns, index=1)
            fig2, ax2 = plt.subplots()
            ax2.scatter(df[x_col], df[y_col], alpha=0.7)
            ax2.set_title(f'{y_col} vs {x_col}')
            ax2.set_xlabel(x_col)
            ax2.set_ylabel(y_col)
            st.pyplot(fig2)
            plt.close(fig2)

        st.write('### Correlation matrix')
        corr = df[numeric_columns].corr()
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        cax = ax3.matshow(corr, cmap='coolwarm')
        fig3.colorbar(cax)
        ax3.set_xticks(range(len(corr.columns)))
        ax3.set_yticks(range(len(corr.index)))
        ax3.set_xticklabels(corr.columns, rotation=45, ha='left')
        ax3.set_yticklabels(corr.index)
        ax3.set_title('Correlation matrix')
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info('No numeric columns are available for visualizations.')


def render_clustering(df, show_data):
    st.title('Clustering analysis')
    st.write('Use clustering to group similar observations in your numeric dataset.')

    if df is None:
        st.info('Upload a dataset or enable sample data to run clustering.')
        return

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not numeric_columns:
        st.warning('No numeric columns available for clustering.')
        return

    if show_data:
        with st.expander('Show dataset preview'):
            st.dataframe(df[numeric_columns].head(20), use_container_width=True)

    selected_columns = st.multiselect('Select numeric columns for clustering', numeric_columns, default=numeric_columns[:2])
    if len(selected_columns) < 2:
        st.warning('Select at least two numeric columns for clustering.')
        return

    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)
    labels, centroids = run_kmeans(df[selected_columns].dropna(), n_clusters=n_clusters)
    st.subheader('Cluster results')
    df_clusters = df.loc[df[selected_columns].dropna().index].copy()
    df_clusters['cluster'] = labels
    st.dataframe(df_clusters.head(20), use_container_width=True)

    st.write('### Cluster counts')
    st.bar_chart(df_clusters['cluster'].value_counts().sort_index())

    if len(selected_columns) >= 2:
        x_col = selected_columns[0]
        y_col = selected_columns[1]
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_clusters[x_col], df_clusters[y_col], c=labels, cmap='tab10', alpha=0.7)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('Clustering result')
        st.pyplot(fig)
        plt.close(fig)

    st.write('### About this clustering')
    st.markdown(
        'This clustering page uses a simple k-means algorithm implemented in NumPy. '
        'It helps reveal structure in numeric data by grouping rows with similar values.'
    )


def render_about():
    st.title('About the system')
    st.write('This Exploratory Data Analysis Web App helps you inspect datasets, visualize numeric relationships, and run clustering analysis.')

    st.markdown(
        '- **Home / Overview**: See dataset metrics, missing values, summary statistics, and visualizations.\n'
        '- **Clustering**: Group your numeric dataset into clusters using k-means.\n'
        '- **About**: Read about the app and its capabilities.'
    )

    st.markdown('''
**How to use the app**

1. Upload a CSV file, or use the built-in sample dataset.
2. Explore the Overview page for data summaries and plots.
3. Open the Clustering page to select numeric features and run group analysis.
4. Visit About for app information and purpose.
''')

    st.markdown('''
**System features**

- CSV upload and sample dataset support
- Data overview, metrics, and visuals
- K-means clustering for numeric columns
- Simple, interactive Streamlit user interface
''')


def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select page', ['Overview', 'Clustering', 'About'])
    df, show_data = get_dataset()

    if page == 'Overview':
        if df is not None:
            render_overview(df, show_data)
    elif page == 'Clustering':
        render_clustering(df, show_data)
    elif page == 'About':
        render_about()


if __name__ == '__main__':
    main()
