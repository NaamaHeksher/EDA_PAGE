import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='SurveyGIS', layout='wide')

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
        st.header('בקרי מדידות')
        uploaded_file = st.file_uploader('העלה קובץ CSV', type=['csv'])
        use_sample = st.checkbox('השתמש בנתונים לדוגמה', value=True)
        show_data = st.checkbox('הצג נתונים גולמיים', value=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif use_sample:
        df = load_sample_data()
    else:
        st.warning('אנא העלה קובץ CSV או הפוך את נתוני הדוגמה לפעילים.')
        return None, False

    return df, show_data


def render_overview(df, show_data):
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    rows, columns = df.shape
    missing_percentage = df.isnull().mean().mean() * 100

    st.title('SurveyGIS - ניתוח מדידות')
    st.write('העלה קובץ מדידות או השתמש בנתונים לדוגמה כדי לחקור מטריקות, סטטיסטיקה וויזואליזציות של המדידות.')

    st.subheader('סקירת מערכת הנתונים')
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric('שורות', rows)
    metric_col2.metric('עמודות', columns)
    metric_col3.metric('חסר %', f'{missing_percentage:.2f}%')
    metric_col4.metric('עמודות מספריות', len(numeric_columns))

    if show_data:
        st.write('### תצוגה מקדימה של נתונים גולמיים')
        st.dataframe(df, use_container_width=True)

    st.write('### סטטיסטיקה מסכמת')
    with st.expander('הצג סטטיסטיקה תיאורית'):
        st.dataframe(df.describe(include='all').T)

    st.write('### פרטי עמודה')
    with st.expander('הצג סוגי נתונים של עמודות וערכים חסרים'):
        summary = pd.DataFrame({
            'סוג נתונים': df.dtypes.astype(str),
            'חסר': df.isnull().sum(),
            'ייחודי': df.nunique(dropna=False),
        })
        st.dataframe(summary)

    if numeric_columns:
        st.write('### ויזואליזציות')

        hist_col = st.selectbox('בחר עמודה מספרית לעלילת התאם', numeric_columns, index=0)
        fig, ax = plt.subplots()
        ax.hist(df[hist_col].dropna(), bins=30, color='cornflowerblue', edgecolor='black')
        ax.set_title(f'Histogram of {hist_col}')
        ax.set_xlabel(hist_col)
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close(fig)

        if len(numeric_columns) >= 2:
            x_col = st.selectbox('בחר את ציר X לעלילת פיזור', numeric_columns, index=0)
            y_col = st.selectbox('בחר את ציר Y לעלילת פיזור', numeric_columns, index=1)
            fig2, ax2 = plt.subplots()
            ax2.scatter(df[x_col], df[y_col], alpha=0.7)
            ax2.set_title(f'{y_col} vs {x_col}')
            ax2.set_xlabel(x_col)
            ax2.set_ylabel(y_col)
            st.pyplot(fig2)
            plt.close(fig2)

        st.write('### Correlation Matrix')
        corr = df[numeric_columns].corr()
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        cax = ax3.matshow(corr, cmap='coolwarm')
        fig3.colorbar(cax)
        ax3.set_xticks(range(len(corr.columns)))
        ax3.set_yticks(range(len(corr.index)))
        ax3.set_xticklabels(corr.columns, rotation=45, ha='left')
        ax3.set_yticklabels(corr.index)
        ax3.set_title('Correlation Matrix')
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info('אין עמודות מספריות זמינות לויזואליזציות.')


def render_clustering(df, show_data):
    st.title('ניתוח clustering של מדידות')
    st.write('השתמש בקיבוץ כדי לקבץ מדידות דומות בנתונים מספריים שלך.')

    if df is None:
        st.info('העלה מערכת נתונים או אפשר נתוני דוגמה כדי להריץ clustering.')
        return

    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    if not numeric_columns:
        st.warning('אין עמודות מספריות זמינות לקיבוץ.')
        return

    if show_data:
        with st.expander('הצג תצוגה מקדימה של מערכת הנתונים'):
            st.dataframe(df[numeric_columns].head(20), use_container_width=True)

    selected_columns = st.multiselect('בחר עמודות מספריות לקיבוץ', numeric_columns, default=numeric_columns[:2])
    if len(selected_columns) < 2:
        st.warning('בחר לפחות שתי עמודות מספריות לקיבוץ.')
        return

    n_clusters = st.slider('מספר קטגוריות', min_value=2, max_value=10, value=3)
    labels, centroids = run_kmeans(df[selected_columns].dropna(), n_clusters=n_clusters)
    st.subheader('תוצאות הקיבוץ')
    df_clusters = df.loc[df[selected_columns].dropna().index].copy()
    df_clusters['cluster'] = labels
    st.dataframe(df_clusters.head(20), use_container_width=True)

    st.write('### ספירת קטגוריות')
    st.bar_chart(df_clusters['cluster'].value_counts().sort_index())

    if len(selected_columns) >= 2:
        x_col = selected_columns[0]
        y_col = selected_columns[1]
        fig, ax = plt.subplots()
        scatter = ax.scatter(df_clusters[x_col], df_clusters[y_col], c=labels, cmap='tab10', alpha=0.7)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('Clustering Result')
        st.pyplot(fig)
        plt.close(fig)

    st.write('### אודות קיבוץ זה')
    st.markdown(
        'עמוד קיבוץ זה משתמש באלגוריתם k-means פשוט שיושם ב-NumPy. '
        'הוא עוזר לחשוף מבנה בנתונים מספריים על ידי קיבוץ שורות בעלות ערכים דומים.'
    )


def render_about():
    st.title('אודות SurveyGIS')
    st.write('מערכת מידע גאוגרפי לניהול, שליפה והצגת מדידות שנעשו בעבר, המיועדת למשרדי מדידות.')

    st.subheader('מטרת המערכת')
    st.write('''
ארגון פנים-משרדי של החומר הקיים בצורה גאוגרפית הנוחה לחיפוש ומעקב אחר מדידות. 
המערכת מאפשרת התמצאות מהירה במרחב ושליפה של פרויקטים רלוונטיים באזור המבוקש.
    ''')

    st.subheader('מקור הנתונים')
    st.write('מפות מדידה שנעשו במשרד (שחולצו מתוך קבצי DWG / DXF).')

    st.markdown('---')
    
    st.markdown(
        '- **בית / סקירה כללית**: ראה מטריקות מערכת נתונים, ערכים חסרים, סטטיסטיקה מסכמת וויזואליזציות.\n'
        '- **Clustering**: קבץ את מערכת הנתונים שלך לקטגוריות באמצעות k-means.\n'
        '- **אודות**: קרא על האפליקציה והיכולות שלה.'
    )


def main():
    st.sidebar.title('ניווט')
    page = st.sidebar.radio('בחר עמוד', ['סקירה כללית', 'קיבוץ', 'אודות'])
    df, show_data = get_dataset()

    if page == 'סקירה כללית':
        if df is not None:
            render_overview(df, show_data)
    elif page == 'קיבוץ':
        render_clustering(df, show_data)
    elif page == 'אודות':
        render_about()


if __name__ == '__main__':
    main()
