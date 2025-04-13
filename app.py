import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import plotly.express as px
import regex as re

vdem_pop = pd.read_csv("vdem_pop.csv")
np.random.seed(7)

regions = {
    "North America": [
        "Canada", "Mexico", "United States of America"
    ],
    "Latin America": [
        "Argentina", "Barbados", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica",
        "Cuba", "Dominican Republic", "Ecuador", "El Salvador", "Guatemala", "Guyana",
        "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Paraguay",
        "Peru", "Suriname", "Trinidad and Tobago", "Uruguay", "Venezuela"
    ],
    "Europe": [
        "Albania", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
        "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Finland", "France",
        "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia",
        "Lithuania", "Luxembourg", "Malta", "Moldova", "Montenegro", "Netherlands",
        "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia", "Serbia",
        "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom"
    ],
    "Sub-Saharan Africa": [
        "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cape Verde",
        "Central African Republic", "Chad", "Comoros", "Democratic Republic of the Congo",
        "Djibouti", "Equatorial Guinea", "Eswatini", "Ethiopia", "Gabon", "Ghana",
        "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia",
        "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Mozambique", "Namibia",
        "Niger", "Nigeria", "Republic of the Congo", "Rwanda", "Sao Tome and Principe",
        "Senegal", "Seychelles", "Sierra Leone", "Solomon Islands", "Somalia", "Somaliland",
        "South Africa", "South Sudan", "Sudan", "Tanzania", "The Gambia", "Togo",
        "Uganda", "Zambia", "Zimbabwe", "Zanzibar"
    ],
    "Middle East & North Africa (MENA)": [
        "Algeria", "Bahrain", "Egypt", "Iran", "Iraq", "Israel", "Jordan", "Kuwait",
        "Lebanon", "Libya", "Morocco", "Oman", "Palestine/Gaza", "Palestine/West Bank",
        "Qatar", "Saudi Arabia", "South Yemen", "Syria", "Tunisia", "Turkey", "United Arab Emirates", "Yemen"
    ],
    "South Asia": [
        "Afghanistan", "Bangladesh", "Bhutan", "India", "Maldives", "Nepal", "Pakistan", "Sri Lanka"
    ],
    "East Asia": [
        "China", "Hong Kong", "Japan", "Mongolia", "North Korea", "South Korea", "Taiwan"
    ],
    "Southeast Asia": [
        "Burma/Myanmar", "Cambodia", "Indonesia", "Laos", "Malaysia", "Philippines",
        "Singapore", "Thailand", "Timor-Leste", "Vietnam"
    ],
    "Oceania": [
        "Australia", "Fiji", "New Zealand", "Papua New Guinea", "Vanuatu"
    ]
}

# Define your function:
def compute_tsne_embeddings_and_plots(d, perp=50, ini='pca', target='pred_status', r = "Global"):
    """
    Runs t-SNE embeddings in 1D and 3D on a numeric dataset and displays 
    the results (descriptive statistics, histogram, scatter plots) in Streamlit.
    """

    # Select numeric columns and drop rows where target is missing
    if r != "Global":
        data = d[d['country_name'].isin(regions[r])]
    else:
        data = d.copy()
    data = data.dropna(subset=[target]) 
    countries = data['country_name']
    years = data['year']
    data = data.select_dtypes(include='number')
    targets = data[target]
    data = data.drop(columns=[target])

    # Drop any columns or rows still containing NaNs
    data = data.dropna(axis=1).dropna(axis=0)

    st.write("Number of features:", len(data.keys()))
    st.write("**Data Description**")
    st.write(data.describe())

    # Min-max normalization
    X = (data - data.min()) / (data.max() - data.min() + 0.1)

    # --- 1D EMBEDDING ---
    X_embedded = TSNE(n_components=1, learning_rate='auto', init=ini, perplexity=perp, random_state=17).fit_transform(X)
    

    # --- 3D EMBEDDING ---
    X3_embedded = TSNE(n_components=3, learning_rate='auto', init=ini, perplexity=perp, random_state=17).fit_transform(X)
    df_3d = pd.DataFrame({
        'year': years,
        'x': X3_embedded[:, 0],
        'y': X3_embedded[:, 1],
        'z': X3_embedded[:, 2],
        'label': targets,
        'country': countries
    })

    fig3 = px.scatter_3d(
        df_3d,
        x='x',
        y='y',
        z='z',
        color='label',
        color_continuous_scale='Pinkyl',
        hover_name='country',
        hover_data=['year'],
        title='Interactive 3D t-SNE'
    )
    fig3.update_traces(marker=dict(size=4, opacity=0.85))
    fig3.update_layout(
        width = 1000,
        height = 800,
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        ),
        legend_title_text='Freedom'
    )
    st.plotly_chart(fig3)

    # Return embeddings if needed elsewhere
    return X_embedded, X3_embedded

def top_correlations(base_df, column: str):
    """
    Compute correlation for each column in the DataFrame with the specified column.
    """
    df = base_df.dropna(subset=[column])
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df = df.select_dtypes(include=["number"]) #only numerical data

    correlations = df.corr()[column].drop(labels=[column])  #exclude self-correlation
    top_correlations = correlations.abs().sort_values(ascending=False) #sort by absolute value
    top_100 = top_correlations.head(100).index.tolist()

    return top_100, correlations


def print_correlations(top, corr_values, num, count_flag=0):
    """
    Prints and returns the top correlations that meet the avaliable data count flag.
    """
    filtered_top = []
    for col in top[:num]:
        if vdem_pop[col].count() > count_flag:
            #print(f"{col}: {corr_values[col]:.4f};                    " + "Count: " + str(vdem[col].count()))
            filtered_top.append(col)
    return filtered_top

def imputing_model(base_df, column: str, num_feats = 100, split = 15):
    """
    Train and evaluate a model to impute missing values in the specified column.

    Random Forest is for any vdem data we will impute, not sure if will use yet.
    """

    if len(base_df[column].unique()) > 10:
        model = DecisionTreeClassifier
        metric = r2_score
        alpha_mod = 0.01
    else:
        model = DecisionTreeClassifier
        metric = accuracy_score
        alpha_mod = 0


    #making sure the training df has no nulls for column to be imputed
    df = base_df.dropna(subset=[column])
    top, corrs = top_correlations(base_df, column)
    filtered_top = print_correlations(top, corrs, 100, 9259)

    # Exclude high-level variables that dominate feature importance
    excluded_features = [
      'v2x_api', 'v2x_libdem', 'v2x_mpi', 'e_lexical_index',
      'v2x_accountability', 'v2x_partipdem', 'v2x_clpol',
      'v2x_veracc', 'v2x_frassoc_thick', 'v2x_delibdem'
        ]

    filtered_top = [feat for feat in filtered_top if feat not in excluded_features]


    X = df[filtered_top[:num_feats]]
    y = df[column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7) #20% saved for test
    clf = model(min_samples_split=15, random_state=42) if model is DecisionTreeClassifier else model(min_samples_split=15, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    train_score = metric(y_train, clf.predict(X_train))
    test_score = metric(y_test, y_pred)
    st.write(f"**Training on {column}**")
    st.write("Train:", train_score)
    st.write("Test:", test_score)

    print("Importances for", column)

    importances = clf.feature_importances_
    std = np.std([clf.feature_importances_ for _ in range(100)], axis=0)
    features = X.columns

    indices = np.argsort(importances)[-15:][::-1]
    top_features = features[indices]
    top_importances = importances[indices]
    top_std = std[indices]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(top_features, top_importances, xerr=top_std, align='center')
    ax2.set_xlabel("Feature Importance")
    ax2.set_ylabel("Feature")
    ax2.invert_yaxis()  # highest importance at the top
    st.pyplot(fig2)

    return clf, filtered_top, corrs

def filter_by_region(df, region_name):
    if region_name not in regions:
        return df
    return df[df['country_name'].isin(regions[region_name])]

def run_model_for_region(region_name, sp):
    region_data = filter_by_region(vdem_pop, region_name)
    print(f"Running model for {region_name}...")
    model, top_features, corrs = imputing_model(region_data, 'e_fh_status', split=sp)
    status = ['Free', 'Partially Free', 'Not Free']
    d = 3
    colors = ['mediumorchid', 'thistle', 'lime']
    
    fig, ax1 = plt.subplots(figsize=[25, 12], dpi=600)

    # this took way too long to figure out
    
    artists = plot_tree(model, feature_names=top_features, class_names=status,
                             filled=False, rounded=True, max_depth=d, ax=ax1, node_ids=True)
    for artist in artists:
        if artist.get_bbox_patch():
            label = artist.get_text()
            id = re.search(r'node #(\d+)', label) #regex magic
    
            if id:
                node_id = int(id.group(1))
                value = model.tree_.value[node_id][0]
                impurity = model.tree_.impurity[node_id]
                majority_class = np.argmax(value)
    
                r, g, b = to_rgb(colors[majority_class])
                f = impurity * 1.5 #normalizing for 3 classes
                artist.get_bbox_patch().set_facecolor((
                    f + (1 - f) * r,
                    f + (1 - f) * g,
                    f + (1 - f) * b
                ))
                artist.get_bbox_patch().set_edgecolor('black')
    ax1.set_title(f"Decision Tree ({region_name})", size=16)
    st.pyplot(fig)
    return model, top_features, corrs


def main():
    """
    Streamlit App Entry Point
    """
    st.title("Freedom Analysis")

    # --- Data Upload ---
    df = pd.read_csv("model_dataset_cleaned.csv")

    # --- General Parameters -----
    region = st.selectbox('Region To analyze: ', ("North America", "Latin America", "Europe", "Sub-Saharan Africa", 
                                         "Middle East & North Africa (MENA)", "South Asia", "East Asia", "Southeast Asia", "Oceania"))

    # --- Parameter Inputs ---
    perp = 50
    ini = st.selectbox("Initialization", ("pca", "random"))
    target_col = st.selectbox("Target Column", ("pred_status", "e_fh_status"))

    split = 15

    # --- Run t-SNE on button click ---
    if st.button("Run t-SNE and Decision Tree"):
        st.write(f"Running t-SNE with Perplexity={perp} and init='{ini}' on target '{target_col}'...")
        compute_tsne_embeddings_and_plots(df, perp=perp, ini=ini, target=target_col, r = region)
        st.write(f"Running Decision Tree Model with minimum_split={split} on region={region}...")
        run_model_for_region(region, split)


if __name__ == "__main__":
    main()
