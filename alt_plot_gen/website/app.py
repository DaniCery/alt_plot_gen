import streamlit as st
import pandas as pd
import requests
import os
import torch

from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from transformers import GPT2Tokenizer
from alt_plot_gen.interface.main  import text_generation
from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH


#url = 'https://alt-plot-gen.xxxxx/generate'
#params={'Title': selected_plot,
#        'Release Year': reduced_df['Release Year'][int(title_id)],
#        'Genre': new_genre
#        }
#generated_plot = requests.get(url, params=params).json

# ----------------------------------

test_set_file = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "test_set_demo.csv")

df = pd.read_csv(test_set_file)
reduced_df = df[['Title', 'Release Year', 'Genre', 'Plot', 'True_end_plot']]
available_genres = ('', 'Action', 'Comedy', 'Drama', 'Horror')

#nbr_trail_words =
model_path = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "trained_model.pt")
model = torch.load(model_path)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

st.set_page_config(layout="wide")

def create_aggrid(head, df):
    st.subheader(head)
    new_df = pd.DataFrame(reduced_df[head])
    new_gd = GridOptionsBuilder.from_dataframe(new_df)
    new_gd.configure_selection()
    gridoptions = new_gd.build()
    grid_new_table = AgGrid(new_df, height=250, gridOptions=gridoptions,
                update_mode=GridUpdateMode.SELECTION_CHANGED)
    return grid_new_table

def split_plot(plot, nbr_trail_words):
    start_plot = ' '.join(plot.split()[:-nbr_trail_words])
    true_end_plot = ' '.join(plot.split()[-nbr_trail_words:])
    return start_plot, true_end_plot

st.header('Alternative Plot Ending Generator')

st.subheader('Select a movie')

selection = st.radio('', ('By title', 'By genre', 'By release year'))

if selection == 'By title':
    grid_title_table = create_aggrid('Title', reduced_df)
elif selection == 'By genre':
    col1, col2 = st.columns(2)
    with col1:
        no_dup_df = reduced_df.drop_duplicates(subset='Genre')
        #st.dataframe(no_dup_df)
        grid_genre_table = create_aggrid('Genre', no_dup_df)
    with col2:
        pass
else:
    col1, col2 = st.columns(2)
    with col1:
        no_dup_df = reduced_df.drop_duplicates(subset='Release Year')
        grid_rel_year_table = create_aggrid('Release Year', no_dup_df)

if selection == 'By genre':
    selected = grid_genre_table.selected_rows
    if selected:
        feature_id = selected[0]['Genre']
        filtered_title = reduced_df[reduced_df['Genre']==feature_id]
        with col2:
            grid_title_table = create_aggrid('Title', filtered_title)

elif selection == 'By release year':
    selected = grid_rel_year_table.selected_rows
    if selected:
        feature_id = selected[0]["Release Year"]
        filtered_title = reduced_df[reduced_df['Release Year']==feature_id]
        with col2:
            grid_title_table = create_aggrid('Title', filtered_title)

if 'grid_title_table' in locals():
    selected_movie = grid_title_table.selected_rows
    if selected_movie:
        title_id = selected_movie[0]["_selectedRowNodeInfo"]["nodeId"]

if 'title_id' in locals():
    start_plot  = reduced_df['Plot'][int(title_id)]
    true_end_plot = reduced_df['True_end_plot'][int(title_id)]

    st.subheader('Original plot without ending')
    plot_height = round(len(start_plot) * 7 / 750 / 3)
    start_plot_txt = st.text_area('', start_plot, height=plot_height)

    col4, col5, col6, col7, col8 = st.columns(5)
#    with col4:
#        new_genre = st.selectbox('Select new genre for alternative ending:', available_genres)
    with col8:
        generate_bt = st.button("Generate")

    col9, col10 = st.columns(2)
    with col9:
        st.subheader('Original plot ending')
        true_end_plot_txt = st.text_area('', true_end_plot)
    if generate_bt:
        selected_plot = reduced_df['Plot'][int(title_id)]
        #Run the functions to generate the alternative endings
        generated_plot, full_test_generated_plot = text_generation(model, tokenizer, selected_plot)

        with col10:
            st.subheader('Generated plot ending')
            gen_end_plot = st.text_area('', generated_plot)
