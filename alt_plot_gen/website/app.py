import streamlit as st
import pandas as pd
#import requests
import os
import torch

from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import DataReturnMode
from transformers import GPT2Tokenizer
from alt_plot_gen.interface.main import text_generation

#from alt_plot_gen.ml_logic.params import LOCAL_DATA_PATH


#url = 'https://alt-plot-gen.xxxxx/generate'
#params={'Title': selected_plot,
#        'Release Year': reduced_df['Release Year'][int(title_id)],
#        'Genre': new_genre
#        }
#generated_plot = requests.get(url, params=params).json

# ----------------------------------

#from PIL import Image
#image = Image.open(os.path.join(os.environ.get("LOCAL_DATA_PATH"), "snowwhite.png"))
#st.image(image, caption='Enter any caption here')


test_set_file = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "dataset_app.csv")

df = pd.read_csv(test_set_file)
reduced_df = df[['Title', 'Release Year', 'Genre', 'Plot', 'True_end_plot']]
available_genres = ('', 'Action', 'Comedy', 'Drama', 'Horror')

#nbr_trail_words =
model_path = os.path.join(os.environ.get("LOCAL_DATA_PATH"), "wreckgar-49.pt")
model = torch.load(model_path)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

st.set_page_config(layout="wide", page_title="Alternative Endings Generator", page_icon=os.path.join(os.environ.get("LOCAL_DATA_PATH"), "snowwhite.ico"))

def create_aggrid(head, df):
    st.subheader(head)
    new_df = pd.DataFrame(reduced_df[head])
    #builds a gridOptions dictionary using a GridOptionsBuilder instance (new_gd)
    new_gd = GridOptionsBuilder.from_dataframe(new_df)
    new_gd.configure_selection()
    gridoptions = new_gd.build()
    #uses the gridOptions dictionary to configure AgGrid behavior.
    grid_new_table = AgGrid(new_df,
                            height=250,
                            width='50%',
                            gridOptions=gridoptions,
                            enable_enterprise_modules=True,
                            fit_columns_on_grid_load=True,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                            reload_data=True,
                            header_checkbox_selection_filtered_only=True,
                            allow_unsafe_jscode=True
                            )
    return grid_new_table

def split_plot(plot, nbr_trail_words):
    start_plot = ' '.join(plot.split()[:-nbr_trail_words])
    true_end_plot = ' '.join(plot.split()[-nbr_trail_words:])
    return start_plot, true_end_plot


def header1(url):
    st.markdown(f'<p style="color:#6AA66A;font-family:verdana;font-size:54px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
def header2(url):
    st.markdown(f'<p style="font-family:verdana;font-size:18px;border-radius:20%;">{url}</p>', unsafe_allow_html=True)

header1('Alternative Endings Generator')
header2('What if you could play with alternative endings or adding up exclusive imaginary content?')

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
    start_plot_txt = st.text_area('', start_plot, height=plot_height*3)

    col4, col5, col6, col7, col8 = st.columns(5)
#    with col4:
#        new_genre = st.selectbox('Select new genre for alternative ending:', available_genres)
    with col8:
        generate_bt = st.button("Generate")

    col9, col10 = st.columns(2)
    with col9:
        st.subheader('Original plot ending')
        endplot_height = round(len(true_end_plot) * 7 / 750 / 3)
        true_end_plot_txt = st.text_area('', true_end_plot, height=endplot_height)
    if generate_bt:
        selected_plot = reduced_df['Plot'][int(title_id)]
        #Run the functions to generate the alternative ending
        with st.spinner("Please wait..."):
            test_generated_plot = text_generation(model, tokenizer, selected_plot)

        with col10:
            st.subheader('Generated plot ending')
            gen_end_plot = st.text_area('', test_generated_plot)



import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(os.path.join(os.environ.get("LOCAL_DATA_PATH"), "fairytalebg.jpg"))
