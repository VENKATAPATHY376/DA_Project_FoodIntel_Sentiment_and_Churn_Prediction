import streamlit as st
from pathlib import Path

st.set_page_config(page_title='Project README', layout='wide')
st.title('FoodIntel AI - Project README')

readme_path = Path('README.md')
if not readme_path.exists():
    readme_path = Path(__file__).parent.parent / 'README.md'

if readme_path.exists():
    content = readme_path.read_text(encoding='utf-8')
    st.markdown(content)
else:
    st.error('README.md not found.')
