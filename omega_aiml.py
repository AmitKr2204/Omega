#install alt_profanity_check module first
import streamlit as st
from profanity_check import predict
if "counter" not in st.session_state:
    st.session_state.counter = 0
st.title("Social Media")
st.image("landscape.jpeg")
if st.session_state.counter<3:
    t=st.text_input("Comment")
    if st.button("Submit"):
        if predict([t]):
            st.session_state.counter+=1
            st.rerun()
        else:
            st.success("Comment Added")
else:
    st.error("You can't comment any more")

