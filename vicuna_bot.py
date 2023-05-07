

from transformers import LlamaTokenizer, LlamaForCausalLM

vicuna_7b_model = 'lmsys/vicuna-7b-delta-v1.1'
vicuna_13b_model = "lmsys/vicuna-13b-delta-v1.1"
chavinlo_alpaca_native = 'chavinlo/alpaca-native'


tokenizer = LlamaTokenizer.from_pretrained(chavinlo_alpaca_native)
model = LlamaForCausalLM.from_pretrained(chavinlo_alpaca_native)


from langchain import ConversationChain, PromptTemplate
prompt_template = PromptTemplate(
    input_prefix="User: ",
    input_suffix="\nBot: ",
    output_prefix="Bot: ",
    output_suffix="\nUser: "
)
chain = ConversationChain(model, tokenizer, prompt_template)

import streamlit as st
import PyPDF2

st.title("Chat with Alpaca")
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
if pdf_file:
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extractText()
    user_input = st.text_input("You:")
    if user_input:
        response = chain.generate_response(user_input, text)
        st.write(f"Bot: {response}")
        st.write(f"Conversation history:\n{chain.conversation_history}")