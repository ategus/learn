import streamlit as st

def main():
    st.title("Personalized Greeting App")
    
    # Text input for user's name
    name = st.text_input("Enter your name:")
    
    if name:
        # Display personalized greeting
        st.write(f"Hello, {name}! Welcome to Streamlit.")

if __name__ == "__main__":
    main()

