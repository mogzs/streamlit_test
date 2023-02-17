mkdir -p ~/.streamlit/

echo "\"
[server]\n\
headless = true\n\
enablesCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml