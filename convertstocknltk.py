
import streamlit as st
import pandas as pd
import nltk
import re

# ==============================================================
# PAGE CONFIG
# ==============================================================

st.set_page_config(
    page_title="Inventory Dashboard",
    page_icon="💍",
    layout="wide"
)

# ==============================================================
# NLTK SETUP
# ==============================================================

@st.cache_resource
def setup_nltk():
    packages = ["punkt", "punkt_tab", "stopwords"]

    for pkg in packages:
        try:
            if pkg in ["punkt", "punkt_tab"]:
                nltk.data.find(f"tokenizers/{pkg}")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    return True


setup_nltk()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# ==============================================================
# TITLE
# ==============================================================

st.title("💍 Product Variations Tracker")
st.write(
    "Use the sidebar to filter the current inventory, "
    "or ask the chatbot below a question."
)


# ==============================================================
# SAMPLE INVENTORY DATA
# ==============================================================

data = {
    "SKU": [
        "JW-001-GLD",
        "JW-001-SLV",
        "JW-002-DIA",
        "JW-003-RBY",
        "JW-003-EMR"
    ],

    "Product": [
        "Classic Band",
        "Classic Band",
        "Halo Ring",
        "Drop Earrings",
        "Drop Earrings"
    ],

    "Variation": [
        "18k Gold",
        "Sterling Silver",
        "Diamond/White Gold",
        "Ruby/Gold",
        "Emerald/Gold"
    ],

    "Price ($)": [
        450,
        120,
        1250,
        850,
        900
    ],

    "Stock Level": [
        12,
        45,
        3,
        0,
        8
    ]
}

df = pd.DataFrame(data)


# ==============================================================
# SIDEBAR
# ==============================================================

st.sidebar.header("🔎 Filter Inventory")

show_out_of_stock = st.sidebar.checkbox(
    "Show Out of Stock Items",
    value=True
)

max_price = st.sidebar.slider(
    "Maximum Price",
    min_value=0,
    max_value=2000,
    value=1500,
    step=50
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    options=sorted(df["Product"].unique()),
    default=sorted(df["Product"].unique())
)


# ==============================================================
# APPLY FILTERS
# ==============================================================

filtered_df = df[
    (df["Price ($)"] <= max_price) &
    (df["Product"].isin(product_filter))
]

if not show_out_of_stock:
    filtered_df = filtered_df[
        filtered_df["Stock Level"] > 0
    ]


# ==============================================================
# INVENTORY TABLE
# ==============================================================

st.subheader("📦 Current Inventory Data")

if filtered_df.empty:
    st.warning("No inventory items match the selected filters.")
else:
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ==============================================================
# METRICS
# ==============================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

total_variations = len(filtered_df)

total_stock = filtered_df["Stock Level"].sum()

total_value = (
    filtered_df["Price ($)"] *
    filtered_df["Stock Level"]
).sum()

out_of_stock = (
    filtered_df["Stock Level"] == 0
).sum()

col1.metric(
    "Variations Displayed",
    total_variations
)

col2.metric(
    "Total Stock",
    total_stock
)

col3.metric(
    "Total Stock Value",
    f"${total_value:,.0f}"
)

col4.metric(
    "Out of Stock",
    out_of_stock
)


# ==============================================================
# NLTK TEXT PROCESSING
# ==============================================================

def clean_tokens(text: str):
    """
    Lowercase, tokenize, remove punctuation,
    remove stopwords and apply stemming.
    """

    tokens = word_tokenize(text.lower())

    tokens = [
        t for t in tokens
        if t.isalnum()
    ]

    return [
        STEMMER.stem(t)
        for t in tokens
        if t not in STOPWORDS
    ]


def raw_tokens(text: str):
    """
    Tokenize and stem without removing stopwords.
    Used for FAQ matching.
    """

    tokens = word_tokenize(text.lower())

    return [
        STEMMER.stem(t)
        for t in tokens
        if t.isalnum()
    ]


# ==============================================================
# FAQ INTENTS
# ==============================================================

FAQ_INTENTS = [

    {
        "name": "greeting",

        "patterns": [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening"
        ],

        "response":
            "Hello! 👋 I can answer questions about the "
            "inventory. Try asking about prices, stock, "
            "or out-of-stock products."
    },

    {
        "name": "farewell",

        "patterns": [
            "bye",
            "goodbye",
            "see you",
            "later",
            "exit",
            "quit"
        ],

        "response":
            "Goodbye! 👋 Come back anytime you have "
            "more inventory questions."
    },

    {
        "name": "thanks",

        "patterns": [
            "thanks",
            "thank you",
            "appreciate it",
            "thx"
        ],

        "response":
            "You're welcome! 😊 Happy to help."
    },

    {
        "name": "who_are_you",

        "patterns": [
            "who are you",
            "what are you",
            "your name",
            "are you a bot",
            "are you human"
        ],

        "response":
            "I'm an offline NLTK-based assistant built "
            "into this Streamlit inventory dashboard."
    },

    {
        "name": "capabilities",

        "patterns": [
            "what can you do",
            "help",
            "how do i use this",
            "what can i ask",
            "commands",
            "options",
            "instructions"
        ],

        "response":
            "I can help with:\n\n"
            "• Product prices\n"
            "• Stock levels\n"
            "• Out-of-stock products\n"
            "• Products under a price\n"
            "• Total stock value\n"
            "• Product information\n\n"
            "Example: 'What is the price of Halo Ring?'"
    },

    {
        "name": "about_app",

        "patterns": [
            "what is this app",
            "what is this dashboard",
            "what does this app do",
            "explain this dashboard",
            "purpose of this app"
        ],

        "response":
            "This is a Product Variations Tracker. "
            "It tracks jewelry SKUs, prices, variations "
            "and stock levels."
    },

    {
        "name": "how_are_you",

        "patterns": [
            "how are you",
            "how are you doing",
            "how's it going"
        ],

        "response":
            "I'm running smoothly and ready to help! 😊"
    },

    {
        "name": "creator",

        "patterns": [
            "who made you",
            "who built you",
            "who created you",
            "who developed this"
        ],

        "response":
            "I was built using Python, Streamlit, "
            "pandas and NLTK."
    }
]


# ==============================================================
# PREPARE FAQ PATTERNS
# ==============================================================

for intent in FAQ_INTENTS:

    intent["_pattern_tokens"] = [
        set(raw_tokens(pattern))
        for pattern in intent["patterns"]
    ]

    intent["_pattern_texts"] = [
        pattern.lower()
        for pattern in intent["patterns"]
    ]


# ==============================================================
# FAQ MATCHING
# ==============================================================

def match_faq(query: str):

    lowered = query.lower().strip()

    q_tokens = set(
        raw_tokens(query)
    )

    if not q_tokens:
        return None


    # Direct text matching
    for intent in FAQ_INTENTS:

        for pattern_text in intent["_pattern_texts"]:

            if (
                pattern_text in lowered
                or lowered in pattern_text
            ):
                return intent["response"]


    # Word overlap matching
    best_score = 0.0

    best_response = None

    for intent in FAQ_INTENTS:

        for pattern_tokens in intent["_pattern_tokens"]:

            if not pattern_tokens:
                continue

            overlap = q_tokens & pattern_tokens

            score = (
                len(overlap) /
                len(pattern_tokens)
            )

            if score > best_score:

                best_score = score

                best_response = intent["response"]


    if best_score >= 0.6:
        return best_response

    return None


# ==============================================================
# FIND MATCHING PRODUCTS
# ==============================================================

def find_matching_products(
    tokens,
    frame: pd.DataFrame
):

    if not tokens:
        return frame.iloc[0:0]


    def row_matches(row):

        row_text = (
            f"{row['Product']} "
            f"{row['Variation']} "
            f"{row['SKU']}"
        ).lower()

        row_tokens = set(
            clean_tokens(row_text)
        )

        return bool(
            set(tokens) & row_tokens
        )


    mask = frame.apply(
        row_matches,
        axis=1
    )

    return frame[mask]


# ==============================================================
# INVENTORY QUESTION ANSWER
# ==============================================================

def answer_inventory_query(
    query: str,
    frame: pd.DataFrame
):

    tokens = clean_tokens(query)

    lowered = query.lower()


    # ----------------------------------------------------------
    # OUT OF STOCK
    # ----------------------------------------------------------

    if (
        ("out" in lowered and "stock" in lowered)
        or "unavailable" in lowered
        or "sold out" in lowered
    ):

        oos = frame[
            frame["Stock Level"] == 0
        ]

        if oos.empty:

            return (
                "Nothing is currently out of stock "
                "in the filtered view."
            )

        items = ", ".join(
            f"{row['Product']} "
            f"({row['Variation']})"
            for _, row in oos.iterrows()
        )

        return f"Out of stock: {items}."


    # ----------------------------------------------------------
    # LOWEST STOCK
    # ----------------------------------------------------------

    if (
        "lowest stock" in lowered
        or "least stock" in lowered
        or "minimum stock" in lowered
    ):

        if frame.empty:
            return "There are no products in the current view."

        min_stock = frame["Stock Level"].min()

        matches = frame[
            frame["Stock Level"] == min_stock
        ]

        items = ", ".join(
            f"{row['Product']} "
            f"({row['Variation']}) - "
            f"{row['Stock Level']} units"
            for _, row in matches.iterrows()
        )

        return f"Lowest stock: {items}."


    # ----------------------------------------------------------
    # HIGHEST STOCK
    # ----------------------------------------------------------

    if (
        "highest stock" in lowered
        or "most stock" in lowered
        or "maximum stock" in lowered
    ):

        if frame.empty:
            return "There are no products in the current view."

        max_stock = frame["Stock Level"].max()

        matches = frame[
            frame["Stock Level"] == max_stock
        ]

        items = ", ".join(
            f"{row['Product']} "
            f"({row['Variation']}) - "
            f"{row['Stock Level']} units"
            for _, row in matches.iterrows()
        )

        return f"Highest stock: {items}."


    # ----------------------------------------------------------
    # PRICE UNDER / BELOW
    # ----------------------------------------------------------

    price_match = re.search(
        r"(under|below|less than|cheaper than)"
        r"\s*\$?(\d+)",
        lowered
    )

    if price_match:

        limit = int(
            price_match.group(2)
        )

        matches = frame[
            frame["Price ($)"] < limit
        ]

        if matches.empty:

            return (
                f"No items under ${limit} "
                "in the current view."
            )

        items = ", ".join(
            f"{row['Product']} "
            f"({row['Variation']}) - "
            f"${row['Price ($)']:,.0f}"
            for _, row in matches.iterrows()
        )

        return (
            f"Items under ${limit}: "
            f"{items}."
        )


    # ----------------------------------------------------------
    # TOTAL STOCK VALUE
    # ----------------------------------------------------------

    if (
        "total" in lowered
        and "value" in lowered
    ):

        val = (
            frame["Price ($)"] *
            frame["Stock Level"]
        ).sum()

        return (
            f"Total stock value in the current "
            f"view is ${val:,.0f}."
        )


    # ----------------------------------------------------------
    # TOTAL STOCK
    # ----------------------------------------------------------

    if (
        "total stock" in lowered
        or "total quantity" in lowered
        or "how many items" in lowered
    ):

        total = frame["Stock Level"].sum()

        return (
            f"Total stock in the current view "
            f"is {total} units."
        )


    # ----------------------------------------------------------
    # PRICE QUESTION
    # ----------------------------------------------------------

    if any(
        word in lowered
        for word in [
            "price",
            "cost",
            "how much"
        ]
    ):

        matches = find_matching_products(
            tokens,
            frame
        )

        if matches.empty:
            return None

        lines = []

        for _, row in matches.iterrows():

            lines.append(
                f"{row['Product']} "
                f"({row['Variation']}): "
                f"${row['Price ($)']:,.0f}"
            )

        return " | ".join(lines)


    # ----------------------------------------------------------
    # STOCK QUESTION
    # ----------------------------------------------------------

    if any(
        word in lowered
        for word in [
            "how many",
            "quantity",
            "stock level",
            "left",
            "in stock"
        ]
    ):

        matches = find_matching_products(
            tokens,
            frame
        )

        if matches.empty:
            return None

        lines = []

        for _, row in matches.iterrows():

            lines.append(
                f"{row['Product']} "
                f"({row['Variation']}): "
                f"{row['Stock Level']} "
                f"in stock"
            )

        return " | ".join(lines)


    # ----------------------------------------------------------
    # GENERAL PRODUCT SEARCH
    # ----------------------------------------------------------

    matches = find_matching_products(
        tokens,
        frame
    )

    if not matches.empty:

        lines = []

        for _, row in matches.iterrows():

            lines.append(
                f"{row['Product']} "
                f"({row['Variation']}) - "
                f"${row['Price ($)']:,.0f}, "
                f"{row['Stock Level']} in stock"
            )

        return " | ".join(lines)


    return None


# ==============================================================
# COMBINED CHATBOT ROUTER
# ==============================================================

def answer_query(
    query: str,
    frame: pd.DataFrame
):

    # First inventory
    inventory_answer = answer_inventory_query(
        query,
        frame
    )

    if inventory_answer:
        return inventory_answer


    # Then FAQ
    faq_answer = match_faq(query)

    if faq_answer:
        return faq_answer


    # Fallback
    return (
        "I don't have an answer for that yet. "
        "I can help with inventory questions such as "
        "stock, price, out-of-stock products, totals, "
        "or type 'help' to see what I support."
    )


# ==============================================================
# CHATBOT UI
# ==============================================================

st.divider()

st.subheader("💬 Ask the Assistant")

st.caption(
    "Try questions such as: "
    "'What is the price of Halo Ring?', "
    "'What is out of stock?', "
    "'Which product has the lowest stock?', "
    "'Show items under $500', or 'help'."
)


# ==============================================================
# SUGGESTED QUESTIONS
# ==============================================================

st.write("### 💡 Suggested Questions")

qcol1, qcol2, qcol3, qcol4 = st.columns(4)


if qcol1.button("💰 Halo Ring Price"):
    st.session_state["suggested_question"] = (
        "What is the price of Halo Ring?"
    )


if qcol2.button("📦 Out of Stock"):
    st.session_state["suggested_question"] = (
        "What is out of stock?"
    )


if qcol3.button("📉 Lowest Stock"):
    st.session_state["suggested_question"] = (
        "Which product has the lowest stock?"
    )


if qcol4.button("💵 Items Under $500"):
    st.session_state["suggested_question"] = (
        "Show items under $500"
    )


# ==============================================================
# CHAT INPUT
# ==============================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "suggested_question" not in st.session_state:

    st.session_state.suggested_question = ""


user_query = st.text_input(
    "Your question:",
    value=st.session_state.suggested_question,
    key="chat_input"
)


# ==============================================================
# ASK BUTTON
# ==============================================================

if st.button(
    "🔍 Ask Assistant",
    use_container_width=True
):

    if user_query.strip():

        response = answer_query(
            user_query,
            filtered_df
        )

        st.session_state.chat_history.append(
            (
                user_query,
                response
            )
        )

        st.session_state.suggested_question = ""


# ==============================================================
# CHAT HISTORY
# ==============================================================

if st.session_state.chat_history:

    st.write("### 🗨️ Conversation")

    for q, a in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"**👤 You:** {q}"
        )

        st.markdown(
            f"**🤖 Assistant:** {a}"
        )

        st.divider()


# ==============================================================
# CLEAR CHAT
# ==============================================================

if st.session_state.chat_history:

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ==============================================================
# FOOTER
# ==============================================================

st.caption(
    "💍 Product Variations Tracker | "
    "Built with Streamlit, Pandas and NLTK"
)

