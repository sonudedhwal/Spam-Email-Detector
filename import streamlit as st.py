import streamlit as st
import pandas as pd

st.set_page_config(page_title="Inventory Dashboard", page_icon="💍")
st.title("Product Variations Tracker")
st.write("Use the sidebar to filter the current inventory.")

# 1. Create a sample DataFrame (simulating a database export)
data = {
    "SKU": ["JW-001-GLD", "JW-001-SLV", "JW-002-DIA", "JW-003-RBY", "JW-003-EMR"],
    "Product": ["Classic Band", "Classic Band", "Halo Ring", "Drop Earrings", "Drop Earrings"],
    "Variation": ["18k Gold", "Sterling Silver", "Diamond/White Gold", "Ruby/Gold", "Emerald/Gold"],
    "Price ($)": [450, 120, 1250, 850, 900],
    "Stock Level": [12, 45, 3, 0, 8]
}
df = pd.DataFrame(data)

# 2. Add interactive filters in the sidebar
st.sidebar.header("Filter Inventory")

# A checkbox to toggle out-of-stock items
show_out_of_stock = st.sidebar.checkbox("Show Out of Stock Items", value=True)

# A slider to filter by maximum price
max_price = st.sidebar.slider(
    "Maximum Price", 
    min_value=0, 
    max_value=2000, 
    value=1500,
    step=50
)

# 3. Apply the filters to our pandas DataFrame
filtered_df = df[df["Price ($)"] <= max_price]

if not show_out_of_stock:
    # Keep only rows where Stock Level is strictly greater than 0
    filtered_df = filtered_df[filtered_df["Stock Level"] > 0]

# 4. Display the filtered data
st.subheader("Current Inventory Data")

# st.dataframe automatically makes the table sortable and scrollable
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# 5. Add dynamic metrics at the bottom
st.divider()
col1, col2 = st.columns(2)

# Calculate total value of the filtered stock
total_value = (filtered_df['Price ($)'] * filtered_df['Stock Level']).sum()

col1.metric("Variations Displayed", len(filtered_df))
col2.metric("Total Stock Value", f"${total_value:,}")
