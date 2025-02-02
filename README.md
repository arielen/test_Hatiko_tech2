# **📌 Collection of data on prices of goods from suppliers**

This script (`collect_data.py`) is designed to **process and match supplier price lists with products in the store**. It extracts data, normalizes it, utilizes `SentenceTransformer` to find the most similar products, and then generates **an Excel table**.

## Table of Contents
| Section | Description |
|---------|------------|
| [Functionality](#-functionality) | Detailed breakdown of the script's capabilities and processing steps |
| [Technologies Used](#️-technologies-used) | List of technologies utilized in the project |
| [Installation and Execution](#-installation-and-execution) | Step-by-step guide on setting up and running the script |
| [How Does `SentenceTransformer` Work?](#-how-does-sentencetransformer-work) | Explanation of how `SentenceTransformer` is used for matching products based on semantic similarity |
| [Project Structure](#-project-structure) | Folder structure explanation |
| [Usage Presentation](#-usage-presentation) | Presentation of a worked script |
| [Contacts](#-contacts) | Developer contact information |

## **🚀 Functionality**
1. **Data Loading**  
   - Loads supplier price lists from `Прайсы с телеграма 28.01.xlsx`.
   - Loads store products from `Товары магазина.xlsx`.

2. **Data Cleaning and Normalization**  
   - Extracts **color** (`get_color()`).
   - Extracts **RAM and ROM** (`get_ram()`, `get_rom()`).
   - Extracts **prices** (`get_price()`).
   - Identifies **manufacturer** (`get_manufacturer()`).

3. **Matching Supplier Products with Store Products**  
   - Encodes product names using `SentenceTransformer`.
   - Computes **cosine similarity** between products.
   - Filters **matches with accuracy above 90%**.

4. **Generating an Excel Table (`output_prices.xlsx`)**  
   - Groups products by name.
   - Lists **prices and suppliers** for each product.

---

## ⚙️ **Technologies Used**
![Python 3.13](https://img.shields.io/badge/Python-3.13-000000?style=for-the-badge&labelColor=fafbfc&logo=python&logoColor=306998&color=2b3137) ![Pandas](https://img.shields.io/badge/Pandas-2b3137?style=for-the-badge&logo=pandas) ![NumPy](https://img.shields.io/badge/NumPy-2b3137?style=for-the-badge&logo=numpy) ![SentenceTransformers](https://img.shields.io/badge/Sentence_Transformers-Custom_Model_v3-000000?style=for-the-badge&labelColor=fafbfc&logo=pytorch&logoColor=306998&color=2b3137) ![Openpyxl](https://img.shields.io/badge/Openpyxl-2b3137?style=for-the-badge&logo=googlesheets)

---

## **📥 Installation and Execution**
### 🔹 **1. Clone the Repository**
```bash
git clone https://github.com/arielen/test_Hatiko_tech.git
cd test_Hatiko_tech
```

### 🔹 **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### 🔹 **3. Prepare the Files**
Place **`Прайсы с телеграма 28.01.xlsx`** and **`Товары магазина.xlsx`** in the **same directory**.

### 🔹 **4. Run the Script**
```bash
python collect_data.py
```

### 🔹 **5 Get the Result**
After execution, **the result will be saved as `output_prices.xlsx`**.

---

## **📌 How Does `SentenceTransformer` Work?**
This script uses the `fine_tuned_mpnet_v3` model for **finding similarity between products**:
1. Encodes product names into vector representations.
2. Compares supplier products with store products.
3. Selects products with **similarity above 90%**.

✅ **Example:**
```
🔍 Query: iPhone 15 Pro Max 512GB
✅ Found: Apple iPhone 15 Pro 512GB (Similarity: 0.92)
```

---

## 📜 **Project Structure**
```
📂 data-parsing/
├── 📜 collect_data.ipynb          # Jupyter notebook for data collection
├── 📝 collect_data.py             # Python script for data processing
├── 📂 fine_tuned_mpnet_v3         # Fine-tuned SentenceTransformer model
├── 📜 fine_tune.csv               # CSV file for fine-tuning the model
├── 📜 learning.ipynb              # Jupyter notebook for model training
├── 📝 learning.py                 # Python script for model fine-tuning
├── 📊 output_prices.xlsx          # Generated Excel file with matched prices
├── 📜 README.md                   # Project documentation
├── 📜 requirements.txt            # Dependencies file
├── 📜 Прайсы с телеграма.xlsx     # Supplier price list
└── 📜 Товары магазина.xlsx        # Store product list
```

---

## 🎥 **Usage presentation**
![image](https://github.com/user-attachments/assets/3d87470d-def5-4ebe-a605-b44517d80d37)

---

## **🎯 Summary**
This script **automates the collection and processing of supplier price lists**, as well as **generates a structured Excel table** with matched products.

## 📞 **Contacts**
💻 **Developer:** [arielen](https://github.com/arielen)  
📧 **Email:** pavlov_zv@mail.ru  
📧 **TG:** [1 0](https://t.me/touch_con)  
