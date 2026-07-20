"""
Script to generate realistic sample CSV datasets for testing.
Run: python data/samples/generate_samples.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent

REGIONS = ["North", "South", "East", "West", "Central"]
PRODUCTS = ["Widget Pro", "Gadget Plus", "Device Max", "Tool Ultra", "System Basic",
            "Module Advanced", "Kit Premium", "Pack Standard", "Unit Economy", "Bundle Elite"]
CUSTOMERS = [f"Customer_{i:03d}" for i in range(1, 51)]
SALES_REPS = [f"Rep_{name}" for name in ["Alice", "Bob", "Carol", "David", "Emma", "Frank"]]

def generate_sales_data(n=500):
    base_date = datetime(2023, 1, 1)
    data = []
    for i in range(n):
        order_date = base_date + timedelta(days=random.randint(0, 364))
        product = random.choice(PRODUCTS)
        region = random.choice(REGIONS)
        customer = random.choice(CUSTOMERS)
        rep = random.choice(SALES_REPS)
        qty = random.randint(1, 50)
        unit_price = round(random.uniform(10, 500), 2)
        discount = random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20])
        revenue = round(qty * unit_price * (1 - discount), 2)
        cost = round(revenue * random.uniform(0.4, 0.7), 2)
        profit = round(revenue - cost, 2)
        # Inject some anomalies
        if random.random() < 0.03:
            revenue *= random.uniform(5, 15)
            revenue = round(revenue, 2)
        data.append({
            "order_id": f"ORD-{i+1:05d}",
            "order_date": order_date.strftime("%Y-%m-%d"),
            "customer_id": customer,
            "product": product,
            "region": region,
            "sales_rep": rep,
            "quantity": qty,
            "unit_price": unit_price,
            "discount": discount,
            "revenue": revenue,
            "cost": cost,
            "profit": profit,
        })
    df = pd.DataFrame(data)
    # Add some missing values
    df.loc[df.sample(frac=0.02).index, "discount"] = None
    df.loc[df.sample(frac=0.01).index, "region"] = None
    return df

def generate_customer_data(n=50):
    data = []
    segments = ["Enterprise", "Mid-Market", "SMB", "Startup"]
    industries = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Education"]
    for i in range(1, n+1):
        data.append({
            "customer_id": f"Customer_{i:03d}",
            "company_name": f"Company {i}",
            "segment": random.choice(segments),
            "industry": random.choice(industries),
            "country": random.choice(["USA", "UK", "Germany", "France", "Canada", "Australia"]),
            "annual_contract_value": round(random.uniform(5000, 500000), 2),
            "lifetime_value": round(random.uniform(10000, 2000000), 2),
            "satisfaction_score": round(random.uniform(1, 10), 1),
            "churn_risk": round(random.uniform(0, 1), 2),
            "account_age_days": random.randint(30, 1825),
            "support_tickets": random.randint(0, 50),
        })
    df = pd.DataFrame(data)
    df.loc[df.sample(frac=0.04).index, "satisfaction_score"] = None
    return df

def generate_product_data():
    data = []
    categories = {"Widget Pro": "Widgets", "Gadget Plus": "Gadgets", "Device Max": "Devices",
                  "Tool Ultra": "Tools", "System Basic": "Systems", "Module Advanced": "Modules",
                  "Kit Premium": "Kits", "Pack Standard": "Packs", "Unit Economy": "Units",
                  "Bundle Elite": "Bundles"}
    for product, category in categories.items():
        data.append({
            "product": product,
            "category": category,
            "sku": f"SKU-{abs(hash(product)) % 10000:04d}",
            "list_price": round(random.uniform(20, 600), 2),
            "cost_price": round(random.uniform(5, 200), 2),
            "weight_kg": round(random.uniform(0.1, 10), 2),
            "stock_units": random.randint(0, 1000),
            "reorder_point": random.randint(10, 100),
            "lead_time_days": random.randint(1, 30),
            "rating": round(random.uniform(1, 5), 1),
            "reviews_count": random.randint(0, 500),
        })
    return pd.DataFrame(data)

if __name__ == "__main__":
    sales_df = generate_sales_data(500)
    sales_df.to_csv(OUTPUT_DIR / "sales_data.csv", index=False)
    print(f"Generated sales_data.csv: {len(sales_df)} rows")

    customer_df = generate_customer_data(50)
    customer_df.to_csv(OUTPUT_DIR / "customer_data.csv", index=False)
    print(f"Generated customer_data.csv: {len(customer_df)} rows")

    product_df = generate_product_data()
    product_df.to_csv(OUTPUT_DIR / "product_data.csv", index=False)
    print(f"Generated product_data.csv: {len(product_df)} rows")
    print("Done.")
