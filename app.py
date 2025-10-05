import streamlit as st
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import matplotlib.pyplot as plt

st.set_page_config(page_title="PPB Predictor", page_icon="💊", layout="wide")

st.title("💊 Plasma Protein Binding Predictor")
st.markdown("### QSAR Model for Drug Development")

@st.cache_resource
def load_model():
    with open('ppb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    desc_dict = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'FractionCsp3': Descriptors.FractionCSP3(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'MolMR': Descriptors.MolMR(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'Chi0v': Descriptors.Chi0v(mol),
        'Chi1v': Descriptors.Chi1v(mol),
        'HallKierAlpha': Descriptors.HallKierAlpha(mol),
        'Kappa1': Descriptors.Kappa1(mol),
        'Kappa2': Descriptors.Kappa2(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
        'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
        'MinPartialCharge': Descriptors.MinPartialCharge(mol)
    }
    return desc_dict, mol

model, scaler, feature_names = load_model()

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model Info"])

with tab1:
    st.markdown("## Single Compound Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input("Enter SMILES:", value="CC(=O)Oc1ccccc1C(O)=O")
        
        if st.button("🔮 Predict PPB", type="primary"):
            desc_dict, mol = calculate_descriptors(smiles_input)
            
            if desc_dict is None:
                st.error("❌ Invalid SMILES!")
            else:
                desc_df = pd.DataFrame([desc_dict])
                for feat in feature_names:
                    if feat not in desc_df.columns:
                        desc_df[feat] = 0
                desc_df = desc_df[feature_names]
                
                desc_scaled = scaler.transform(desc_df)
                ppb_pred = model.predict(desc_scaled)[0]
                
                st.success("✅ Prediction Complete!")
                st.markdown(f"## Predicted PPB: **{ppb_pred:.2f}%**")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Molecular Weight", f"{desc_dict['MolWt']:.1f} Da")
                col_b.metric("LogP", f"{desc_dict['LogP']:.2f}")
                col_c.metric("TPSA", f"{desc_dict['TPSA']:.1f} Ų")
    
    with col2:
        st.markdown("### Molecule Structure")
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img)

with tab2:
    st.markdown("## Model Information")
    st.info("""
    **Training Data:** 1,614 compounds from TDC AstraZeneca
    
    **Performance:**
    - Test R² = 0.84
    - Test MAE = 3.8%
    
    **Features:** 22 molecular descriptors
    """)

st.markdown("---")
st.markdown("*For research purposes only*")
