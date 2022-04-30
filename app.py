#####################################################################
# Deploy Machine Learning Models Using StreamLit                    #                          
# Last modified: 04/30/2022                                         #
#####################################################################

import sys, os, joblib
from rdkit import Chem
from run_biasnet import preprocess_smi
from rdkit.Chem import rdDepictor
from features import FeaturesGeneration
import requests
import streamlit as st

def main():
    
    st.title("BiasNet: A Model to Predict Ligand Bias Toward GPCR Signaling")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Biasnet ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    SMILES = st.text_input("SMILES", "Insert SMILES here")
    
    
    def get_features(processed_smiles):

        fg = FeaturesGeneration()
        features = fg.get_fingerprints(processed_smiles)

        return features

    def get_results(smiles):
        # Load models
        model = joblib.load('model/model_lecfp4.mlp')
        
        features = get_features(smiles)


        model_result = {}

        label_zero = model.predict_proba(features)[0][0].round(3)
        label_one = model.predict_proba(features)[0][1].round(3)

        if label_one >= 0.5:
            model_result['Prediction'] = 'B-Arrestin'
            model_result['Confidence'] = label_one

        else:
            model_result['Prediction'] = 'G-Protein'
            model_result['Confidence'] = label_zero
        return model_result

    if st.button('Predict'):
        Dictn = dict()
        # Preprocessing of the SMILES
        processed_smiles = preprocess_smi(SMILES)

        model_result = get_results(processed_smiles)
        Dictn[SMILES] = model_result
        result = Dictn[SMILES]['Prediction']
        st.success('The output is {}'.format(result))

if __name__ == '__main__':
    main()
