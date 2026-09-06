====================
House Price Prediction
====================

The capstone housing workflow is in
``notebooks/housing_price/housing_price_analysis.ipynb``. Configure the
input file and training parameters in
``notebooks/housing_price/conf/config.yml``.

The notebook uses ``ta_lib.custom.SignedLog1pTransformer`` for numeric feature
transformation. The transformer supports both ``transform`` and
``inverse_transform`` and is compatible with scikit-learn pipelines.
