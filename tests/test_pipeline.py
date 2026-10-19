from pathlib import Path

def test_pipeline_files():
    assert Path("src/pipeline.py").exists()
    assert Path("src/data_ingestion.py").exists()
    assert Path("src/preprocessing.py").exists()
    assert Path("src/customer_segmentation.py").exists()
    assert Path("src/sentiment_analysis.py").exists()
    assert Path("src/churn_prediction.py").exists()
    assert Path("src/statistical_analysis.py").exists()
    assert Path("src/eda.py").exists()
    assert Path("src/visualization.py").exists()
