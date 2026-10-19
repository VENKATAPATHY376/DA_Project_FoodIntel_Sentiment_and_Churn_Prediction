from pathlib import Path

def test_structure():
    assert Path("src/data_ingestion.py").exists()
    assert Path("src/preprocessing.py").exists()
    assert Path("src/sentiment_analysis.py").exists()
    assert Path("src/customer_segmentation.py").exists()
    assert Path("src/churn_prediction.py").exists()
    assert Path("src/eda.py").exists()
    assert Path("src/statistical_analysis.py").exists()
    assert Path("src/visualization.py").exists()
    assert Path("src/live_fetch.py").exists()
    assert Path("src/pipeline.py").exists()
    assert Path("pages/01_Executive_Overview.py").exists()
    assert Path("pages/06_Reports_and_Export.py").exists()
    assert Path("pages/08_Platform_Comparison.py").exists()
    assert Path("app.py").exists()
    assert Path("requirements.txt").exists()
    assert Path("README.md").exists()
