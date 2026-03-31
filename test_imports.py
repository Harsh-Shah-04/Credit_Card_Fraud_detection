"""
Test script to verify all required dependencies are installed correctly
Run this to diagnose import issues before deploying
"""

import sys

def test_imports():
    """Test all required imports"""
    
    print("Testing imports for Credit Card Fraud Detection App...")
    print("=" * 60)
    
    modules_to_test = [
        ("streamlit", "st"),
        ("pandas", "pd"),
        ("numpy", "np"),
        ("joblib", None),
        ("tensorflow", "tf"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("sklearn", "sklearn"),
        ("keras", None),
    ]
    
    failed = []
    
    for module_name, alias in modules_to_test:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"✅ {module_name:30} - OK")
        except Exception as e:
            print(f"❌ {module_name:30} - FAILED: {str(e)}")
            failed.append(module_name)
    
    print("=" * 60)
    
    if failed:
        print(f"\n❌ {len(failed)} module(s) failed to import:")
        for module in failed:
            print(f"   - {module}")
        print("\n📦 Please install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All imports successful!")
        print("\n🚀 You can now run:")
        print("   streamlit run streamlit_app.py")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
