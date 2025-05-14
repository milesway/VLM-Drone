# Genesis Documentation (English Version)

1. Create a clean env using python >= 3.9, install Sphinx and other dependencies

```bash
pip install genesis-world  # Requires Python >= 3.9;
```

2. Build the documentation and watch the change lively

```bash
# In genesis-doc/
rm -rf build/; make html; sphinx-autobuild ./source ./build/html
```
