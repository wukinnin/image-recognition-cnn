# Dogs and Cats image recognition using CNN

**Requirements:**
- Python
- numpy
- opencv-python
- tensorflow
- keras

**Training File Structure**

```
/train
  |
  ----- /train/cat
        |
        ---------- cat0.jpg
        |
        ---------- [...]
  |
  ----- /train/dog
        |
        ---------  dog0.jpg
        |
        ---------  [...]
```

---

**Training Database:**

Provided by Kaggle via [Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

**Usage**

1. Train by running `python ./ccn-train.py`
2. Expected output is `animals_ccn.keras` when done
3. Test by editing path to test image on `ccn-test.py`. Edit: `TEST_IMG = '[link to image file]'`
4. Run `python ./ccn-test.py`
