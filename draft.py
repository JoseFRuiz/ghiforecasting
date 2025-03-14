import pandas as pd

# Replace with your file ID
file_id = "1O-dOOvSbkTwUv1Qyh33RCzDcOh5GiL2y"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)
print(df.head())


file_id_locations = {
    "Jaisalmer": {2017: "1O-dOOvSbkTwUv1Qyh33RCzDcOh5GiL2y",
                  2018: "1JgIxhAck67nxXFAPrKHcX8Ql-w4wXXB_",
                  2019: "1ayaT36iSigV5V7DG-haWVM8NO-kCfTv3"},
}


