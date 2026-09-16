import argparse
import pandas as pd
from .weak_label import normalize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=30000)
    args = ap.parse_args()

    cols = ["tweet_id","author_id","inbound","created_at","text","response_tweet_id","in_response_to_tweet_id"]
    chunks = []
    for chunk in pd.read_csv(args.input, usecols=cols, chunksize=100000):
        # AppleSupport authored replies can be used as resolution evidence;
        # customer messages are inbound and linked to an AppleSupport response.
        chunk["text"] = chunk["text"].fillna("").astype(str)
        mask_customer = chunk["inbound"].eq(True) | chunk["inbound"].astype(str).str.lower().eq("true")
        mask_link = chunk["response_tweet_id"].notna() | chunk["in_response_to_tweet_id"].notna()
        c = chunk[mask_customer & mask_link].copy()
        c["customer_text"] = c["text"].map(normalize)
        c = c[c["customer_text"].str.len() >= 8]
        chunks.append(c)
        if sum(len(x) for x in chunks) >= args.max_rows:
            break

    out = pd.concat(chunks, ignore_index=True).head(args.max_rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out}")

if __name__ == "__main__":
    main()
