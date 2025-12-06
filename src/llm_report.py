import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
events_csv_path = os.path.join(BASE_DIR, "results", "traffic_events_core.csv")

load_dotenv()  # read .env
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Please create a .env file or env var.")

client = OpenAI(api_key=api_key)

def generate_report():
    df = pd.read_csv(events_csv_path)

    total_crossings = len(df)
    total_violations = (df["event_type"] == "RED_LIGHT_VIOLATION").sum()
    avg_speed = df["speed_kmh"].dropna().mean()
    max_speed = df["speed_kmh"].dropna().max()

    if len(df) > 0:
        duration_sec = df["time_sec"].max() - df["time_sec"].min()
    else:
        duration_sec = 0

    stats_text = f"""
Total crossings: {total_crossings}
Red-light violations: {total_violations}
Average speed (km/h): {avg_speed:.1f if not pd.isna(avg_speed) else 'no data'}
Maximum speed (km/h): {max_speed:.1f if not pd.isna(max_speed) else 'no data'}
Time span of logged events (seconds): {duration_sec:.1f}
"""

    violations_sample = df[df["event_type"] == "RED_LIGHT_VIOLATION"].head(10)
    if len(violations_sample) > 0:
        sample_table = violations_sample.to_markdown(index=False)
    else:
        sample_table = "No red-light violations were detected in this video."

    prompt = f"""
You are an AI assistant specialized in traffic analysis.

I have a traffic event log extracted from a video using object detection, tracking,
speed estimation and red-light violation detection.

Here are some summary statistics:

{stats_text}

Here is a small sample of red-light violations (if any):

{sample_table}

Please generate a clear, concise, and professional traffic analysis report in English.

Your report should include:
- A brief overview of the observed traffic density and flow
- An assessment of speed behavior and potential speeding risks
- A statement about red-light violations (if any)
- 2–3 practical safety or infrastructure improvement recommendations

Write the report in well-structured paragraphs.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in traffic analysis."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    report = response.choices[0].message.content
    print("\n========== TRAFFIC REPORT ==========\n")
    print(report)
    print("\n====================================\n")


if __name__ == "__main__":
    generate_r_
