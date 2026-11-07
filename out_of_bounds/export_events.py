
from pathlib import Path

from kloppy import metrica

from games import asset_paths
from out_of_bounds.pipeline import detect_out_of_bounds


HALF_RANGES = [
    (11185, 90651),
    (110808, 190197),
]


def main():
    paths = asset_paths()

    dataset = metrica.load_tracking_epts(
        meta_data=paths["atd_meta"],
        raw_data=paths["atd_raw"]
    )

    tracking = dataset.to_df()
    tracking["video_frame"] = tracking["frame_id"] - 1

    events = detect_out_of_bounds(
        tracking,
        half_ranges=HALF_RANGES
    )

    output_dir = Path("out_of_bounds/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "events.csv"

    events.to_csv(output_path, index=False)

    print("Exported:", output_path)
    print("Events:", len(events))
    print(events["exit_reliability"].value_counts())


if __name__ == "__main__":
    main()
