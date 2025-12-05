import os
import subprocess
import tarfile
import tempfile

from chord_recognition import chord_recognition


def process_tar_mp3s(tar_path, out_dir, chord_dict_name="submission"):
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if m.isfile() and m.name.endswith("other.mp3"):
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                    f.write(tar.extractfile(m).read())
                    f.flush()
                    out_path = os.path.join(
                        out_dir, os.path.basename(m.name).replace(".mp3", ".lab")
                    )
                    chord_recognition(f.name, out_path, chord_dict_name)


def process_tar_mixed_mp3s(tar_path, out_dir, chord_dict_name="submission"):
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        files = {
            m.name: m
            for m in tar
            if m.isfile()
            and (m.name.endswith("other.mp3") or m.name.endswith("bass.mp3"))
        }
        track_ids = set()
        for name in files:
            if name.endswith("other.mp3"):
                track_ids.add(name[:-10])

        print(f"Found {len(track_ids)} tracks with 'other.mp3' files.")
        for tid in track_ids:
            other_name = tid + ".other.mp3"
            bass_name = tid + ".bass.mp3"
            if other_name in files and bass_name in files:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f_other:
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=True
                    ) as f_bass:
                        with tempfile.NamedTemporaryFile(
                            suffix=".mp3", delete=True
                        ) as f_mix:
                            f_other.write(tar.extractfile(files[other_name]).read())
                            f_other.flush()
                            f_bass.write(tar.extractfile(files[bass_name]).read())
                            f_bass.flush()
                            # ffmpegでミックス（mp3出力）
                            cmd = [
                                "ffmpeg",
                                "-y",
                                "-i",
                                f_other.name,
                                "-i",
                                f_bass.name,
                                "-filter_complex",
                                "[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0[mix]",
                                "-map",
                                "[mix]",
                                f_mix.name,
                            ]
                            subprocess.run(cmd, check=True)
                            out_path = os.path.join(
                                out_dir, os.path.basename(tid) + ".lab"
                            )
                            chord_recognition(f_mix.name, out_path, chord_dict_name)


if __name__ == "__main__":
    # process_tar_mixed_mp3s(
    #     "/data/musdb18hq/train.tar",
    #     "/data/musdb_simple_train",
    #     "small",
    # )
    # process_tar_mixed_mp3s(
    #     "/data/musdb18hq/test_ordered.tar",
    #     "/data/musdb_simple_test",
    #     "small",
    # )

    # process_tar_mixed_mp3s(
    #     "/data/musdb18hq/train.tar",
    #     "/data/musdb_large_train",
    #     "submission",
    # )
    # process_tar_mixed_mp3s(
    #     "/data/musdb18hq/test_ordered.tar",
    #     "/data/musdb_large_test",
    #     "submission",
    # )

    process_tar_mixed_mp3s(
        "/data/musdb18hq/train.tar",
        "/data/musdb_train",
        "ismir2017",
    )
    process_tar_mixed_mp3s(
        "/data/musdb18hq/test_ordered.tar",
        "/data/musdb_test",
        "ismir2017",
    )
