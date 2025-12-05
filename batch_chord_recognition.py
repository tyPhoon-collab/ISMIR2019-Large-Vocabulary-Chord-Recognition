#!/usr/bin/env python
"""
バッチ和音推定実行スクリプト

生成されたWAVファイルに対して、バッチ処理で和音推定を実行し、
LAB形式のアノテーションを生成します。

高速化: モデルを事前ロードし、サブプロセス呼び出しを廃止しました。
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

# このディレクトリをパスに追加してインポート可能にする
sys.path.insert(0, str(Path(__file__).parent))

from chord_recognition import ChordRecognizer

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(
        description="生成音声に対してバッチ和音推定を実行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="入力音声ディレクトリ (生成されたWAVファイルを含む)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="出力ディレクトリ (推定LABファイルを保存)",
    )
    parser.add_argument(
        "--chord_dict",
        type=str,
        default="small",
        choices=["full", "ismir2017", "submission", "extended", "small"],
        help="和音辞書 (デフォルト: small)",
    )

    return parser.parse_args()


def find_wav_files(input_dir: Path) -> List[Path]:
    """入力ディレクトリからWAVファイルを探索

    Args:
        input_dir: 入力ディレクトリ

    Returns:
        見つかったWAVファイルのパスリスト
    """
    wav_files = sorted(input_dir.rglob("*.wav"))
    logger.info("見つかったWAVファイル: %d個" % len(wav_files))

    if len(wav_files) == 0:
        logger.warning("警告: %s内にWAVファイルが見つかりません" % input_dir)

    return wav_files


def estimate_chords_batch(
    recognizer: ChordRecognizer,
    wav_files: List[Path],
    output_dir: Path,
) -> Dict:
    """バッチで和音推定を実行（モデル常駐版）

    Args:
        recognizer: 事前ロード済みのChordRecognizerインスタンス
        wav_files: WAVファイルのパスリスト
        output_dir: 出力ディレクトリ

    Returns:
        実行統計 (成功数、失敗数など)
    """
    stats: Dict = {
        "total": len(wav_files),
        "success": 0,
        "failed": 0,
        "errors": [],
    }

    logger.info("\n=== 和音推定開始 (総ファイル数: %d) ===" % len(wav_files))

    for audio_path in tqdm(wav_files, desc="和音推定", unit="file"):
        # 出力パスの決定
        output_path = output_dir / (audio_path.stem + ".lab")

        try:
            # 和音推定を実行（モデルは再利用）
            recognizer.recognize_and_save(audio_path, output_path)
            stats["success"] += 1
        except Exception as e:
            tqdm.write(f"❌ エラー ({audio_path.name}): {e}")
            stats["failed"] += 1
            stats["errors"].append(
                {
                    "file": audio_path.name,
                    "output": output_path.name,
                    "error": str(e),
                }
            )

    return stats


def main() -> None:
    """メイン処理"""
    print("=== バッチ和音推定実行スクリプト ===\n")

    # コマンドライン引数のパース
    args = parse_args()

    # 入出力ディレクトリの確認
    if not args.input_dir.exists():
        raise FileNotFoundError("入力ディレクトリが見つかりません: %s" % args.input_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("出力ディレクトリ: %s" % args.output_dir)

    # WAVファイルの探索
    wav_files = find_wav_files(args.input_dir)

    if len(wav_files) == 0:
        logger.error("処理するWAVファイルが見つかりません")
        sys.exit(1)

    # 和音推定器を初期化（モデルを事前ロード）
    logger.info("モデルをロード中 (和音辞書: %s)..." % args.chord_dict)
    recognizer = ChordRecognizer(chord_dict_name=args.chord_dict, verbose=False)
    logger.info("モデルのロード完了")

    # 和音推定を実行
    stats = estimate_chords_batch(recognizer, wav_files, args.output_dir)

    # 結果を表示
    print("\n" + "=" * 50)
    print("=== 処理完了 ===")
    print("総ファイル数: %d" % stats["total"])
    print("✅ 成功: %d" % stats["success"])
    print("❌ 失敗: %d" % stats["failed"])

    if stats["errors"]:
        print("\n失敗ファイル:")
        for error in stats["errors"]:
            print("  - %s" % error["file"])

    # 結果をJSONで保存
    results_json_path = args.output_dir / "chord_estimation_stats.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("統計情報を保存: %s" % results_json_path)

    # 終了コード
    exit_code = 0 if stats["failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
