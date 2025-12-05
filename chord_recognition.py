import sys
from pathlib import Path
from typing import List, Union

import numpy as np
from chordnet_ismir_naive import ChordNet
from extractors.cqt import CQTV2
from extractors.xhmm_ismir import XHMMDecoder
from io_new.chordlab_io import ChordLabIO
from mir import DataEntry, io
from mir.nn.train import NetworkInterface
from settings import DEFAULT_HOP_LENGTH, DEFAULT_SR

MODEL_NAMES = [
    "joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best" % i for i in range(5)
]


class ChordRecognizer:
    """和音推定器（モデル常駐版）

    モデルを事前ロードし、複数ファイルの推定時にオーバーヘッドを削減します。
    """

    def __init__(self, chord_dict_name: str = "submission", verbose: bool = False):
        """初期化

        Args:
            chord_dict_name: 和音辞書名 (full, ismir2017, submission, extended, small)
            verbose: 詳細ログを出力するか
        """
        self.chord_dict_name = chord_dict_name
        self.verbose = verbose

        # HMMデコーダの初期化
        template_file = (
            Path(__file__).parent / "data" / ("%s_chord_list.txt" % chord_dict_name)
        )
        self.hmm = XHMMDecoder(template_file=str(template_file))

        # 5つのアンサンブルモデルを事前ロード
        self.models: List[NetworkInterface] = []
        for model_name in MODEL_NAMES:
            if self.verbose:
                print("Loading model: %s" % model_name)
            net = NetworkInterface(
                ChordNet(None).cuda(), model_name, load_checkpoint=False
            )
            self.models.append(net)

        if self.verbose:
            print("ChordRecognizer initialized with %d models" % len(self.models))

    def _prepare_entry(self, audio_path: Union[str, Path]) -> DataEntry:
        """音声ファイルからDataEntryを準備

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            CQT特徴量を含むDataEntry
        """
        entry = DataEntry()
        entry.prop.set("sr", DEFAULT_SR)
        entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
        entry.append_file(str(audio_path), io.MusicIO, "music")
        entry.append_extractor(CQTV2, "cqt")
        return entry

    def recognize(self, audio_path: Union[str, Path]) -> List:
        """単一ファイルの和音推定

        Args:
            audio_path: 音声ファイルのパス

        Returns:
            和音ラベルのリスト [[start, end, chord_name], ...]
        """
        entry = self._prepare_entry(audio_path)

        # アンサンブル推論
        probs = []
        for i, net in enumerate(self.models):
            if self.verbose:
                print("Inference: %s on %s" % (MODEL_NAMES[i], audio_path))
            probs.append(net.inference(entry.cqt))

        # 確率のアンサンブル平均
        probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]

        # HMMデコード
        chordlab = self.hmm.decode_to_chordlab(entry, probs, False)
        return chordlab

    def recognize_and_save(
        self, audio_path: Union[str, Path], lab_path: Union[str, Path]
    ) -> None:
        """和音推定を実行してLABファイルに保存

        Args:
            audio_path: 音声ファイルのパス
            lab_path: 出力LABファイルのパス
        """
        entry = self._prepare_entry(audio_path)

        # アンサンブル推論
        probs = []
        for i, net in enumerate(self.models):
            if self.verbose:
                print("Inference: %s on %s" % (MODEL_NAMES[i], audio_path))
            probs.append(net.inference(entry.cqt))

        # 確率のアンサンブル平均
        probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]

        # HMMデコード
        chordlab = self.hmm.decode_to_chordlab(entry, probs, False)

        # 保存
        entry.append_data(chordlab, ChordLabIO, "chord")
        entry.save("chord", str(lab_path))


def chord_recognition(audio_path, lab_path, chord_dict_name="submission"):
    """和音推定（後方互換性のための関数）

    注意: この関数は毎回モデルをロードするため、複数ファイル処理には
    ChordRecognizerクラスを使用してください。
    """
    hmm = XHMMDecoder(template_file="data/%s_chord_list.txt" % chord_dict_name)
    entry = DataEntry()
    entry.prop.set("sr", DEFAULT_SR)
    entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
    entry.append_file(audio_path, io.MusicIO, "music")
    entry.append_extractor(CQTV2, "cqt")
    probs = []
    for model_name in MODEL_NAMES:
        net = NetworkInterface(ChordNet(None).cuda(), model_name, load_checkpoint=False)
        print("Inference: %s on %s" % (model_name, audio_path))
        probs.append(net.inference(entry.cqt))
    probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]
    chordlab = hmm.decode_to_chordlab(entry, probs, False)
    entry.append_data(chordlab, ChordLabIO, "chord")
    entry.save("chord", lab_path)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        chord_recognition(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        chord_recognition(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(
            "Usage: chord_recognition.py path_to_audio_file path_to_output_file [chord_dict=submission]"
        )
        print(
            "\tChord dict can be one of the following: full, ismir2017, submission, extended"
        )
        exit(0)
