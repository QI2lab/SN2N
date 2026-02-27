# -*- coding: utf-8 -*-

import itertools
import os

import numpy as np
import tifffile
import torch
import tqdm

from SN2N.utils import TOTENSOR_, normalize


class Predictor2D:
    def __init__(self, img_path, model_path, infer_mode):
        """
        Self-inspired Noise2Noise

        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to inference
        model_path:
            Path of model for inference
        infer_mode:
            Prediction Mode
            0: Predict the results of all models generated during training
            under the default "models" directory on the img_path.
            1: Predict the results of the models provided by the user under
            the given model_path on the Img_path provided by the user.

        """
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, "datasets")
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.model_save_path = os.path.join(self.parent_dir, "models")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.save_path = os.path.join(self.parent_dir, "predictions")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_path = model_path
        self.infer_mode = infer_mode

    def execute(self):
        img_path = self.img_path
        infer_mode = self.infer_mode
        if infer_mode == 0:
            model_path = self.model_save_path
        else:
            model_path = self.model_path
        save_path = self.save_path
        print(
            "The path for the raw images used for training is located under:\n%s"
            % (img_path)
        )
        print("Models is being saved under:\n%s" % (model_path))
        print("Predictions is being saved under:\n%s" % (save_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for mroot, mdirs, mfiles in os.walk(model_path):
            for jj, model_Ufile in enumerate(mfiles):
                print("=====Model: %d=====" % (jj + 1))
                m_path = os.path.join(mroot, model_Ufile)
                model = torch.load(m_path, map_location=device, weights_only=False)
                model = model.to(device)
                with torch.no_grad():
                    for root, dirs, files in os.walk(img_path):
                        for j, Ufile in enumerate(files):
                            imgpath = os.path.join(root, Ufile)
                            image_data = tifffile.imread(imgpath)
                            try:
                                [t, x, y] = image_data.shape
                                test_pred_np = np.zeros((t, x, y))
                                for taxial in range(t):
                                    datatensor = TOTENSOR_(
                                        normalize(image_data[taxial, :, :])
                                    )
                                    test_pred = model(datatensor.to(device))
                                    test_pred = test_pred.to(torch.device("cpu"))
                                    test_pred_np[taxial, :, :] = 255 * normalize(
                                        test_pred.numpy()
                                    )
                                    os.makedirs(save_path, exist_ok=True)
                                tifffile.imwrite(
                                    "%s/%s_%s.tif" % (save_path, Ufile, model_Ufile),
                                    test_pred_np.astype("uint8"),
                                )
                                print("Frame: %d" % (j + 1))
                            except ValueError:
                                datatensor = TOTENSOR_(normalize(image_data))
                                test_pred = model(datatensor.to(device))
                                test_pred = test_pred.to(torch.device("cpu"))
                                test_pred_np = 255 * normalize(test_pred.numpy())
                                os.makedirs(save_path, exist_ok=True)
                                tifffile.imwrite(
                                    "%s/%s_%s.tif" % (save_path, Ufile, model_Ufile),
                                    test_pred_np.astype("uint8"),
                                )
                                print("Frame: %d" % (j + 1))
            return


class Predictor3D:
    def __init__(self, img_path, model_path, infer_mode, overlap_shape="2,256,256"):
        """
        Self-inspired Noise2Noise

        -----Parameters------
        =====Important==========
        img_path:
            Path of raw images to inference
        model_path:
            Path of model for inference
        infer_mode:
            Prediction Mode
            0: Predict the results of all models generated during training
            under the default "models" directory on the img_path.
            1: Predict the results of the models provided by the user under
            the given model_path on the Img_path provided by the user.
        overlap_shape:
            Overlap shape in 3D stitching prediction.
            {default: '2, 256, 256'}
        """
        self.img_path = img_path
        self.parent_dir = os.path.dirname(img_path)
        self.dataset_path = os.path.join(self.parent_dir, "datasets")
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.model_save_path = os.path.join(self.parent_dir, "models")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.save_path = os.path.join(self.parent_dir, "predictions")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_path = model_path
        self.infer_mode = infer_mode
        self.overlap_shape = tuple(map(int, overlap_shape.split(",")))

    @staticmethod
    def _axis_starts(axis_len, tile_len, step_len):
        """Compute tile starts and always anchor the final tile at the end."""
        if axis_len <= tile_len:
            return [0]
        starts = list(range(0, axis_len - tile_len + 1, step_len))
        last_start = axis_len - tile_len
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _normalize_raw_volume(image):
        """Normalize raw input to [0, 1] using dtype-aware scaling."""
        if np.issubdtype(image.dtype, np.integer):
            denom = float(np.iinfo(image.dtype).max)
            if denom <= 0:
                return image.astype(np.float32)
            return image.astype(np.float32) / denom

        image = image.astype(np.float32, copy=False)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = float(np.max(image)) if image.size else 1.0
        if max_val > 1.0:
            image = image / max_val
        return image

    def execute(self, verbose=False):
        """Run tiled 3D inference and blend overlaps to reduce seam artifacts."""
        img_path = self.img_path
        infer_mode = self.infer_mode
        model_path = self.model_save_path if infer_mode == 0 else self.model_path
        save_path = self.save_path
        overlap_shape = self.overlap_shape

        print("Raw image path :", img_path)
        print("Model path     :", model_path)
        print("Predictions save to :", save_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for mroot, _mdirs, mfiles in os.walk(model_path):
            for jj, model_file in enumerate(sorted(mfiles)):
                print("===== Model %d : %s =====" % (jj + 1, model_file))
                m_path = os.path.join(mroot, model_file)
                model = torch.load(m_path, map_location=device, weights_only=False).to(
                    device
                )
                model.eval()

                with torch.no_grad():
                    for root, _dirs, files in os.walk(img_path):
                        for j, img_file in enumerate(sorted(files)):
                            if not img_file.lower().endswith((".tif", ".tiff")):
                                continue
                            img_path_full = os.path.join(root, img_file)
                            image = tifffile.imread(img_path_full).squeeze()
                            if image.ndim != 3:
                                raise ValueError("3D model expects 3D input.")
                            image = self._normalize_raw_volume(image)

                            model_in_shape = (16, 640, 640)
                            model_out_shape = model_in_shape
                            image_dim = 3
                            num_in_ch = 1
                            batch_size = 1

                            if overlap_shape is None:
                                overlap_shape = (2, 32, 32)
                            elif len(overlap_shape) != image_dim:
                                raise ValueError(
                                    f"Overlap shape must be {image_dim}D; got {overlap_shape}"
                                )
                            if any(o <= 0 for o in overlap_shape):
                                raise ValueError(
                                    f"Overlap shape must be positive; got {overlap_shape}"
                                )
                            step_shape = tuple(
                                m - o for m, o in zip(model_in_shape, overlap_shape)
                            )
                            if any(s <= 0 for s in step_shape):
                                raise ValueError(
                                    "Invalid overlap: each overlap must be smaller than "
                                    f"tile size. tile={model_in_shape}, overlap={overlap_shape}"
                                )

                            block_weight = np.ones(
                                [
                                    m - 2 * o
                                    for m, o in zip(model_out_shape, overlap_shape)
                                ],
                                dtype=np.float32,
                            )
                            if any(s <= 0 for s in block_weight.shape):
                                raise ValueError(
                                    "Overlap too large for blending weight. "
                                    f"tile={model_out_shape}, overlap={overlap_shape}"
                                )
                            block_weight = np.pad(
                                block_weight,
                                [(o + 1, o + 1) for o in overlap_shape],
                                "linear_ramp",
                            )[(slice(1, -1),) * image_dim]

                            applied = np.zeros(image.shape, dtype=np.float32)
                            sum_weight = np.zeros_like(applied)

                            blocks = list(
                                itertools.product(
                                    *[
                                        self._axis_starts(i, m, s)
                                        for i, m, s in zip(
                                            image.shape, model_in_shape, step_shape
                                        )
                                    ]
                                )
                            )

                            for chunk_idx in tqdm.trange(
                                0,
                                len(blocks),
                                batch_size,
                                disable=not verbose,
                                dynamic_ncols=True,
                            ):
                                rois = []
                                batch = np.zeros(
                                    (batch_size, num_in_ch, *model_in_shape),
                                    dtype=np.float32,
                                )

                                for b_idx, tl in enumerate(
                                    blocks[chunk_idx : chunk_idx + batch_size]
                                ):
                                    br = [
                                        min(t + m, i)
                                        for t, m, i in zip(
                                            tl, model_in_shape, image.shape
                                        )
                                    ]
                                    r1, r2 = zip(
                                        *[
                                            (slice(s, e), slice(0, e - s))
                                            for s, e in zip(tl, br)
                                        ]
                                    )
                                    roi = image[r1]
                                    if roi.shape != model_in_shape:
                                        pad_w = [
                                            (0, m - s)
                                            for m, s in zip(model_in_shape, roi.shape)
                                        ]
                                        roi = np.pad(roi, pad_w, "reflect")
                                    batch[b_idx, 0] = roi
                                    rois.append((r1, r2))

                                tensor = TOTENSOR_(batch)
                                pred = model(tensor.to(device)).cpu().numpy()

                                for b_idx in range(len(rois)):
                                    p = pred[b_idx, 0] * block_weight
                                    r1, r2 = rois[b_idx]
                                    applied[r1] += p[r2]
                                    sum_weight[r1] += block_weight[r2]

                            valid = sum_weight > 0
                            applied[valid] /= sum_weight[valid]
                            applied[~valid] = 0

                            out_file = os.path.join(
                                save_path, f"{img_file}_{model_file}"
                            )
                            tifffile.imwrite(
                                out_file, applied.astype("float32"), imagej=True
                            )
                            print("Saved -->", out_file)
