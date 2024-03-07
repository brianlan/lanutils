import attr
import numpy as np


@attr.s
class RunLengthEncoder:
    direction = attr.ib(default="downward_then_rightward")

    @direction.validator
    def direction_validator(self, attribute, value):
        assert value in ["downward_then_rightward", "rightward_then_downward"]

    def encode(self, mask):
        """
        :param mask: 2-dimensional ndarray with unique values 0 and 1.
        :return: encoded sequence, the very first pixel (mask[0, 0]) idx is encoded as 1.
        """
        assert mask.ndim == 2
        assert set(np.unique(mask).tolist()) == {0, 1}
        mask_flatten = (mask if self.direction == "rightward_then_downward" else mask.T).flatten()
        padded = np.insert(np.insert(mask_flatten, len(mask_flatten), 0), 0, 0)  # pad 0 at the beginning and the end.
        pix_grp_key_pos, = np.where(padded[1:] != padded[:-1])
        grp_start_pos = pix_grp_key_pos[::2] + 1
        npix_in_grp = pix_grp_key_pos[1::2] - pix_grp_key_pos[::2]
        encoded = np.vstack((grp_start_pos, npix_in_grp)).transpose().flatten().tolist()
        return encoded

    def decode(self, encoded_sequence, target_size):
        """
        :param encoded_sequence: the very first pixel (mask[0, 0]) idx is encoded as 1.
        :param target_size: tuple (width, height)
        :return: 2d ndarray with unique values 0 and 1.
        """
        assert len(target_size) == 2
        pix_pos = np.concatenate(([
            range(st_pos, st_pos + npix) for st_pos, npix in zip(encoded_sequence[::2], encoded_sequence[1::2])
        ]))
        operating_size = target_size if self.direction == "rightward_then_downward" else target_size[::-1]
        mask = np.zeros(operating_size[::-1], dtype=np.uint8)
        row_idx = (pix_pos - 1) // operating_size[0]  # minus 1 due to encoded_sequence is starting with 1.
        col_idx = (pix_pos - 1) % operating_size[0]
        mask[row_idx, col_idx] = 1
        return mask if self.direction == "rightward_then_downward" else mask.T
